import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


logger = logging.getLogger(__name__)

LABELS = ["Parasitized", "Uninfected"]
LOW_CONFIDENCE_THRESHOLD = 75.0


@dataclass
class PredictionResult:
    result: str
    confidence: float
    gradcam_image: str | None
    low_confidence: bool


class MalariaInferenceService:
    def __init__(self, model_path: str = "malaria_model.pth") -> None:
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()
        self.weights_loaded_from_file = self._try_load_weights()
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.target_layers = [self.model.features[-1][0]]
        # Grad-CAM registers hooks and performs a backward pass.  Serialising it
        # keeps concurrent API requests from mutating the shared model together.
        self._gradcam_lock = Lock()
        self.model_loaded = self.weights_loaded_from_file or self.used_imagenet_pretrained

    def _build_model(self) -> nn.Module:
        self.used_imagenet_pretrained = False
        # A locally trained checkpoint contains every model parameter, so loading
        # ImageNet weights first only adds startup latency and may require network.
        # Preserve the fallback behaviour when no checkpoint has been supplied.
        if self.model_path.exists():
            model = mobilenet_v2(weights=None)
        else:
            try:
                model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
                self.used_imagenet_pretrained = True
            except (RuntimeError, OSError, ValueError):
                model = mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        return model.to(self.device)

    def _try_load_weights(self) -> bool:
        if not self.model_path.exists():
            return False
        try:
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch versions before ``weights_only`` support.
            state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return True

    def _make_gradcam(self, input_tensor: torch.Tensor, original_image: Image.Image) -> str:
        # Keep the standard screening path import-light; Grad-CAM is only needed
        # for callers that explicitly request an explanation image.
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        rgb_image = np.asarray(original_image.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
        with self._gradcam_lock, GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        output = io.BytesIO()
        Image.fromarray(cam_image).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")

    def predict(self, image: Image.Image, *, include_gradcam: bool = True) -> PredictionResult:
        return self.predict_batch([image], include_gradcam=include_gradcam)[0]

    def predict_batch(
        self, images: Sequence[Image.Image], *, include_gradcam: bool = True
    ) -> list[PredictionResult]:
        """Run one vectorised classifier pass for an uploaded image batch."""
        if not images:
            return []
        transformed = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(transformed), dim=1)

        results: list[PredictionResult] = []
        for index, image in enumerate(images):
            predicted_index = int(torch.argmax(probabilities[index]).item())
            confidence = float(probabilities[index, predicted_index].item() * 100.0)
            gradcam_image = None
            if include_gradcam:
                try:
                    gradcam_image = self._make_gradcam(transformed[index : index + 1], image)
                except (ImportError, OSError, RuntimeError):
                    # A screening result remains useful if the optional
                    # explanation dependency is unavailable on a host.
                    logger.exception("Grad-CAM generation failed; returning prediction without attention map.")
            results.append(
                PredictionResult(
                    result=LABELS[predicted_index],
                    confidence=round(confidence, 2),
                    gradcam_image=gradcam_image,
                    low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
                )
            )
        return results
