import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


LABELS = ["Parasitized", "Uninfected"]
LOW_CONFIDENCE_THRESHOLD = 75.0


@dataclass
class PredictionResult:
    result: str
    confidence: float
    gradcam_image: str
    low_confidence: bool


class MalariaInferenceService:
    def __init__(self, model_path: str = "malaria_model.pth") -> None:
        self.device = torch.device("cpu")
        self.model_path = Path(model_path)
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
        self.model_loaded = self.weights_loaded_from_file or self.used_imagenet_pretrained

    def _build_model(self) -> nn.Module:
        self.used_imagenet_pretrained = False
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
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return True

    def _make_gradcam(self, input_tensor: torch.Tensor, original_image: Image.Image) -> str:
        rgb_image = np.asarray(original_image.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
        with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0, :]
        cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        output = io.BytesIO()
        Image.fromarray(cam_image).save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")

    def predict(self, image: Image.Image) -> PredictionResult:
        transformed = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(transformed)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_index].item() * 100.0)
        gradcam_image = self._make_gradcam(transformed, image)
        return PredictionResult(
            result=LABELS[predicted_index],
            confidence=round(confidence, 2),
            gradcam_image=gradcam_image,
            low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
        )
