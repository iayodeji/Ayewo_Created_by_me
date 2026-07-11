import io
import uuid
from collections import OrderedDict
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from app.inference import MalariaInferenceService
from app.reporting import build_batch_pdf

app = FastAPI(title="Ayewo Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_service = MalariaInferenceService()
# Keep only the latest reports in memory to prevent unbounded growth in this demo app.
MAX_BATCH_REPORTS = 500
MAX_BATCH_SIZE = 50
batch_store: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
batch_store_lock = Lock()


async def read_image(upload: UploadFile) -> Image.Image:
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{upload.filename}' is empty.")
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}") from exc
    return image.convert("RGB")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": bool(inference_service.model_loaded)}


@app.post("/predict/single")
async def predict_single(file: UploadFile = File(...)) -> dict[str, Any]:
    image = await read_image(file)
    result = inference_service.predict(image)
    return {
        "result": result.result,
        "confidence": result.confidence,
        "gradcam_image": result.gradcam_image,
        "low_confidence": result.low_confidence,
    }


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_BATCH_SIZE} files per batch.")

    batch_id = str(uuid.uuid4())
    response_rows = []
    report_rows = []
    for upload in files:
        image = await read_image(upload)
        result = inference_service.predict(image)
        row = {
            "filename": upload.filename,
            "result": result.result,
            "confidence": result.confidence,
            "gradcam_image": result.gradcam_image,
            "low_confidence": result.low_confidence,
        }
        response_rows.append(row)
        report_rows.append(
            {
                "filename": upload.filename,
                "result": result.result,
                "confidence": result.confidence,
            }
        )

    with batch_store_lock:
        batch_store[batch_id] = report_rows
        while len(batch_store) > MAX_BATCH_REPORTS:
            batch_store.popitem(last=False)
    return {"batch_id": batch_id, "results": response_rows}


@app.get("/report/{batch_id}")
def get_report(batch_id: str) -> StreamingResponse:
    with batch_store_lock:
        rows = batch_store.get(batch_id)
    if rows is None:
        raise HTTPException(status_code=404, detail="Batch report not found.")
    pdf_content = build_batch_pdf(batch_id, rows)
    return StreamingResponse(
        io.BytesIO(pdf_content),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="batch_report_{batch_id}.pdf"'},
    )
