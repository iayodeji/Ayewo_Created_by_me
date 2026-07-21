import io
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import partial
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from .app.inference import MalariaInferenceService
from .app.reporting import build_batch_pdf

# Keep only the latest reports in memory to prevent unbounded growth in this demo app.
MAX_BATCH_REPORTS = 500
MAX_BATCH_SIZE = 50
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def read_image(upload: UploadFile) -> Image.Image:
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{upload.filename}' is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file '{upload.filename}' exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}") from exc
    return image.convert("RGB")


def create_app(inference_service: MalariaInferenceService | None = None) -> FastAPI:
    """Create an app without loading the model until the server starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.inference_service = inference_service or MalariaInferenceService()
        app.state.batch_store = OrderedDict()
        app.state.batch_store_lock = Lock()
        yield

    app = FastAPI(title="Ayewo Backend", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def service(request: Request) -> MalariaInferenceService:
        try:
            return request.app.state.inference_service
        except AttributeError as exc:
            raise HTTPException(status_code=503, detail="Model is still starting up.") from exc

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        model = service(request)
        return {"status": "ok", "model_loaded": bool(model.model_loaded), "device": str(model.device)}


    @app.post("/predict/single")
    async def predict_single(
        request: Request,
        file: UploadFile = File(...),
        include_gradcam: bool = Query(True),
    ) -> dict[str, Any]:
        try:
            image = await read_image(file)
            result = await run_in_threadpool(
                partial(service(request).predict, image, include_gradcam=include_gradcam)
            )
            return {
                "result": result.result,
                "confidence": result.confidence,
                "gradcam_image": result.gradcam_image,
                "low_confidence": result.low_confidence,
            }
        finally:
            await file.close()


    @app.post("/predict/batch")
    async def predict_batch(
        request: Request,
        files: list[UploadFile] = File(...),
        include_gradcam: bool = Query(True),
    ) -> dict[str, Any]:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(status_code=400, detail=f"Maximum {MAX_BATCH_SIZE} files per batch.")
        try:
            images = [await read_image(upload) for upload in files]
            predictions = await run_in_threadpool(
                partial(service(request).predict_batch, images, include_gradcam=include_gradcam)
            )
        finally:
            for upload in files:
                await upload.close()

        batch_id = str(uuid.uuid4())
        response_rows = []
        report_rows = []
        for upload, result in zip(files, predictions, strict=True):
            row = {
                "filename": upload.filename,
                "result": result.result,
                "confidence": result.confidence,
                "gradcam_image": result.gradcam_image,
                "low_confidence": result.low_confidence,
            }
            response_rows.append(row)
            report_rows.append({key: row[key] for key in ("filename", "result", "confidence")})

        with request.app.state.batch_store_lock:
            request.app.state.batch_store[batch_id] = report_rows
            while len(request.app.state.batch_store) > MAX_BATCH_REPORTS:
                request.app.state.batch_store.popitem(last=False)
        return {"batch_id": batch_id, "results": response_rows}


    @app.get("/report/{batch_id}")
    def get_report(request: Request, batch_id: str) -> StreamingResponse:
        with request.app.state.batch_store_lock:
            rows = request.app.state.batch_store.get(batch_id)
        if rows is None:
            raise HTTPException(status_code=404, detail="Batch report not found.")
        pdf_content = build_batch_pdf(batch_id, rows)
        return StreamingResponse(
            io.BytesIO(pdf_content),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="batch_report_{batch_id}.pdf"'},
        )

    return app


app = create_app()
