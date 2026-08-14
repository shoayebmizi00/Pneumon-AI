from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.image_utils import InvalidImage
from backend.model_service import ModelUnavailable, PneumoniaModelService


app = FastAPI(title="PneumoAI API", version="1.0.0", description="Research-only chest X-ray classification API")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
service = PneumoniaModelService(settings.model_path, settings.metadata_path)
request_times: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def security_headers_and_rate_limit(request: Request, call_next):
    if request.url.path == "/api/predict":
        now = time.monotonic()
        key = request.client.host if request.client else "unknown"
        bucket = request_times[key]
        while bucket and bucket[0] < now - 60:
            bucket.popleft()
        if len(bucket) >= settings.requests_per_minute:
            return JSONResponse(status_code=429, content={"detail": "Too many requests. Please wait and try again."})
        bucket.append(now)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/api/health")
def health() -> dict:
    return {"status": "ready" if service.ready else "degraded", "model_loaded": service.ready, "model": service.metadata.get("model"), "detail": service.load_error}


@app.get("/api/performance")
def performance() -> dict:
    if not service.metadata:
        raise HTTPException(status_code=503, detail="Model metadata is not available.")
    return {"model": service.metadata.get("model"), "image_size": service.metadata.get("image_size"),
            "validation_metrics": service.metadata.get("validation_metrics"), "test_metrics": service.metadata.get("test_metrics")}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), gradcam: bool = False) -> dict:
    data = await file.read(settings.max_upload_size + 1)
    await file.close()
    if not data:
        raise HTTPException(status_code=400, detail="No image data was uploaded.")
    if len(data) > settings.max_upload_size:
        raise HTTPException(status_code=413, detail=f"Image is too large. Maximum size is {settings.max_upload_size // (1024 * 1024)} MB.")
    try:
        return service.predict(data, file.content_type, with_gradcam=gradcam)
    except InvalidImage as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ModelUnavailable:
        raise HTTPException(status_code=503, detail="The trained model is not available. Complete model training before requesting predictions.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed safely. Please try a different image.") from exc
