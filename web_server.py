"""Serve the scan workspace and transient, size-limited classifier requests."""
import asyncio
import io
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent / "web"
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000


def create_server(model):
    from mri_model import classify

    api = FastAPI(docs_url=None, redoc_url=None)
    slots = asyncio.Semaphore(1)

    @api.get("/")
    def home():
        return FileResponse(ROOT / "index.html", media_type="text/html; charset=utf-8")

    @api.get("/api/health")
    def health():
        return {"app": "neuroscope-mri", "ready": model is not None}

    def evaluate(data):
        try:
            with Image.open(io.BytesIO(data), formats=["JPEG", "PNG"]) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise HTTPException(413, "Use an image with at most 16 million pixels.")
                image.load()
                scores = classify(model, image)
        except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, "Upload a readable PNG or JPEG image.") from exc
        return {"classification": max(scores, key=scores.get), "scores": scores}

    @api.post("/api/predict")
    async def predict(request: Request):
        if model is None:
            raise HTTPException(503, "Model unavailable. Check the server log and restart.")
        if request.headers.get("content-type", "").split(";")[0] not in {"image/jpeg", "image/png"}:
            raise HTTPException(415, "Choose a PNG or JPEG image.")
        async with slots:
            data = bytearray()
            async for chunk in request.stream():
                if len(data) + len(chunk) > MAX_BYTES:
                    raise HTTPException(413, "Choose a file of 10 MiB or less.")
                data.extend(chunk)
            if not data:
                raise HTTPException(422, "Choose an image first.")
            try:
                result = await run_in_threadpool(evaluate, data)
            except HTTPException:
                raise
            except Exception:
                logging.getLogger(__name__).exception("MRI inference failed")
                raise HTTPException(500, "Analysis failed. Try again or check the server log.")
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    api.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")

    @api.exception_handler(404)
    async def missing(request, exc):
        return HTMLResponse('<meta charset="utf-8"><title>Page missing | NeuroScope</title><main style="font:1.2rem system-ui;padding:8%;color:#123149"><h1>NeuroScope</h1><p>This page could not be found.</p><a href="/">Return to the MRI workspace</a></main>', status_code=404, media_type="text/html; charset=utf-8")

    return api

