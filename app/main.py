"""Local, single-process API for recorded-video detection jobs."""
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.detection import process_video

app = FastAPI(title="Object Detection Backend", version="0.1.0")
STORAGE = Path(os.getenv("DETECTION_STORAGE", "storage"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
jobs = {}
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)


def run_job(job_id: str, source: Path, confidence: float) -> None:
    with lock:
        jobs[job_id]["status"] = "processing"
    folder = source.parent
    try:
        process_video(source, folder / "annotated.mp4", folder / "detections.jsonl", confidence)
        with lock:
            jobs[job_id]["status"] = "completed"
    except Exception as exc:
        for name in ("annotated.mp4", "detections.jsonl"):
            (folder / name).unlink(missing_ok=True)
        with lock:
            jobs[job_id].update(status="failed", error=str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/jobs", status_code=202)
def create_job(file: UploadFile = File(...), confidence: float = Form(0.25)):
    if not 0 < confidence <= 1:
        raise HTTPException(422, "confidence must be greater than 0 and at most 1")
    job_id = uuid.uuid4().hex
    folder = STORAGE / job_id
    folder.mkdir(parents=True, exist_ok=False)
    source = folder / "input.video"
    try:
        size = 0
        with source.open("wb") as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Video exceeds upload limit")
                target.write(chunk)
        if not size:
            raise HTTPException(400, "Video is empty")
        with lock:
            jobs[job_id] = {"id": job_id, "status": "queued", "error": None}
        executor.submit(run_job, job_id, source, confidence)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        file.file.close()
    return {"id": job_id, "status": "queued"}


def get_job(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job.copy()


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if job["status"] == "completed":
        job["video_url"] = f"/jobs/{job_id}/video"
        job["detections_url"] = f"/jobs/{job_id}/detections"
    return job


def result_file(job_id: str, filename: str, media_type: str):
    job = get_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(409, "Job is not complete")
    return FileResponse(STORAGE / job_id / filename, media_type=media_type, filename=filename)


@app.get("/jobs/{job_id}/video")
def video(job_id: str):
    return result_file(job_id, "annotated.mp4", "video/mp4")


@app.get("/jobs/{job_id}/detections")
def detections(job_id: str):
    return result_file(job_id, "detections.jsonl", "application/x-ndjson")
