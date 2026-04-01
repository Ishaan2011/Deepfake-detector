from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("deepfake_api")

app = FastAPI(title="Deepfake Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

CURRENT_DIR = Path(__file__).parent.absolute()
RUN_MODELS_PATH = CURRENT_DIR / "run_models.py"
DOWNLOAD_DIR = CURRENT_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

YOUTUBE_PATTERN = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/(watch\?.*v=|shorts/)|youtu\.be/)[A-Za-z0-9_-]{6,}"
)
MESO_SCORE_PATTERN = re.compile(r"MesoNet Score:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")
D3_SCORE_PATTERN = re.compile(r"Temp-D3 Score:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
MAX_JOB_LOG_LINES = 1000
DOWNLOAD_LOG_STEP_PERCENT = 2.5


class AnalyzeRequest(BaseModel):
    youtube_url: str


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_PATTERN.match(url.strip()))


def _sigmoid(value: float) -> float:
    clamped = max(min(value, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-clamped))


def _compute_overall_verdict(
    mesonet_score: float | None, temp_d3_sigmoid_score: float
) -> tuple[str, float]:
    scores = [temp_d3_sigmoid_score]
    if mesonet_score is not None:
        scores.append(mesonet_score)
    combined_score = float(sum(scores) / len(scores))
    verdict = "fake" if combined_score >= 0.5 else "real"
    return verdict, combined_score


def _format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "0B"
    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while size >= 1024.0 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    return f"{size:.2f}{units[index]}"


def _new_job(job_id: str, youtube_url: str) -> dict[str, Any]:
    now = _utcnow_iso()
    return {
        "job_id": job_id,
        "youtube_url": youtube_url,
        "status": "queued",
        "phase": "Queued",
        "download_percent": 0.0,
        "logs": [],
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
    }


def _append_job_log(job_id: str, message: str, level: str = "INFO") -> None:
    timestamp = _utcnow_iso()
    entry = {"ts": timestamp, "level": level, "message": message}
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        logs = job["logs"]
        logs.append(entry)
        if len(logs) > MAX_JOB_LOG_LINES:
            del logs[:-MAX_JOB_LOG_LINES]
        job["updated_at"] = timestamp

    message_with_job = f"job={job_id} | {message}"
    if level == "ERROR":
        LOGGER.error(message_with_job)
    elif level == "WARNING":
        LOGGER.warning(message_with_job)
    else:
        LOGGER.info(message_with_job)


def _update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = _utcnow_iso()


def _get_job_snapshot(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        snapshot = dict(job)
        snapshot["logs"] = list(job["logs"])
        return snapshot


def _download_youtube_video(url: str, job_id: str) -> Path:
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with: pip install yt-dlp"
        ) from exc

    _append_job_log(job_id, f"[Download] Initializing downloader for URL: {url}")
    video_id = uuid.uuid4().hex
    output_template = str(DOWNLOAD_DIR / f"{video_id}.%(ext)s")
    progress_state = {"last_logged_percent": -DOWNLOAD_LOG_STEP_PERCENT}

    def progress_hook(progress_data: dict[str, Any]) -> None:
        status = progress_data.get("status")

        if status == "downloading":
            downloaded_bytes = float(progress_data.get("downloaded_bytes") or 0.0)
            total_bytes = progress_data.get("total_bytes") or progress_data.get(
                "total_bytes_estimate"
            )
            total_bytes_val = float(total_bytes) if total_bytes else 0.0
            percent = (
                (downloaded_bytes / total_bytes_val) * 100.0
                if total_bytes_val > 0
                else 0.0
            )
            speed = progress_data.get("speed")
            eta = progress_data.get("eta")

            _update_job(
                job_id,
                status="downloading",
                phase="Downloading video...",
                download_percent=round(percent, 2),
            )

            should_emit_log = (
                percent - progress_state["last_logged_percent"]
            ) >= DOWNLOAD_LOG_STEP_PERCENT or percent >= 99.9
            if should_emit_log:
                speed_txt = (
                    f"{_format_bytes(speed)}/s"
                    if isinstance(speed, (int, float)) and speed > 0
                    else "unknown"
                )
                eta_txt = (
                    f"{int(eta)}s"
                    if isinstance(eta, (int, float)) and eta >= 0
                    else "unknown"
                )
                total_txt = _format_bytes(total_bytes_val) if total_bytes_val > 0 else "?"
                _append_job_log(
                    job_id,
                    (
                        f"[Download] {percent:.1f}% "
                        f"({ _format_bytes(downloaded_bytes) }/{total_txt}) "
                        f"speed={speed_txt}, eta={eta_txt}"
                    ),
                )
                progress_state["last_logged_percent"] = percent
        elif status == "finished":
            filename = progress_data.get("filename")
            name = Path(filename).name if filename else "downloaded file"
            _update_job(
                job_id,
                status="downloading",
                phase="Download complete. Finalizing file...",
                download_percent=100.0,
            )
            _append_job_log(job_id, f"[Download] Stream download complete: {name}")

    ydl_opts: dict[str, Any] = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress_hook],
    }

    _append_job_log(job_id, "[Download] Starting yt-dlp extraction and download.")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = ydl.prepare_filename(info)

    candidate = Path(downloaded_path)
    if candidate.suffix.lower() != ".mp4":
        mp4_candidate = candidate.with_suffix(".mp4")
        if mp4_candidate.exists():
            return mp4_candidate

    if candidate.exists():
        return candidate

    mp4_fallback = DOWNLOAD_DIR / f"{video_id}.mp4"
    if mp4_fallback.exists():
        return mp4_fallback

    raise RuntimeError("Video download failed: output file not found.")


def _run_models_script_with_logs(
    video_path: Path, job_id: str
) -> tuple[float | None, float | None, str]:
    command = [
        sys.executable,
        "-u",
        str(RUN_MODELS_PATH),
        str(video_path),
        "--verbose-status",
    ]
    _append_job_log(job_id, f"[Models] Launching process: {' '.join(command)}")
    _update_job(job_id, status="running_models", phase="Running model pipeline...")

    process = subprocess.Popen(
        command,
        cwd=str(CURRENT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Unable to capture run_models.py output stream.")

    meso_score: float | None = None
    d3_score: float | None = None
    output_lines: list[str] = []

    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output_lines.append(line)
        _append_job_log(job_id, f"[run_models] {line}")

        if line.startswith("[MesoNet]"):
            _update_job(job_id, status="running_mesonet", phase=line)
        elif line.startswith("[Temp-D3]"):
            _update_job(job_id, status="running_temp_d3", phase=line)

        meso_match = MESO_SCORE_PATTERN.search(line)
        if meso_match:
            meso_score = float(meso_match.group(1))
            _append_job_log(
                job_id, f"[Parser] Parsed MesoNet score: {meso_score:.6f}"
            )

        d3_match = D3_SCORE_PATTERN.search(line)
        if d3_match:
            d3_score = float(d3_match.group(1))
            _append_job_log(job_id, f"[Parser] Parsed Temp-D3 raw score: {d3_score:.6f}")

    return_code = process.wait()
    _append_job_log(job_id, f"[Models] run_models.py exited with code {return_code}.")
    combined_output = "\n".join(output_lines)

    if return_code != 0:
        tail = "\n".join(output_lines[-30:])
        raise RuntimeError(
            f"run_models.py failed with exit code {return_code}. Output tail:\n{tail}"
        )

    return meso_score, d3_score, combined_output


def _analysis_worker(job_id: str, youtube_url: str) -> None:
    started_perf = time.perf_counter()
    video_path: Path | None = None
    _update_job(job_id, status="started", phase="Worker started", started_at=_utcnow_iso())
    _append_job_log(job_id, "[Worker] Analysis worker started.")

    try:
        _append_job_log(job_id, "[Worker] Beginning download stage.")
        video_path = _download_youtube_video(youtube_url, job_id)
        _append_job_log(job_id, f"[Download] Local file ready: {video_path}")

        _append_job_log(job_id, "[Worker] Beginning inference stage.")
        meso_score, d3_raw_score, model_output = _run_models_script_with_logs(
            video_path, job_id
        )

        if d3_raw_score is None:
            raise RuntimeError(
                "Temp-D3 score not found in run_models.py output. "
                f"Output:\n{model_output}"
            )

        d3_sigmoid_score = _sigmoid(d3_raw_score)
        overall_verdict, combined_score = _compute_overall_verdict(
            meso_score, d3_sigmoid_score
        )
        elapsed_seconds = round(time.perf_counter() - started_perf, 2)

        result = {
            "video_file": video_path.name,
            "mesonet_score": None if meso_score is None else float(meso_score),
            "temp_d3_raw_score": float(d3_raw_score),
            "temp_d3_score": float(d3_sigmoid_score),
            "overall_verdict": overall_verdict,
            "processing_seconds": elapsed_seconds,
        }
        _update_job(
            job_id,
            status="completed",
            phase="Completed",
            result=result,
            error=None,
            finished_at=_utcnow_iso(),
            download_percent=100.0,
        )
        _append_job_log(
            job_id,
            (
                f"[Result] Completed in {elapsed_seconds}s | "
                f"MesoNet={result['mesonet_score']} | "
                f"Temp-D3 raw={result['temp_d3_raw_score']:.6f} | "
                f"Temp-D3 sigmoid={result['temp_d3_score']:.6f} | "
                f"Combined={combined_score:.6f} | "
                f"Verdict={overall_verdict.upper()}"
            ),
        )
    except Exception as exc:
        elapsed_seconds = round(time.perf_counter() - started_perf, 2)
        _update_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(exc),
            finished_at=_utcnow_iso(),
        )
        _append_job_log(
            job_id, f"[Error] Job failed after {elapsed_seconds}s: {exc}", level="ERROR"
        )
    finally:
        if video_path and video_path.exists():
            try:
                os.remove(video_path)
                _append_job_log(job_id, f"[Cleanup] Removed temporary file: {video_path}")
            except OSError as cleanup_error:
                _append_job_log(
                    job_id,
                    f"[Cleanup] Failed to remove temporary file {video_path}: {cleanup_error}",
                    level="WARNING",
                )
        _append_job_log(job_id, "[Worker] Analysis worker finished.")


@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    youtube_url = payload.youtube_url.strip()
    if not youtube_url:
        raise HTTPException(status_code=400, detail="Please provide a YouTube URL.")

    if not _is_valid_youtube_url(youtube_url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = _new_job(job_id, youtube_url)

    _append_job_log(job_id, f"[Request] Analyze request accepted for URL: {youtube_url}")
    worker = threading.Thread(
        target=_analysis_worker,
        args=(job_id, youtube_url),
        daemon=True,
        name=f"analyze-{job_id[:8]}",
    )
    worker.start()
    _append_job_log(job_id, f"[Worker] Thread started: {worker.name}")

    return {"job_id": job_id, "status_endpoint": f"/api/analyze/{job_id}"}


@app.get("/api/analyze/{job_id}")
def analyze_status(job_id: str):
    snapshot = _get_job_snapshot(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Job not found.")
    return snapshot
