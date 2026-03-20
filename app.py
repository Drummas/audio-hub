import os
import uuid
import json
import shutil
import psutil
import platform
import datetime
import threading
import tempfile
import logging
import sys
import time
import math
import asyncio
import httpx
import numpy as np
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from logging.handlers import RotatingFileHandler
from collections import deque
from queue import Queue

import httpx
from TTS.api import TTS
from pydub import AudioSegment
import matplotlib.pyplot as plt

# -----------------------------
# Paths and constants
# -----------------------------
FILES_DIR = "/app/files"
BATCHES_DIR = "/app/batches"
WAVEFORMS_DIR = "/app/waveforms"
SPEAKERS_DIR = "/app/speakers"
CONFIG_PATH = "/app/config.json"
METADATA_PATH = "/app/metadata.json"
LOG_FILE = "/app/logs/app.log"
PROGRESS = {}      # batch_id -> dict
WS_CONNECTIONS = {}  # batch_id -> set of websockets
# Progress tracking
PROGRESS_BATCH: dict[str, dict] = {}
PROGRESS_SINGLE: dict[str, dict] = {}
WS_BATCH: dict[str, set[WebSocket]] = {}
WS_SINGLE: dict[str, set[WebSocket]] = {}
WS_JOBCOUNT = set()
CPU_HISTORY = deque(maxlen=60)
RAM_HISTORY = deque(maxlen=60)
NET_LAST = psutil.net_io_counters()
DISK_LAST = psutil.disk_io_counters()
FASTER_WHISPER_URL = os.getenv(
    "FASTER_WHISPER_URL",
    "http://faster-whisper:10300/inference"
)
CHUNK_QUEUE: Queue = Queue()

os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(BATCHES_DIR, exist_ok=True)
os.makedirs(WAVEFORMS_DIR, exist_ok=True)
os.makedirs(SPEAKERS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

DEFAULT_CONFIG = {
    "whisper_model": os.getenv("WHISPER_MODEL", "large-v3"),
    "whisper_timeout": os.getenv("WHISPER_TIMEOUT", 600),
    "whisper_model_override": os.getenv("WHISPER_MODEL_OVERRIDE", None),
    "auto_model_enabled": os.getenv("AUTO_MODEL_ENABLED", True),    
    "xtts_language": os.getenv("XTTS_LANGUAGE", "en"),
    "xtts_voice": os.getenv("XTTS_VOICE", "default"),
    "default_translation_language": os.getenv("DEFAULT_TRANSLATION_LANGUAGE", "pt"),
}

# -----------------------------
# App and templates
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/waveforms", StaticFiles(directory=WAVEFORMS_DIR), name="waveforms")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Config helpers
# -----------------------------
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_PATH) as f:
        data = json.load(f)
    return {**DEFAULT_CONFIG, **data}

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

# -----------------------------
# Metadata helpers
# -----------------------------
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)

def save_metadata(meta):
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)

# -----------------------------
# Logging helpers
# -----------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

logger = logging.getLogger("audio-hub")
logger.setLevel(logging.INFO)

# Console
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(JsonFormatter())
logger.addHandler(ch)

# File (rotating)
fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
fh.setFormatter(JsonFormatter())
logger.addHandler(fh)

class TimeoutConfig(BaseModel):
    timeout: int

class ModelConfig(BaseModel):
    auto_enabled: bool
    override_model: str | None

class ChunkJob(BaseModel):
    file_id: str
    language: str
    task: str  # "transcribe" or "translate"

def get_system_info():
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)
    uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_percent": cpu_percent,
        "memory_percent": mem.percent,
        "memory_used": mem.used,
        "memory_total": mem.total,
        "uptime": str(uptime),  # must be string for JSON
        "cpu_count": psutil.cpu_count(),
    }

# -----------------------------
# WebSocket + SSE helpers
# -----------------------------
async def notify_batch_progress(batch_id: str):
    data = PROGRESS_BATCH.get(batch_id, {})
    dead = []
    for ws in WS_BATCH.get(batch_id, set()):
        try:
            await ws.send_json(data)
        except WebSocketDisconnect:
            dead.append(ws)
    for ws in dead:
        WS_BATCH[batch_id].discard(ws)
    # also notify jobcount listeners
    await notify_jobcount()
    
# -----------------------------
# Single-file helpers
# -----------------------------
async def notify_single_progress(progress_id: str):
    data = PROGRESS_SINGLE.get(progress_id, {})
    dead = []
    for ws in WS_SINGLE.get(progress_id, set()):
        try:
            await ws.send_json(data)
        except WebSocketDisconnect:
            dead.append(ws)
    for ws in dead:
        WS_SINGLE[progress_id].discard(ws)
    await notify_jobcount()

def process_single(progress_id: str, tmp_path: str, mode: str, language: str, target_language: str, response_format: str):
    PROGRESS_SINGLE[progress_id] = {
        "status": "running",
        "total": 1,
        "completed": 0,
        "mode": mode,
        "language": language,
        "target_language": target_language,
        "started_at": time.time(),
    }
    try:
        asyncio.run(notify_single_progress(progress_id))
    except RuntimeError:
        pass

    task = "translate" if mode == "translate" else "transcribe"
    lang = target_language if task == "translate" else language

    try:
        with httpx.Client(timeout=600) as client:
            logger.info(json.dumps({"event": "single_start", "progress_id": progress_id, "task": task, "language": lang}))
            resp = client.post(
                FASTER_WHISPER_URL,
                files={"audio_file": open(tmp_path, "rb")},
                data={"task": task, "language": lang},
            )
            logger.info(json.dumps({"event": "single_done", "progress_id": progress_id, "status_code": resp.status_code}))
        data = resp.json()
    except Exception as e:
        logger.exception("single_error")
        PROGRESS_SINGLE[progress_id]["status"] = "error"
        PROGRESS_SINGLE[progress_id]["error"] = str(e)
        try:
            asyncio.run(notify_single_progress(progress_id))
        except RuntimeError:
            pass
        return

    PROGRESS_SINGLE[progress_id]["completed"] = 1
    PROGRESS_SINGLE[progress_id]["status"] = "completed"
    PROGRESS_SINGLE[progress_id]["finished_at"] = time.time()
    PROGRESS_SINGLE[progress_id]["result"] = {
        "text": data.get("text", ""),
        "segments": data.get("segments", []),
        "response_format": response_format,
    }
    try:
        asyncio.run(notify_single_progress(progress_id))
    except RuntimeError:
        pass

# -----------------------------
# XTTS v2
# -----------------------------
logger.info("Loading XTTS v2 model…")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
logger.info("Loading XTTS v2 model…")

def get_speaker_ref(voice: str):
    path = os.path.join(SPEAKERS_DIR, f"{voice}.wav")
    return path if os.path.exists(path) else None

# -----------------------------
# Waveform helper
# -----------------------------
def generate_waveform_png(audio_path: str) -> str:
    audio = AudioSegment.from_file(audio_path)
    samples = audio.get_array_of_samples()
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(samples, linewidth=0.5, color="#80cbc4")
    ax.set_axis_off()
    out_path = os.path.join(WAVEFORMS_DIR, f"{uuid.uuid4()}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path

# -----------------------------
# SRT helper
# -----------------------------
def segments_to_srt(segments):
    def fmt(t):
        ms = int((t - int(t)) * 1000)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(
            f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{seg['text'].strip()}\n"
        )
    return "\n".join(lines)

# -----------------------------
# Jobcount notification helper
# -----------------------------
async def notify_jobcount():
    state = {
        "single": list(PROGRESS_SINGLE.keys()),
        "batch": list(PROGRESS_BATCH.keys())
    }
    dead = []
    for ws in WS_JOBCOUNT:
        try:
            await ws.send_json(state)
        except WebSocketDisconnect:
            dead.append(ws)
    for ws in dead:
        WS_JOBCOUNT.discard(ws)

# -----------------------------
# Model helper
# -----------------------------
def get_effective_whisper_model():
    if CONFIG["whisper_model_override"]:
        return CONFIG["whisper_model_override"]
    return CONFIG["whisper_model"]

# -----------------------------
# Disk + model stats
# -----------------------------
def get_disk_usage(path):
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": path,
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent": round(usage.used / usage.total * 100, 2)
        }
    except Exception as e:
        return {"path": path, "error": str(e)}

def get_dir_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
            except:
                pass
    return total

def get_cache_health(path):
    if not os.path.exists(path):
        return {"exists": False, "status": "missing"}

    size = get_dir_size(path)
    files = sum(len(files) for _, _, files in os.walk(path))

    if size < 10_000_000:  # <10MB means incomplete XTTS download
        status = "incomplete"
    else:
        status = "healthy"

    return {
        "exists": True,
        "status": status,
        "files": files,
        "size": size
    }

    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)
    uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_percent": cpu_percent,
        "memory_percent": mem.percent,
        "memory_used": mem.used,
        "memory_total": mem.total,
        "uptime": str(uptime),
        "cpu_count": psutil.cpu_count(),
    }

# -----------------------------
# Progress broadcast
# -----------------------------
async def notify_progress(batch_id: str):
    data = PROGRESS.get(batch_id, {})
    dead = []
    for ws in WS_CONNECTIONS.get(batch_id, set()):
        try:
            await ws.send_json(data)
        except WebSocketDisconnect:
            dead.append(ws)
    for ws in dead:
        WS_CONNECTIONS[batch_id].discard(ws)

# -----------------------------
# Model recommendation
# -----------------------------
def estimate_recommended_model(cpu_count: int, tokens_per_second: float, has_gpu: bool):
    # Very simple heuristic
    if has_gpu:
        if tokens_per_second > 50:
            return "large-v3"
        elif tokens_per_second > 30:
            return "medium"
        else:
            return "small"
    else:
        if tokens_per_second > 30:
            return "medium"
        elif tokens_per_second > 20:
            return "small"
        elif tokens_per_second > 10:
            return "base"
        else:
            return "tiny"

# -----------------------------
# Timeout config
# -----------------------------
async def set_timeout(cfg: TimeoutConfig):
    CONFIG["whisper_timeout"] = max(60, min(cfg.timeout, 7200))
    return {"status": "ok", "timeout": CONFIG["whisper_timeout"]}


# -----------------------------
# Model config endpoint
# -----------------------------
@app.post("/api/set_model_config")
async def set_model_config(cfg: ModelConfig):
    CONFIG["auto_model_enabled"] = cfg.auto_enabled
    CONFIG["whisper_model_override"] = cfg.override_model
    return {"status": "ok", "config": CONFIG}

# -----------------------------
# Speedtest endpoint
# -----------------------------
@app.post("/api/model_speedtest")
async def model_speedtest():
    # synthetic 5s sine wave @ 16kHz mono
    sr = 16000
    duration = 5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    audio_bytes = audio.astype("float32").tobytes()

    url = os.getenv("FASTER_WHISPER_URL", "http://faster-whisper:10300/inference")

    start = time.time()
    with httpx.Client(timeout=CONFIG["whisper_timeout"]) as client:
        resp = client.post(
            url,
            files={"audio_file": ("test.raw", audio_bytes, "application/octet-stream")},
            data={"task": "transcribe", "language": "en"},
        )
    elapsed = time.time() - start

    # crude tokens/sec estimate: assume ~4 tokens/sec of audio
    tokens = duration * 4
    tps = tokens / elapsed if elapsed > 0 else 0.0

    sysinfo = get_system_info()
    has_gpu = False  # extend later if you want GPU detection

    recommended = estimate_recommended_model(sysinfo["cpu_count"], tps, has_gpu)

    return {
        "model": CONFIG["whisper_model"],
        "processing_seconds": elapsed,
        "tokens_per_second": tps,
        "estimated_minutes_per_minute": (elapsed / 60) / (duration / 60),
        "hardware": "CPU",
        "dtype": "float32",
        "recommended_model": recommended,
    }

# -----------------------------
# Chunked queue endpoint
# -----------------------------
@app.post("/api/transcribe_chunked")
async def transcribe_chunked(job: ChunkJob):
    # Here you’d:
    # 1. Locate file by file_id
    # 2. Split into chunks (e.g. via ffmpeg or pydub)
    # 3. Enqueue each chunk into CHUNK_QUEUE
    # 4. Process sequentially or via worker
    # 5. Merge results
    # For now, just return a placeholder
    return {"status": "not_implemented_yet"}

# -----------------------------
# WebSocket endpoints
# -----------------------------
@app.websocket("/ws/progress/{batch_id}")
async def ws_progress_batch(websocket: WebSocket, batch_id: str):
    await websocket.accept()
    WS_BATCH.setdefault(batch_id, set()).add(websocket)
    try:
        await websocket.send_json(PROGRESS_BATCH.get(batch_id, {"status": "unknown"}))
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        WS_BATCH[batch_id].discard(websocket)

@app.websocket("/ws/single/{progress_id}")
async def ws_progress_single(websocket: WebSocket, progress_id: str):
    await websocket.accept()
    WS_SINGLE.setdefault(progress_id, set()).add(websocket)
    try:
        await websocket.send_json(PROGRESS_SINGLE.get(progress_id, {"status": "unknown"}))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        WS_SINGLE[progress_id].discard(websocket)

# -----------------------------
# SSE endpoint
# -----------------------------
@app.get("/v1/progress/{batch_id}")
async def sse_progress_batch(batch_id: str):
    async def event_stream():
        last = None
        while True:
            state = PROGRESS_BATCH.get(batch_id)
            if state != last:
                last = state
                yield f"data: {json.dumps(state or {})}\n\n"
            if state and state.get("status") == "completed":
                break
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/single-progress/{progress_id}")
async def sse_progress_single(progress_id: str):
    async def event_stream():
        last = None
        while True:
            state = PROGRESS_SINGLE.get(progress_id)
            if state != last:
                last = state
                yield f"data: {json.dumps(state or {})}\n\n"
            if state and state.get("status") == "completed":
                break
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# -----------------------------
# Logs endpoint
# -----------------------------
@app.get("/logs")
async def get_logs(lines: int = 200):
    if not os.path.exists(LOG_FILE):
        return {"logs": []}
    with open(LOG_FILE, "r") as f:
        all_lines = f.readlines()
    tail = all_lines[-lines:]
    return {"logs": [l.rstrip("\n") for l in tail]}

# -----------------------------
# Jobcount endpoint
# -----------------------------
@app.websocket("/ws/jobcount")
async def ws_jobcount(websocket: WebSocket):
    await websocket.accept()
    WS_JOBCOUNT.add(websocket)
    try:
        await websocket.send_json({
            "single": list(PROGRESS_SINGLE.keys()),
            "batch": list(PROGRESS_BATCH.keys())
        })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        WS_JOBCOUNT.discard(websocket)

@app.get("/v1/jobcount")
async def sse_jobcount():
    async def event_stream():
        last = None
        while True:
            state = {
                "single": list(PROGRESS_SINGLE.keys()),
                "batch": list(PROGRESS_BATCH.keys())
            }
            if state != last:
                last = state
                yield f"data: {json.dumps(state)}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# -----------------------------
# System endpoint
# -----------------------------
@app.get("/ui/system", response_class=HTMLResponse)
async def ui_system(request: Request):
    tts_cache = "/root/.local/share/tts"
    hf_cache = "/root/.cache/huggingface"
    files_dir = "/app/files"
    batches_dir = "/app/batches"

    data = {
        "system": get_system_info(),
        "disk": [
            get_disk_usage("/"),
            get_disk_usage("/app"),
            get_disk_usage(tts_cache),
            get_disk_usage(hf_cache),
        ],
        "models": {
            "xtts_cache": get_cache_health(tts_cache),
            "hf_cache": get_cache_health(hf_cache),
        },
        "directories": {
            "files": get_dir_size(files_dir),
            "batches": get_dir_size(batches_dir),
        }
    }

    return templates.TemplateResponse("system.html", {
        "request": request,
        "data": data
    })

# -----------------------------
# Live system stats endpoint
# -----------------------------
@app.get("/api/system_stats")
async def system_stats():
    global NET_LAST, DISK_LAST, CPU_HISTORY, RAM_HISTORY

    # CPU
    cpu = psutil.cpu_percent(interval=None)
    CPU_HISTORY.append(cpu)

    # RAM
    mem = psutil.virtual_memory()
    RAM_HISTORY.append(mem.percent)

    # Disk I/O
    disk_now = psutil.disk_io_counters()
    disk_read = disk_now.read_bytes - DISK_LAST.read_bytes
    disk_write = disk_now.write_bytes - DISK_LAST.write_bytes
    DISK_LAST = disk_now

    # Network throughput
    net_now = psutil.net_io_counters()
    net_recv = net_now.bytes_recv - NET_LAST.bytes_recv
    net_sent = net_now.bytes_sent - NET_LAST.bytes_sent
    NET_LAST = net_now

    # Temperatures
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        temps = {}

    return {
        "system": get_system_info(),
        "cpu_history": list(CPU_HISTORY),
        "ram_history": list(RAM_HISTORY),
        "disk_io": {"read": disk_read, "write": disk_write},
        "network": {"recv": net_recv, "sent": net_sent},
        "temps": temps,
        "config": CONFIG,
    }

# -----------------------------
# Clear cache + rebuild models endpoint
# -----------------------------
@app.post("/api/clear_xtts_cache")
async def clear_xtts_cache():
    path = "/root/.local/share/tts"
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        return {"status": "ok", "message": "XTTS cache cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/clear_hf_cache")
async def clear_hf_cache():
    path = "/root/.cache/huggingface"
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        return {"status": "ok", "message": "HuggingFace cache cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/rebuild_models")
async def rebuild_models():
    try:
        # Clear XTTS + HF caches
        xtts = "/root/.local/share/tts"
        hf = "/root/.cache/huggingface"

        if os.path.exists(xtts):
            shutil.rmtree(xtts)
        if os.path.exists(hf):
            shutil.rmtree(hf)

        os.makedirs(xtts, exist_ok=True)
        os.makedirs(hf, exist_ok=True)

        # Force reload on next request
        logger.info("Model rebuild requested — caches cleared")

        return {"status": "ok", "message": "Model caches cleared. XTTS will rebuild on next use."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -----------------------------
# OpenAI-compatible endpoints
# -----------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "whisper-1", "object": "model", "owned_by": "local-faster-whisper"},
            {"id": "xtts-v2", "object": "model", "owned_by": "local-xtts"},
        ]
    }

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("json"),
    language: str = Form("auto"),
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    start = time.time()
    logger.info(f"transcribe_start filename={file.filename} language={language}")

    async with httpx.AsyncClient(timeout=1800) as client:
        resp = await client.post(
            FASTER_WHISPER_URL,
            files={"audio_file": open(path, "rb")},
            data={"task": "transcribe", "language": language},
        )    
    data = resp.json()
    
    elapsed = time.time() - start
    logger.info(f"transcribe_done filename={file.filename} status_code={resp.status_code} elapsed={elapsed:.2f}")     

    if response_format == "srt":
        return segments_to_srt(data["segments"])

    return {"text": data.get("text", "")}

@app.post("/v1/audio/translations")
async def translate_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("srt"),
    target_language: str = Form(os.getenv("DEFAULT_TRANSLATION_LANGUAGE", "pt")),
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    start = time.time()
    logger.info(f"translate_start filename={file.filename} language={language}")

    async with httpx.AsyncClient(timeout=1800) as client:
        resp = await client.post(
            FASTER_WHISPER_URL,
            files={"audio_file": open(path, "rb")},
            data={"task": "translate", "language": target_language},
        )
    data = resp.json()

    elapsed = time.time() - start
    logger.info(f"translate_done filename={file.filename} status_code={resp.status_code} elapsed={elapsed:.2f}")  

    if response_format == "srt":
        return segments_to_srt(data["segments"])

    return {"text": data.get("text", "")}

@app.post("/v1/audio/speech")
async def text_to_speech(
    input: str = Form(...),
    language: str = Form("en"),
    voice: str = Form("default"),
    model: str = Form("xtts-v2"),
):
    out_path = f"/tmp/{uuid.uuid4()}.wav"
    speaker_ref = get_speaker_ref(voice)

    tts.tts_to_file(
        text=input,
        file_path=out_path,
        language=language,
        speaker_wav=speaker_ref if speaker_ref else None,
    )

    return FileResponse(out_path, media_type="audio/wav")

@app.post("/v1/files")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    dest = os.path.join(FILES_DIR, file_id)

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    audio = AudioSegment.from_file(dest)
    duration_sec = len(audio) / 1000.0

    meta = load_metadata()
    meta[file_id] = {
        "filename": file.filename,
        "bytes": os.path.getsize(dest),
        "duration": duration_sec,
        "language": None,
    }
    save_metadata(meta)

    return {
        "id": file_id,
        "object": "file",
        "filename": file.filename,
        "bytes": os.path.getsize(dest),
    }

@app.get("/v1/files")
async def list_files():
    meta = load_metadata()
    data = []
    for fid, info in meta.items():
        data.append(
            {
                "id": fid,
                "object": "file",
                "bytes": info.get("bytes", 0),
                "filename": info.get("filename", ""),
            }
        )
    return {"data": data}

@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    path = os.path.join(FILES_DIR, file_id)
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path)

# -----------------------------
# Batch processing
# -----------------------------
def process_batch(batch_id, file_ids, task, language):
    batch_path = os.path.join(BATCHES_DIR, batch_id + ".json")
    results = []
    total = len(file_ids)

    PROGRESS_BATCH[batch_id] = {
        "status": "running",
        "total": total,
        "completed": 0,
        "current_file": None,
        "task": task,
        "language": language,
        "started_at": time.time(),
    }
    logger.info(json.dumps({"event": "batch_start", "batch_id": batch_id, "total": total, "task": task, "language": language}))

    for idx, fid in enumerate(file_ids, start=1):
        fpath = os.path.join(FILES_DIR, fid)
        if not os.path.exists(fpath):
            logger.warning(json.dumps({"event": "batch_missing_file", "batch_id": batch_id, "file_id": fid}))
            continue

        PROGRESS_BATCH[batch_id]["current_file"] = fid
        PROGRESS_BATCH[batch_id]["completed"] = idx - 1
        try:
            asyncio.run(notify_batch_progress(batch_id))
        except RuntimeError:
            pass

        with open(fpath, "rb") as audio:
            with httpx.Client(timeout=600) as client:
                logger.info(json.dumps({"event": "batch_file_start", "batch_id": batch_id, "file_id": fid}))
                resp = client.post(
                    FASTER_WHISPER_URL,
                    files={"audio_file": audio},
                    data={"task": task, "language": language},
                )
                logger.info(json.dumps({"event": "batch_file_done", "batch_id": batch_id, "file_id": fid, "status_code": resp.status_code}))

        results.append({"file_id": fid, "result": resp.json()})

        PROGRESS_BATCH[batch_id]["completed"] = idx
        try:
            asyncio.run(notify_batch_progress(batch_id))
        except RuntimeError:
            pass

    PROGRESS_BATCH[batch_id]["status"] = "completed"
    PROGRESS_BATCH[batch_id]["finished_at"] = time.time()
    try:
        asyncio.run(notify_batch_progress(batch_id))
    except RuntimeError:
        pass

    with open(batch_path, "w") as f:
        json.dump({"status": "completed", "results": results}, f)

    logger.info(json.dumps({"event": "batch_complete", "batch_id": batch_id, "total": total}))

@app.post("/v1/batches")
async def create_batch(
    file_ids: str = Form(...),
    task: str = Form("transcribe"),
    language: str = Form("auto"),
):
    batch_id = str(uuid.uuid4())
    batch_path = os.path.join(BATCHES_DIR, batch_id + ".json")

    with open(batch_path, "w") as f:
        json.dump({"status": "running"}, f)

    ids = [x.strip() for x in file_ids.split(",") if x.strip()]

    threading.Thread(
        target=process_batch, args=(batch_id, ids, task, language), daemon=True
    ).start()

    return {"id": batch_id, "status": "running"}

@app.get("/v1/batches/{batch_id}")
async def get_batch(batch_id: str):
    path = os.path.join(BATCHES_DIR, batch_id + ".json")
    if not os.path.exists(path):
        raise HTTPException(404)
    with open(path) as f:
        return json.load(f)

# -----------------------------
# UI routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def ui_home(request: Request):
    cfg = load_config()
    return templates.TemplateResponse("home.html", {"request": request, "cfg": cfg})

@app.post("/ui/transcribe", response_class=HTMLResponse)
async def ui_transcribe(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("transcribe"),
    language: str = Form("auto"),
    target_language: str = Form(os.getenv("DEFAULT_TRANSLATION_LANGUAGE", "pt")),
    response_format: str = Form("json"),
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    async with httpx.AsyncClient(timeout=1800) as client:
        if mode == "translate":
            resp = await client.post(
                FASTER_WHISPER_URL,
                files={"audio_file": open(path, "rb")},
                data={"task": "translate", "language": target_language},
            )
        else:
            resp = await client.post(
                FASTER_WHISPER_URL,
                files={"audio_file": open(path, "rb")},
                data={"task": "transcribe", "language": language},
            )

    data = resp.json()

    if response_format == "srt":
        content = f"<pre>{segments_to_srt(data['segments'])}</pre>"
    else:
        content = f"<pre>{json.dumps(data, indent=2, ensure_ascii=False)}</pre>"

    return templates.TemplateResponse(
        "result.html", {"request": request, "content": content}
    )

@app.post("/ui/transcribe_async")
async def ui_transcribe_async(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("transcribe"),
    language: str = Form("auto"),
    target_language: str = Form(os.getenv("DEFAULT_TRANSLATION_LANGUAGE", "pt")),
    response_format: str = Form("json"),
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    progress_id = str(uuid.uuid4())

    threading.Thread(
        target=process_single,
        args=(progress_id, path, mode, language, target_language, response_format),
        daemon=True,
    ).start()

    return {"progress_id": progress_id}

@app.post("/ui/batch", response_class=HTMLResponse)
async def ui_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    task: str = Form("transcribe"),
    language: str = Form("auto"),
):
    ids = []
    for f in files:
        fid = str(uuid.uuid4())
        dest = os.path.join(FILES_DIR, fid)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)

        audio = AudioSegment.from_file(dest)
        duration_sec = len(audio) / 1000.0
        meta = load_metadata()
        meta[fid] = {
            "filename": f.filename,
            "bytes": os.path.getsize(dest),
            "duration": duration_sec,
            "language": None,
        }
        save_metadata(meta)
        ids.append(fid)

    batch_id = str(uuid.uuid4())
    batch_path = os.path.join(BATCHES_DIR, batch_id + ".json")
    with open(batch_path, "w") as f:
        json.dump({"status": "running"}, f)

    threading.Thread(
        target=process_batch, args=(batch_id, ids, task, language), daemon=True
    ).start()

    return templates.TemplateResponse(
        "batch_status.html",
        {"request": request, "batch_id": batch_id, "data": {"status": "running"}},
    )

@app.get("/ui/batch_status", response_class=HTMLResponse)
async def ui_batch_status(request: Request, batch_id: str):
    path = os.path.join(BATCHES_DIR, batch_id + ".json")
    if not os.path.exists(path):
        return templates.TemplateResponse(
            "error.html", {"request": request, "message": "Batch not found"}
        )
    with open(path) as f:
        data = json.load(f)
    return templates.TemplateResponse(
        "batch_status.html", {"request": request, "batch_id": batch_id, "data": data}
    )

@app.get("/ui/batch_results", response_class=HTMLResponse)
async def ui_batch_results(request: Request, batch_id: str):
    path = os.path.join(BATCHES_DIR, batch_id + ".json")
    if not os.path.exists(path):
        return templates.TemplateResponse(
            "error.html", {"request": request, "message": "Batch not found"}
        )
    with open(path) as f:
        data = json.load(f)

    results = []
    for item in data.get("results", []):
        r = item["result"]
        results.append(
            {
                "file_id": item["file_id"],
                "result": {
                    "text": r.get("text", ""),
                    "segments": [
                        {"start": s["start"], "end": s["end"], "text": s["text"]}
                        for s in r.get("segments", [])
                    ],
                },
            }
        )

    return templates.TemplateResponse(
        "batch_results.html",
        {"request": request, "batch_id": batch_id, "results": results},
    )

@app.post("/ui/tts", response_class=HTMLResponse)
async def ui_tts(
    request: Request,
    text: str = Form(...),
    language: str = Form("en"),
    voice: str = Form("default"),
):
    out_path = f"/tmp/{uuid.uuid4()}.wav"
    speaker_ref = get_speaker_ref(voice)

    tts.tts_to_file(
        text=text,
        file_path=out_path,
        language=language,
        speaker_wav=speaker_ref if speaker_ref else None,
    )

    return templates.TemplateResponse(
        "tts.html", {"request": request, "path": out_path}
    )

@app.get("/download")
async def download(path: str):
    return FileResponse(path, media_type="audio/wav")

@app.get("/ui/files", response_class=HTMLResponse)
async def ui_files(request: Request):
    meta = load_metadata()
    files = []
    for fid, info in meta.items():
        files.append(
            {
                "id": fid,
                "filename": info.get("filename", ""),
                "bytes": info.get("bytes", 0),
                "duration": info.get("duration", 0.0),
                "language": info.get("language"),
            }
        )
    return templates.TemplateResponse(
        "files.html", {"request": request, "files": files}
    )

@app.get("/ui/waveform", response_class=HTMLResponse)
async def ui_waveform(request: Request, file_id: str):
    path = os.path.join(FILES_DIR, file_id)
    if not os.path.exists(path):
        return templates.TemplateResponse(
            "error.html", {"request": request, "message": "File not found"}
        )
    png_path = generate_waveform_png(path)
    png_name = os.path.basename(png_path)
    return templates.TemplateResponse(
        "waveform.html",
        {"request": request, "file_id": file_id, "png_name": png_name},
    )

@app.get("/ui/subtitles", response_class=HTMLResponse)
async def ui_subtitles(request: Request, batch_id: str, file_id: str):
    path = os.path.join(BATCHES_DIR, batch_id + ".json")
    if not os.path.exists(path):
        return templates.TemplateResponse(
            "error.html", {"request": request, "message": "Batch not found"}
        )
    with open(path) as f:
        data = json.load(f)

    segments = []
    for item in data.get("results", []):
        if item["file_id"] == file_id:
            r = item["result"]
            segments = [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in r.get("segments", [])
            ]
            break

    return templates.TemplateResponse(
        "subtitles.html", {"request": request, "segments": segments}
    )

@app.get("/ui/settings", response_class=HTMLResponse)
async def ui_settings(request: Request):
    cfg = load_config()
    return templates.TemplateResponse(
        "settings.html", {"request": request, "cfg": cfg}
    )

@app.get("/ui/performance", response_class=HTMLResponse)
async def ui_performance(request: Request):
    cfg = load_config()
    return templates.TemplateResponse(
        "performance.html", {"request": request, "cfg": cfg}
    )

@app.post("/ui/settings", response_class=HTMLResponse)
async def ui_settings_post(
    request: Request,
    whisper_model: str = Form(...),
    xtts_language: str = Form(...),
    xtts_voice: str = Form(...),
):   
    cfg = load_config()

    # Override with environment variables if present
    cfg["whisper_model"] = os.getenv("WHISPER_MODEL", cfg["whisper_model"])
    cfg["xtts_language"] = os.getenv("XTTS_LANGUAGE", cfg["xtts_language"])
    cfg["xtts_voice"] = os.getenv("XTTS_VOICE", cfg["xtts_voice"])       
    
    save_config(cfg)
    return templates.TemplateResponse(
        "settings.html", {"request": request, "cfg": cfg}
    )

@app.get("/ui/error", response_class=HTMLResponse)
async def ui_error(request: Request, message: str = "Error"):
    return templates.TemplateResponse(
        "error.html", {"request": request, "message": message}
    )

@app.get("/ui/monitor", response_class=HTMLResponse)
async def ui_monitor(request: Request):
    return templates.TemplateResponse("monitor.html", {"request": request})

@app.get("/ui/monitor_data")
async def ui_monitor_data():
    return {
        "single": list(PROGRESS_SINGLE.keys()),
        "batch": list(PROGRESS_BATCH.keys()),
    }

@app.get("/ui/logs", response_class=HTMLResponse)
async def ui_logs(request: Request):
    return templates.TemplateResponse("logs.html", {"request": request})

@app.get("/ui/dashboard", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})
    
logger.info(json.dumps({
    "event": "config",
    "FASTER_WHISPER_URL": FASTER_WHISPER_URL,
    "whisper_model": DEFAULT_CONFIG["whisper_model"],
    "xtts_language": DEFAULT_CONFIG["xtts_language"],
    "xtts_voice": DEFAULT_CONFIG["xtts_voice"],
}, ensure_ascii=False))
