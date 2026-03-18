import os
import uuid
import json
import shutil
import threading
import tempfile

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
FASTER_WHISPER_URL = "http://192.168.2.132:10300/inference"

os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(BATCHES_DIR, exist_ok=True)
os.makedirs(WAVEFORMS_DIR, exist_ok=True)
os.makedirs(SPEAKERS_DIR, exist_ok=True)

DEFAULT_CONFIG = {
    "whisper_model": "large-v3",
    "xtts_language": "en",
    "xtts_voice": "default",
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
# XTTS v2
# -----------------------------
print("Loading XTTS v2 model…")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
print("XTTS v2 loaded.")

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

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            FASTER_WHISPER_URL,
            files={"audio_file": open(path, "rb")},
            data={"task": "transcribe", "language": language},
        )
    data = resp.json()

    if response_format == "srt":
        return segments_to_srt(data["segments"])

    return {"text": data.get("text", "")}

@app.post("/v1/audio/translations")
async def translate_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    response_format: str = Form("json"),
    target_language: str = Form("en"),
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            FASTER_WHISPER_URL,
            files={"audio_file": open(path, "rb")},
            data={"task": "translate", "language": target_language},
        )
    data = resp.json()

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

    for fid in file_ids:
        fpath = os.path.join(FILES_DIR, fid)
        if not os.path.exists(fpath):
            continue

        with open(fpath, "rb") as audio:
            with httpx.Client(timeout=600) as client:
                resp = client.post(
                    FASTER_WHISPER_URL,
                    files={"audio_file": audio},
                    data={"task": task, "language": language},
                )

        results.append({"file_id": fid, "result": resp.json()})

    with open(batch_path, "w") as f:
        json.dump({"status": "completed", "results": results}, f)

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
    target_language: str = Form("en"),
    response_format: str = Form("json"),
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name

    async with httpx.AsyncClient(timeout=600) as client:
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

@app.post("/ui/settings", response_class=HTMLResponse)
async def ui_settings_post(
    request: Request,
    whisper_model: str = Form(...),
    xtts_language: str = Form(...),
    xtts_voice: str = Form(...),
):
    cfg = {
        "whisper_model": whisper_model,
        "xtts_language": xtts_language,
        "xtts_voice": xtts_voice,
    }
    save_config(cfg)
    return templates.TemplateResponse(
        "settings.html", {"request": request, "cfg": cfg}
    )

@app.get("/ui/error", response_class=HTMLResponse)
async def ui_error(request: Request, message: str = "Error"):
    return templates.TemplateResponse(
        "error.html", {"request": request, "message": message}
    )
