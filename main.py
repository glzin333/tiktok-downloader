import os
import json
import uuid
import secrets
import subprocess
import shutil
import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel
import yt_dlp

# ── Config ────────────────────────────────────────────────────────────────────
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin-change-me")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
KEYS_FILE = os.getenv("KEYS_FILE", "/app/data/api_keys.json")
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/tiktok_dl")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(KEYS_FILE), exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="TikTok Downloader API", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_header_key = APIKeyHeader(name="X-API-Key", auto_error=False)
_query_key = APIKeyQuery(name="api_key", auto_error=False)


# ── Key helpers ───────────────────────────────────────────────────────────────
def load_keys() -> dict:
    if not Path(KEYS_FILE).exists():
        return {}
    with open(KEYS_FILE) as f:
        return json.load(f)


def save_keys(keys: dict):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


async def get_api_key(
    h: Optional[str] = Depends(_header_key),
    q: Optional[str] = Depends(_query_key),
):
    if not REQUIRE_API_KEY:
        return "public"
    key = h or q
    if not key:
        raise HTTPException(401, "API key obrigatória — use o header X-API-Key ou ?api_key=")
    if key == ADMIN_KEY or key in load_keys():
        return key
    raise HTTPException(401, "API key inválida")


async def require_admin(
    h: Optional[str] = Depends(_header_key),
    q: Optional[str] = Depends(_query_key),
):
    if (h or q) != ADMIN_KEY:
        raise HTTPException(403, "Admin key obrigatória")
    return ADMIN_KEY


# ── Models ────────────────────────────────────────────────────────────────────
class URLRequest(BaseModel):
    url: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_filename(title: str, max_len: int = 60) -> str:
    return "".join(c for c in title if c.isalnum() or c in " -_").strip()[:max_len] or "video"


def _best_video_format(info: dict) -> tuple[str, dict]:
    """Return (url, http_headers) for best video format."""
    formats = info.get("formats", [info])
    for fmt in reversed(formats):
        if fmt.get("vcodec") not in (None, "none") and fmt.get("url"):
            return fmt["url"], fmt.get("http_headers", {})
    return info.get("url", ""), info.get("http_headers", {})


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/info")
async def video_info(url: str = Query(...), _=Depends(get_api_key)):
    """Retorna metadados do vídeo sem fazer download."""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "duration": info.get("duration"),
            "uploader": info.get("uploader"),
            "uploader_id": info.get("uploader_id"),
            "thumbnail": info.get("thumbnail"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "comment_count": info.get("comment_count"),
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/download")
async def download_video(req: URLRequest, bg: BackgroundTasks, _=Depends(get_api_key)):
    """Baixa o vídeo sem marca d'água e retorna o arquivo MP4."""
    work = Path(TEMP_DIR) / str(uuid.uuid4())
    work.mkdir(parents=True)
    try:
        opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(work / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(req.url, download=True)

        files = list(work.glob("*"))
        if not files:
            raise HTTPException(500, "Download não produziu arquivo de saída")

        video_file = files[0]
        filename = f"{_safe_filename(info.get('title', 'video'))}{video_file.suffix}"

        bg.add_task(shutil.rmtree, work, True)
        return FileResponse(str(video_file), media_type="video/mp4", filename=filename)

    except HTTPException:
        shutil.rmtree(work, True)
        raise
    except Exception as e:
        shutil.rmtree(work, True)
        raise HTTPException(500, str(e))


@app.post("/api/thumbnail")
async def get_thumbnail(req: URLRequest, bg: BackgroundTasks, _=Depends(get_api_key)):
    """Baixa o vídeo, extrai o primeiro frame e retorna como JPEG."""
    work = Path(TEMP_DIR) / str(uuid.uuid4())
    work.mkdir(parents=True)
    try:
        opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(work / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(req.url, download=True)

        files = list(work.glob("*"))
        if not files:
            raise HTTPException(500, "Download falhou")

        video_file = files[0]
        thumb = work / "frame.jpg"

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_file), "-vframes", "1", "-q:v", "2", str(thumb)],
            capture_output=True, timeout=60,
        )

        if not thumb.exists():
            raise HTTPException(500, f"Extração de frame falhou: {result.stderr.decode()[-400:]}")

        filename = f"{_safe_filename(info.get('title', 'frame'), 40)}_frame.jpg"

        bg.add_task(shutil.rmtree, work, True)
        return FileResponse(str(thumb), media_type="image/jpeg", filename=filename)

    except HTTPException:
        shutil.rmtree(work, True)
        raise
    except Exception as e:
        shutil.rmtree(work, True)
        raise HTTPException(500, str(e))


# ── Admin ─────────────────────────────────────────────────────────────────────
@app.get("/admin/keys")
async def list_keys(_=Depends(require_admin)):
    keys = load_keys()
    return {"keys": [{"key": k, **v} for k, v in keys.items()], "count": len(keys)}


@app.post("/admin/keys")
async def create_key(name: str = Query(""), _=Depends(require_admin)):
    keys = load_keys()
    new_key = secrets.token_urlsafe(32)
    keys[new_key] = {
        "name": name,
        "id": str(uuid.uuid4()),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    save_keys(keys)
    return {"key": new_key, "name": name}


@app.delete("/admin/keys/{key_id}")
async def revoke_key(key_id: str, _=Depends(require_admin)):
    keys = load_keys()
    if key_id not in keys:
        raise HTTPException(404, "Key não encontrada")
    del keys[key_id]
    save_keys(keys)
    return {"revoked": key_id}


# ── Static (deve ser o último mount) ─────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
