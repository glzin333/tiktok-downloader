import os
import re
import json
import uuid
import time
import asyncio
import base64
import secrets
import subprocess
import shutil
import datetime
import httpx
from pathlib import Path
from typing import Optional
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Request
from fastapi.responses import FileResponse, JSONResponse
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

# Rate limit: máximo de requisições por janela de tempo
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))  # requests
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))       # segundos

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


# ── Rate limiter (in-memory por API key / IP) ─────────────────────────────────
_rate_buckets: dict[str, list[float]] = defaultdict(list)

def check_rate_limit(identifier: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    bucket = _rate_buckets[identifier]

    # Remove entradas fora da janela
    _rate_buckets[identifier] = [t for t in bucket if t > window_start]

    if len(_rate_buckets[identifier]) >= RATE_LIMIT_REQUESTS:
        retry_after = int(_rate_buckets[identifier][0] + RATE_LIMIT_WINDOW - now) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Limite de {RATE_LIMIT_REQUESTS} requisições por {RATE_LIMIT_WINDOW}s atingido. Tente novamente em {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    _rate_buckets[identifier].append(now)


# ── URL validation ────────────────────────────────────────────────────────────
TIKTOK_PATTERN = re.compile(
    r"https?://(www\.|vm\.|vt\.|m\.)?tiktok\.com/",
    re.IGNORECASE,
)

def validate_tiktok_url(url: str):
    if not TIKTOK_PATTERN.match(url):
        raise HTTPException(400, "URL inválida — apenas links do TikTok são aceitos.")


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
    request: Request,
    h: Optional[str] = Depends(_header_key),
    q: Optional[str] = Depends(_query_key),
):
    if not REQUIRE_API_KEY:
        identifier = request.client.host
        check_rate_limit(identifier)
        return "public"

    key = h or q
    if not key:
        raise HTTPException(401, "API key obrigatória — use o header X-API-Key")
    if key == ADMIN_KEY or key in load_keys():
        check_rate_limit(key)
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


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/info")
async def video_info(url: str = Query(...), _=Depends(get_api_key)):
    validate_tiktok_url(url)
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


class GeminiUploadRequest(BaseModel):
    url: str
    gemini_key: str


@app.post("/api/upload/gemini")
async def upload_to_gemini(req: GeminiUploadRequest, bg: BackgroundTasks, _=Depends(get_api_key)):
    """Baixa o vídeo, converte para H.264 e faz upload direto para o Gemini File API."""
    validate_tiktok_url(req.url)
    work = Path(TEMP_DIR) / str(uuid.uuid4())
    work.mkdir(parents=True)
    try:
        opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(work / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "concurrent_fragment_downloads": 4,
            "socket_timeout": 30,
            "retries": 3,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(req.url, download=True)

        files = list(work.glob("*"))
        if not files:
            raise HTTPException(500, "Download falhou")

        source = files[0]
        converted = work / "converted.mp4"

        subprocess.run([
            "ffmpeg", "-y", "-i", str(source),
            "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-vf", "scale=-2:720",
            "-acodec", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            str(converted)
        ], capture_output=True, timeout=120)

        if not converted.exists():
            raise HTTPException(500, "Conversão H.264 falhou")

        video_bytes = converted.read_bytes()
        upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={req.gemini_key}"

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                upload_url,
                content=video_bytes,
                headers={"Content-Type": "video/mp4"},
            )

        if resp.status_code != 200:
            raise HTTPException(500, f"Upload Gemini falhou: {resp.text[:300]}")

        file_data = resp.json().get("file", {})
        file_name = file_data.get("name")

        # Aguarda o arquivo ficar ACTIVE (polling)
        status_url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={req.gemini_key}"
        async with httpx.AsyncClient(timeout=60) as client:
            for _ in range(20):
                await asyncio.sleep(3)
                status_resp = await client.get(status_url)
                if status_resp.status_code == 200:
                    state = status_resp.json().get("state", "")
                    if state == "ACTIVE":
                        file_data = status_resp.json()
                        break
                    if state == "FAILED":
                        raise HTTPException(500, "Gemini falhou ao processar o arquivo")
            else:
                raise HTTPException(500, "Timeout aguardando Gemini processar o arquivo")

        bg.add_task(shutil.rmtree, work, True)
        return JSONResponse({
            "file_uri": file_data.get("uri"),
            "file_name": file_data.get("name"),
            "state": file_data.get("state"),
            "video_id": info.get("id"),
            "video_title": info.get("title"),
        })

    except HTTPException:
        shutil.rmtree(work, True)
        raise
    except Exception as e:
        shutil.rmtree(work, True)
        raise HTTPException(500, str(e))


@app.post("/api/download")
async def download_video(req: URLRequest, bg: BackgroundTasks, _=Depends(get_api_key)):
    validate_tiktok_url(req.url)
    work = Path(TEMP_DIR) / str(uuid.uuid4())
    work.mkdir(parents=True)
    try:
        opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(work / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "concurrent_fragment_downloads": 4,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(req.url, download=True)

        files = list(work.glob("*"))
        if not files:
            raise HTTPException(500, "Download não produziu arquivo de saída")

        video_file = files[0]
        filename = f"tiktok_{info.get('id', 'video')}{video_file.suffix}"

        bg.add_task(shutil.rmtree, work, True)
        return FileResponse(str(video_file), media_type="video/mp4", filename=filename)

    except HTTPException:
        shutil.rmtree(work, True)
        raise
    except Exception as e:
        shutil.rmtree(work, True)
        raise HTTPException(500, str(e))


@app.post("/api/download/gemini")
async def download_for_gemini(req: URLRequest, bg: BackgroundTasks, _=Depends(get_api_key)):
    """Baixa o vídeo e converte para H.264 — compatível com Gemini File API."""
    validate_tiktok_url(req.url)
    work = Path(TEMP_DIR) / str(uuid.uuid4())
    work.mkdir(parents=True)
    try:
        opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(work / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "concurrent_fragment_downloads": 4,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(req.url, download=True)

        files = list(work.glob("*"))
        if not files:
            raise HTTPException(500, "Download falhou")

        source = files[0]
        converted = work / "converted.mp4"

        subprocess.run([
            "ffmpeg", "-y", "-i", str(source),
            "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-vf", "scale=-2:720",
            "-acodec", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            str(converted)
        ], capture_output=True, timeout=120)

        if not converted.exists():
            raise HTTPException(500, "Conversão para H.264 falhou")

        filename = f"tiktok_{info.get('id', 'video')}_h264.mp4"
        bg.add_task(shutil.rmtree, work, True)
        return FileResponse(str(converted), media_type="video/mp4", filename=filename)

    except HTTPException:
        shutil.rmtree(work, True)
        raise
    except Exception as e:
        shutil.rmtree(work, True)
        raise HTTPException(500, str(e))


@app.post("/api/thumbnail")
async def get_thumbnail(req: URLRequest, bg: BackgroundTasks, _=Depends(get_api_key)):
    validate_tiktok_url(req.url)
    work = Path(TEMP_DIR) / str(uuid.uuid4())
    work.mkdir(parents=True)
    try:
        opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(work / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "concurrent_fragment_downloads": 4,
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

        filename = f"tiktok_{info.get('id', 'frame')}_frame.jpg"

        bg.add_task(shutil.rmtree, work, True)
        return FileResponse(str(thumb), media_type="image/jpeg", filename=filename)

    except HTTPException:
        shutil.rmtree(work, True)
        raise
    except Exception as e:
        shutil.rmtree(work, True)
        raise HTTPException(500, str(e))


@app.post("/api/thumbnail/base64")
async def get_thumbnail_base64(req: URLRequest, bg: BackgroundTasks, _=Depends(get_api_key)):
    """Extrai o 1º frame e retorna como JSON com image_base64 — ideal para n8n + Claude."""
    validate_tiktok_url(req.url)
    work = Path(TEMP_DIR) / str(uuid.uuid4())
    work.mkdir(parents=True)
    try:
        opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(work / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "concurrent_fragment_downloads": 4,
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

        image_base64 = base64.b64encode(thumb.read_bytes()).decode("utf-8")

        bg.add_task(shutil.rmtree, work, True)
        return JSONResponse({
            "id": info.get("id"),
            "title": info.get("title"),
            "uploader": info.get("uploader"),
            "duration": info.get("duration"),
            "image_base64": image_base64,
            "image_media_type": "image/jpeg",
        })

    except HTTPException:
        shutil.rmtree(work, True)
        raise
    except Exception as e:
        shutil.rmtree(work, True)
        raise HTTPException(500, str(e))


# ── RunningHub ───────────────────────────────────────────────────────────────
RUNNINGHUB_BASE = "https://www.runninghub.ai"
RUNNINGHUB_WORKFLOW_ID = os.getenv("RUNNINGHUB_WORKFLOW_ID", "2049480632044097538")
RUNNINGHUB_NODE_ID = os.getenv("RUNNINGHUB_NODE_ID", "45")
RUNNINGHUB_VIDEO_WORKFLOW_ID = os.getenv("RUNNINGHUB_VIDEO_WORKFLOW_ID", "2049538202083528705")
RUNNINGHUB_VIDEO_IMAGE_NODE = os.getenv("RUNNINGHUB_VIDEO_IMAGE_NODE", "52")
RUNNINGHUB_VIDEO_PROMPT_NODE = os.getenv("RUNNINGHUB_VIDEO_PROMPT_NODE", "6")


class RunImageRequest(BaseModel):
    prompt: str
    runninghub_key: str



class RunVideoRequest(BaseModel):
    image_url: str
    motion_prompt: str
    runninghub_key: str


@app.post("/api/run/video")
async def run_video(req: RunVideoRequest, _=Depends(get_api_key)):
    """Operação DRIFT — anima imagem do FORGE via RunningHub img2video."""
    try:
        auth_headers = {
            "Authorization": f"Bearer {req.runninghub_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            img_resp = await client.get(req.image_url)
        if img_resp.status_code != 200:
            raise HTTPException(500, f"Falha ao baixar imagem: {img_resp.status_code}")

        async with httpx.AsyncClient(timeout=60) as client:
            upload_resp = await client.post(
                f"{RUNNINGHUB_BASE}/task/openapi/upload",
                headers={"Authorization": f"Bearer {req.runninghub_key}"},
                files={"file": ("image.png", img_resp.content, "image/png")},
                data={"apiKey": req.runninghub_key},
            )
        if upload_resp.status_code != 200:
            raise HTTPException(500, f"Upload RunningHub falhou: {upload_resp.text[:300]}")

        upload_data = upload_resp.json()
        raw = upload_data.get("data")
        if isinstance(raw, dict):
            file_name = raw.get("fileName") or raw.get("filename") or raw.get("name")
        elif isinstance(raw, str):
            file_name = raw
        else:
            file_name = upload_data.get("fileName") or upload_data.get("filename")
        if not file_name:
            raise HTTPException(500, f"fileName não retornado. Resposta: {upload_resp.text[:300]}")

        async with httpx.AsyncClient(timeout=30) as client:
            create_resp = await client.post(
                f"{RUNNINGHUB_BASE}/task/openapi/create",
                headers=auth_headers,
                json={
                    "workflowId": RUNNINGHUB_VIDEO_WORKFLOW_ID,
                    "apiKey": req.runninghub_key,
                    "nodeInfoList": [
                        {"nodeId": RUNNINGHUB_VIDEO_IMAGE_NODE, "fieldName": "image", "fieldValue": file_name},
                        {"nodeId": RUNNINGHUB_VIDEO_PROMPT_NODE, "fieldName": "text", "fieldValue": req.motion_prompt},
                    ],
                },
            )
        if create_resp.status_code != 200:
            raise HTTPException(500, f"RunningHub create falhou: {create_resp.text[:500]}")

        create_data = create_resp.json()
        raw = create_data.get("data")
        if isinstance(raw, str):
            task_id = raw
        elif isinstance(raw, dict):
            task_id = raw.get("taskId")
        else:
            task_id = create_data.get("taskId")
        if not task_id:
            raise HTTPException(500, f"taskId não retornado. Resposta: {create_resp.text[:500]}")

        async with httpx.AsyncClient(timeout=10) as client:
            for i in range(120):
                await asyncio.sleep(5)
                try:
                    status_resp = await client.post(
                        f"{RUNNINGHUB_BASE}/task/openapi/status",
                        headers=auth_headers,
                        json={"taskId": task_id, "apiKey": req.runninghub_key},
                    )
                except Exception:
                    continue
                if status_resp.status_code != 200:
                    continue
                status_data = status_resp.json().get("data", "")
                status = status_data if isinstance(status_data, str) else status_data.get("taskStatus", "")
                if status == "SUCCESS":
                    break
                if status == "FAILED":
                    raise HTTPException(500, "RunningHub DRIFT: geração de vídeo falhou")
            else:
                raise HTTPException(500, "Timeout aguardando RunningHub DRIFT")

        async with httpx.AsyncClient(timeout=30) as client:
            out_resp = await client.post(
                f"{RUNNINGHUB_BASE}/task/openapi/outputs",
                headers=auth_headers,
                json={"taskId": task_id, "apiKey": req.runninghub_key},
            )
        if out_resp.status_code != 200:
            raise HTTPException(500, f"RunningHub outputs falhou: {out_resp.text[:500]}")

        out_json = out_resp.json()
        out_data = out_json.get("results") or out_json.get("data") or []
        if not out_data:
            raise HTTPException(500, f"Nenhum output retornado. Resposta: {out_resp.text[:500]}")

        if isinstance(out_data, list):
            first = out_data[0]
            video_url = first if isinstance(first, str) else (
                first.get("fileUrl") or first.get("url") or first.get("videoUrl")
            )
        elif isinstance(out_data, dict):
            video_url = out_data.get("fileUrl") or out_data.get("url") or out_data.get("videoUrl")
        else:
            video_url = str(out_data)

        return JSONResponse({
            "task_id": task_id,
            "video_url": video_url,
            "outputs": out_data if isinstance(out_data, list) else [out_data],
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erro interno DRIFT: {str(e)}")


@app.post("/api/run/image")
async def run_image(req: RunImageRequest, _=Depends(get_api_key)):
    """Dispara workflow de imagem no RunningHub e aguarda o resultado."""
    try:
        auth_headers = {
            "Authorization": f"Bearer {req.runninghub_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            create_resp = await client.post(
                f"{RUNNINGHUB_BASE}/task/openapi/create",
                headers=auth_headers,
                json={
                    "workflowId": RUNNINGHUB_WORKFLOW_ID,
                    "apiKey": req.runninghub_key,
                    "nodeInfoList": [
                        {
                            "nodeId": RUNNINGHUB_NODE_ID,
                            "fieldName": "text",
                            "fieldValue": req.prompt,
                        }
                    ],
                },
            )

        if create_resp.status_code != 200:
            raise HTTPException(500, f"RunningHub create falhou ({create_resp.status_code}): {create_resp.text[:500]}")

        create_data = create_resp.json()
        raw = create_data.get("data")
        if isinstance(raw, str):
            task_id = raw
        elif isinstance(raw, dict):
            task_id = raw.get("taskId")
        else:
            task_id = create_data.get("taskId")
        if not task_id:
            raise HTTPException(500, f"taskId não retornado. Resposta: {create_resp.text[:500]}")

        async with httpx.AsyncClient(timeout=10) as client:
            for i in range(60):
                await asyncio.sleep(5)
                try:
                    status_resp = await client.post(
                        f"{RUNNINGHUB_BASE}/task/openapi/status",
                        headers=auth_headers,
                        json={"taskId": task_id, "apiKey": req.runninghub_key},
                    )
                except Exception:
                    continue
                if status_resp.status_code != 200:
                    continue
                status_data = status_resp.json().get("data", "")
                status = status_data if isinstance(status_data, str) else status_data.get("taskStatus", "")
                if status == "SUCCESS":
                    break
                if status == "FAILED":
                    raise HTTPException(500, "RunningHub: geração falhou")
            else:
                raise HTTPException(500, "Timeout aguardando RunningHub")

        # Busca outputs
        async with httpx.AsyncClient(timeout=30) as client:
            out_resp = await client.post(
                f"{RUNNINGHUB_BASE}/task/openapi/outputs",
                headers=auth_headers,
                json={"taskId": task_id, "apiKey": req.runninghub_key},
            )

        if out_resp.status_code != 200:
            raise HTTPException(500, f"RunningHub outputs falhou: {out_resp.text[:500]}")

        out_json = out_resp.json()
        # RunningHub retorna results no topo ({"results": [...]}) ou dentro de data
        out_data = out_json.get("results") or out_json.get("data") or []
        if not out_data:
            raise HTTPException(500, f"Nenhum output retornado. Resposta: {out_resp.text[:500]}")

        if isinstance(out_data, str):
            image_url = out_data
            outputs = [out_data]
        elif isinstance(out_data, list):
            outputs = out_data
            first = out_data[0]
            image_url = first if isinstance(first, str) else (
                first.get("url") or first.get("fileUrl") or first.get("imageUrl")
            )
        elif isinstance(out_data, dict):
            outputs = [out_data]
            image_url = out_data.get("url") or out_data.get("fileUrl") or out_data.get("imageUrl")
        else:
            raise HTTPException(500, f"Formato de output inesperado: {out_resp.text[:500]}")

        return JSONResponse({
            "task_id": task_id,
            "image_url": image_url,
            "outputs": outputs,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erro interno: {str(e)}")


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
