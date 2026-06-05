from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from camst.camera import create_camera
from camst.recorder import (
    MotionRecorder,
    list_clips,
    list_favorites,
    set_favorite,
    thumbnail_for_clip,
)
from camst.webrtc import CameraVideoTrack

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(
    source: str = "oak",
    device: int | str = "Leap Motion",
    rotate: int = 0,
    eye: str = "left",
    correct: bool = False,
    clahe_clip: float = 2.0,
    denoise: int = 1,
    nlm: bool = False,
    nlm_h: float = 1.0,
    nlm_scale: float = 2.0,
    nlm_template: int = 7,
    nlm_search: int = 21,
    record: bool = False,
    recordings_dir: str = "recordings",
    motion_area: float = 0.0008,
    motion_threshold: int = 22,
) -> FastAPI:
    camera = create_camera(
        source=source, device=device, rotate=rotate, eye=eye,
        correct=correct, clahe_clip=clahe_clip, denoise=denoise,
        nlm=nlm, nlm_h=nlm_h, nlm_scale=nlm_scale,
        nlm_template=nlm_template, nlm_search=nlm_search,
    )
    aspect_w, aspect_h = (9, 16) if rotate in (90, 270) else (16, 9)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    pcs: set[RTCPeerConnection] = set()
    rec_dir = Path(recordings_dir)
    recorder = (
        MotionRecorder(
            camera,
            directory=rec_dir,
            min_area_ratio=motion_area,
            diff_threshold=motion_threshold,
        )
        if record
        else None
    )

    def recording_path(name: str) -> Path | None:
        # ディレクトリトラバーサル防止: 想定する命名のファイルだけを許可する。
        path = (rec_dir / name).resolve()
        if (
            not name.startswith("motion_")
            or path.parent != rec_dir.resolve()
            or not path.is_file()
        ):
            return None
        return path

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        camera.start()
        if recorder is not None:
            recorder.start()
        try:
            yield
        finally:
            if recorder is not None:
                recorder.stop()
            for pc in list(pcs):
                await pc.close()
            pcs.clear()
            camera.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"aspect_w": aspect_w, "aspect_h": aspect_h},
        )

    @app.post("/offer")
    async def offer(request: Request) -> JSONResponse:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            # disconnected は一時的な切断で自然回復しうるため閉じない。
            # ここで閉じると低速回線での一時的なパケットロスでも接続が永久に切れ、
            # リロードするまで復帰しなくなる。回復不能な failed と、
            # 明示終了の closed のときだけ破棄する。
            if pc.connectionState in {"failed", "closed"}:
                await pc.close()
                pcs.discard(pc)

        pc.addTrack(CameraVideoTrack(camera))
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return JSONResponse(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        )

    @app.get("/status", response_class=HTMLResponse)
    async def status() -> HTMLResponse:
        return HTMLResponse(f"<span class='font-mono'>{camera.fps:5.1f} fps</span>")

    @app.get("/recordings", response_class=HTMLResponse)
    async def recordings(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "recordings.html",
            {"clips": list_clips(rec_dir), "record": record},
        )

    @app.get("/recordings/list", response_class=HTMLResponse)
    async def recordings_list(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "_recordings_list.html",
            {"clips": list_clips(rec_dir)},
        )

    @app.post("/recordings/favorite/{name}", response_class=HTMLResponse)
    async def recording_favorite(request: Request, name: str) -> HTMLResponse:
        path = recording_path(name)
        if path is None:
            return HTMLResponse("not found", status_code=404)
        # 現在の状態を反転して保存し、更新後のカードだけを返す(htmx で差し替え)。
        favorite = name not in list_favorites(rec_dir)
        set_favorite(rec_dir, name, favorite)
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            size = path.stat().st_size
        except OSError:
            return HTMLResponse("not found", status_code=404)
        clip = {
            "name": name,
            "size": size,
            "mtime": mtime,
            "favorite": favorite,
        }
        return templates.TemplateResponse(
            request, "_recording_card.html", {"clip": clip}
        )

    @app.get("/recordings/media/{name}", response_model=None)
    async def recording_media(name: str) -> FileResponse | JSONResponse:
        path = recording_path(name)
        if path is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path)

    @app.get("/recordings/thumbnail/{name}", response_model=None)
    async def recording_thumbnail(name: str) -> FileResponse | JSONResponse:
        path = recording_path(name)
        if path is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        thumb = thumbnail_for_clip(path)
        if thumb is None:
            return JSONResponse({"error": "thumbnail not available"}, status_code=404)
        # サムネイルは確定済みの録画から生成され内容が変わらないため、
        # ブラウザに長期キャッシュさせて再読み込み時の転送を減らす。
        return FileResponse(
            thumb,
            media_type="image/webp",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    return app
