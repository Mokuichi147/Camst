from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from camst.camera import create_camera
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        camera.start()
        try:
            yield
        finally:
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
            if pc.connectionState in {"failed", "closed", "disconnected"}:
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

    return app
