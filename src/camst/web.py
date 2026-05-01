from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from camst.camera import CameraStream
from camst.webrtc import CameraVideoTrack

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    camera = CameraStream()
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
        return templates.TemplateResponse(request, "index.html")

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
