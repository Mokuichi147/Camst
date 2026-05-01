from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from camst.camera import HLSStream

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    camera = HLSStream()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        camera.start()
        try:
            yield
        finally:
            camera.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "index.html", {"playlist": f"/hls/{camera.playlist_name}"}
        )

    @app.get("/hls/{filename}")
    async def hls_file(filename: str) -> FileResponse:
        if "/" in filename or ".." in filename:
            raise HTTPException(status_code=400)
        path = camera.directory / filename
        if not path.is_file():
            raise HTTPException(status_code=404)
        if filename.endswith(".m3u8"):
            media_type = "application/vnd.apple.mpegurl"
        elif filename.endswith(".ts"):
            media_type = "video/mp2t"
        else:
            media_type = "application/octet-stream"
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/status", response_class=HTMLResponse)
    async def status() -> HTMLResponse:
        fps = camera.fps
        return HTMLResponse(f"<span class='font-mono'>{fps:5.1f} fps</span>")

    return app
