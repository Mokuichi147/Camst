from __future__ import annotations

import cv2
import typer
import uvicorn

from camst.camera import CameraStream
from camst.web import create_app

app = typer.Typer(add_completion=False, help="OAK-D LITE RGB ビューアー")


def _run_local(rotate: int) -> None:
    camera = CameraStream(rotate=rotate)
    camera.start()
    try:
        while True:
            frame = camera.latest()
            if frame is not None:
                cv2.imshow("OAK-D LITE RGB", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def _run_webui(host: str, port: int, rotate: int) -> None:
    uvicorn.run(create_app(rotate=rotate), host=host, port=port)


@app.command()
def main(
    webui: bool = typer.Option(False, "--webui", help="ブラウザでストリームを表示"),
    host: str = typer.Option("127.0.0.1", "--host", help="WebUIのバインドホスト"),
    port: int = typer.Option(8000, "--port", help="WebUIのポート"),
    rotate: int = typer.Option(
        0,
        "--rotate",
        help="映像の回転角度 (0, 90, 180, 270)",
        case_sensitive=False,
    ),
) -> None:
    """OAK-D LITE のRGB映像をリアルタイム表示する。"""
    if rotate not in (0, 90, 180, 270):
        raise typer.BadParameter("--rotate は 0/90/180/270 のいずれかです")
    if webui:
        typer.echo(f"WebUI を起動: http://{host}:{port}")
        _run_webui(host, port, rotate)
    else:
        _run_local(rotate)


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
