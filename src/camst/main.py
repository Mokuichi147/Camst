from __future__ import annotations

import cv2
import typer
import uvicorn

from camst.camera import create_camera
from camst.web import create_app

app = typer.Typer(add_completion=False, help="カメラ映像ビューアー (OAK-D LITE / UVC)")


def _run_local(
    source: str, device: int | str, rotate: int, eye: str,
    correct: bool, clahe_clip: float,
) -> None:
    camera = create_camera(
        source=source, device=device, rotate=rotate, eye=eye,
        correct=correct, clahe_clip=clahe_clip,
    )
    camera.start()
    try:
        while True:
            frame = camera.latest()
            if frame is not None:
                cv2.imshow("camst", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def _run_webui(
    host: str, port: int, source: str, device: int | str, rotate: int, eye: str,
    correct: bool, clahe_clip: float,
) -> None:
    uvicorn.run(
        create_app(
            source=source, device=device, rotate=rotate, eye=eye,
            correct=correct, clahe_clip=clahe_clip,
        ),
        host=host,
        port=port,
    )


@app.command()
def main(
    source: str = typer.Option(
        "oak",
        "--source",
        help="カメラの種類 (oak | leap | uvc)",
        case_sensitive=False,
    ),
    device: str = typer.Option(
        "Leap Motion",
        "--device",
        help="leap/uvc用のデバイス指定。番号 (例: 1) かデバイス名の一部 (例: Leap)",
    ),
    eye: str = typer.Option(
        "left", "--eye", help="leap用: 使う眼 (left | right | both)"
    ),
    correct: bool = typer.Option(
        False, "--correct", help="leap用: 明るさ補正(CLAHE)を有効化する"
    ),
    clahe_clip: float = typer.Option(
        2.0, "--clahe-clip", help="leap用: CLAHEのクリップ上限(大きいほど強い補正)"
    ),
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
    """カメラ映像をリアルタイム表示する。"""
    if source not in ("oak", "leap", "uvc"):
        raise typer.BadParameter("--source は oak / leap / uvc のいずれかです")
    if rotate not in (0, 90, 180, 270):
        raise typer.BadParameter("--rotate は 0/90/180/270 のいずれかです")
    if webui:
        typer.echo(f"WebUI を起動: http://{host}:{port}")
        _run_webui(host, port, source, device, rotate, eye, correct, clahe_clip)
    else:
        _run_local(source, device, rotate, eye, correct, clahe_clip)


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
