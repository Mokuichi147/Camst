from __future__ import annotations

import cv2
import depthai as dai
import typer
import uvicorn

app = typer.Typer(add_completion=False, help="OAK-D LITE RGB ビューアー")


def _run_local() -> None:
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        output = cam.requestOutput((1280, 720), dai.ImgFrame.Type.NV12)
        queue = output.createOutputQueue()
        pipeline.start()
        while pipeline.isRunning():
            frame = queue.get()
            cv2.imshow("OAK-D LITE RGB", frame.getCvFrame())
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()


def _run_webui(host: str, port: int) -> None:
    uvicorn.run("camst.web:create_app", host=host, port=port, factory=True)


@app.command()
def main(
    webui: bool = typer.Option(False, "--webui", help="ブラウザでストリームを表示"),
    host: str = typer.Option("127.0.0.1", "--host", help="WebUIのバインドホスト"),
    port: int = typer.Option(8000, "--port", help="WebUIのポート"),
) -> None:
    """OAK-D LITE のRGB映像をリアルタイム表示する。"""
    if webui:
        typer.echo(f"WebUI を起動: http://{host}:{port}")
        _run_webui(host, port)
    else:
        _run_local()


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
