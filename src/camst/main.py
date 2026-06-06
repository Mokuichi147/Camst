from __future__ import annotations

import cv2
import typer
import uvicorn

from camst.camera import create_camera
from camst.web import create_app

app = typer.Typer(add_completion=False, help="カメラ映像ビューアー (OAK-D LITE / UVC)")


def _run_local(
    source: str, device: int | str, rotate: int, eye: str,
    correct: bool, clahe_clip: float, denoise: int,
    nlm: bool, nlm_h: float, nlm_scale: float, nlm_template: int, nlm_search: int,
) -> None:
    camera = create_camera(
        source=source, device=device, rotate=rotate, eye=eye,
        correct=correct, clahe_clip=clahe_clip, denoise=denoise,
        nlm=nlm, nlm_h=nlm_h, nlm_scale=nlm_scale,
        nlm_template=nlm_template, nlm_search=nlm_search,
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
    correct: bool, clahe_clip: float, denoise: int,
    nlm: bool, nlm_h: float, nlm_scale: float, nlm_template: int, nlm_search: int,
    record: bool, motion_area: float, motion_threshold: int,
    encode_crf: int, encode_width: int | None,
) -> None:
    uvicorn.run(
        create_app(
            source=source, device=device, rotate=rotate, eye=eye,
            correct=correct, clahe_clip=clahe_clip, denoise=denoise,
            nlm=nlm, nlm_h=nlm_h, nlm_scale=nlm_scale,
            nlm_template=nlm_template, nlm_search=nlm_search,
            record=record, motion_area=motion_area,
            motion_threshold=motion_threshold,
            encode_crf=encode_crf, encode_width=encode_width,
        ),
        host=host,
        port=port,
    )


@app.command()
def main(
    source: str = typer.Option(
        "uvc",
        "--source",
        help="カメラの種類 (oak | leap | uvc)",
        case_sensitive=False,
    ),
    device: str = typer.Option(
        "0",
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
    denoise: int = typer.Option(
        1, "--denoise", help="leap用: ノイズ低減のため直近Nフレームを移動平均(1=無効)"
    ),
    nlm: bool = typer.Option(
        False, "--nlm", help="leap用: 空間NLMeansによる強力なノイズ除去を有効化"
    ),
    nlm_h: float = typer.Option(
        1.0, "--nlm-h", help="leap用: NLMeansの強度(大きいほど強く除去。強すぎると潰れる)"
    ),
    nlm_scale: float = typer.Option(
        2.0, "--nlm-scale", help="leap用: NLMeansを縮小して高速化する倍率(大きいほど速いが粗い)"
    ),
    nlm_template: int = typer.Option(
        7, "--nlm-template", help="leap用: NLMeansのテンプレート窓サイズ(奇数)"
    ),
    nlm_search: int = typer.Option(
        21, "--nlm-search", help="leap用: NLMeansの探索窓サイズ(奇数。小さいほど速い)"
    ),
    record: bool = typer.Option(
        False,
        "--record",
        help="動体検知でクリップを自動録画する(WebUI用。直近100件・1本最大1分)",
    ),
    motion_area: float = typer.Option(
        0.0008,
        "--motion-area",
        help="録画用: 動きとみなす最小面積(画面に占める割合)。小さいほど"
        "小動物など小さな動きを拾う",
    ),
    motion_threshold: int = typer.Option(
        22,
        "--motion-threshold",
        help="録画用: 動きとみなすフレーム差分の閾値(0-255)。小さいほど"
        "弱いコントラストの動きを拾う",
    ),
    encode_crf: int = typer.Option(
        26,
        "--encode-crf",
        help="録画の再エンコード品質(H.264 CRF)。大きいほど小さく低画質(目安18〜30)",
    ),
    encode_width: int = typer.Option(
        0,
        "--encode-width",
        help="録画をこの横幅まで縮小して保存しさらに小さくする(0=縮小しない)",
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
    if denoise < 1:
        raise typer.BadParameter("--denoise は1以上です")
    if not 0.0 < motion_area <= 1.0:
        raise typer.BadParameter("--motion-area は0より大きく1以下です")
    if not 0 <= motion_threshold <= 255:
        raise typer.BadParameter("--motion-threshold は0以上255以下です")
    if not 0 <= encode_crf <= 51:
        raise typer.BadParameter("--encode-crf は0〜51です(目安18〜30)")
    if encode_width < 0:
        raise typer.BadParameter("--encode-width は0以上です(0=縮小しない)")
    if webui:
        typer.echo(f"WebUI を起動: http://{host}:{port}")
        _run_webui(
            host, port, source, device, rotate, eye,
            correct, clahe_clip, denoise,
            nlm, nlm_h, nlm_scale, nlm_template, nlm_search,
            record, motion_area, motion_threshold,
            encode_crf, encode_width or None,
        )
    else:
        _run_local(
            source, device, rotate, eye, correct, clahe_clip, denoise,
            nlm, nlm_h, nlm_scale, nlm_template, nlm_search,
        )


def cli() -> None:
    app()


reencode_app = typer.Typer(
    add_completion=False,
    help="既存の録画を H.264 へ再エンコードして縮小する(1回限りの一括変換)",
)


@reencode_app.command()
def reencode(
    recordings_dir: str = typer.Option(
        "recordings", "--recordings-dir", help="録画の保存ディレクトリ"
    ),
    encode_crf: int = typer.Option(
        26, "--encode-crf", help="再エンコード品質(H.264 CRF)。大きいほど小さく低画質"
    ),
    encode_width: int = typer.Option(
        0, "--encode-width", help="この横幅まで縮小する(0=縮小しない)"
    ),
    force: bool = typer.Option(
        False, "--force", help="すでに H.264 のクリップも再変換する"
    ),
) -> None:
    """recordings ディレクトリ内の録画をまとめて H.264 へ変換し小さくする。"""
    from pathlib import Path

    from camst.recorder import reencode_directory

    if not 0 <= encode_crf <= 51:
        raise typer.BadParameter("--encode-crf は0〜51です(目安18〜30)")
    if encode_width < 0:
        raise typer.BadParameter("--encode-width は0以上です(0=縮小しない)")
    d = Path(recordings_dir)
    if not d.is_dir():
        raise typer.BadParameter(f"ディレクトリがありません: {recordings_dir}")

    def progress(name: str, status: str, index: int, total: int) -> None:
        mark = {"ok": "✓", "skip": "-", "failed": "✗"}.get(status, "?")
        typer.echo(f"[{index}/{total}] {mark} {name} ({status})")

    typer.echo(f"再エンコード開始: {recordings_dir} (CRF={encode_crf})")
    done, skipped, failed = reencode_directory(
        d, crf=encode_crf, max_width=encode_width or None, force=force,
        on_progress=progress,
    )
    typer.echo(f"完了: 変換 {done} / スキップ {skipped} / 失敗 {failed}")


def reencode_cli() -> None:
    reencode_app()


if __name__ == "__main__":
    cli()
