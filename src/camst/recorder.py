from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camst.camera import BaseCameraStream


def _favorites_dir(directory: str | Path) -> Path:
    return Path(directory) / ".favorites"


def list_favorites(directory: str | Path) -> set[str]:
    """お気に入り登録されたクリップ名の集合を返す。

    お気に入りは .favorites ディレクトリ内の空マーカーファイルとして保持する
    (.thumbnails と同じファイルベースの仕組みで、再起動後も残る)。
    """
    fav_dir = _favorites_dir(directory)
    if not fav_dir.is_dir():
        return set()
    return {p.name for p in fav_dir.iterdir() if p.is_file()}


def set_favorite(directory: str | Path, name: str, favorite: bool) -> None:
    """クリップのお気に入り状態を設定する。name は検証済みのファイル名を渡す。"""
    fav_dir = _favorites_dir(directory)
    marker = fav_dir / name
    if favorite:
        fav_dir.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)
    else:
        try:
            marker.unlink()
        except OSError:
            pass


def list_clips(directory: str | Path) -> list[dict]:
    """保存済みクリップを新しい順に返す(ページ表示用)。"""
    d = Path(directory)
    if not d.is_dir():
        return []
    favorites = list_favorites(d)
    clips: list[dict] = []
    for p in sorted(d.glob("motion_*.*"), key=lambda p: p.name, reverse=True):
        try:
            stat = p.stat()
        except OSError:
            continue
        clips.append(
            {
                "name": p.name,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime),
                "favorite": p.name in favorites,
            }
        )
    return clips


def thumbnail_for_clip(path: str | Path) -> Path | None:
    """動画の先頭フレームからサムネイルWebPを作り、キャッシュパスを返す。"""
    clip = Path(path)
    try:
        clip_stat = clip.stat()
    except OSError:
        return None

    thumb_dir = clip.parent / ".thumbnails"
    thumb = thumb_dir / f"{clip.stem}.webp"
    try:
        if thumb.is_file() and thumb.stat().st_mtime >= clip_stat.st_mtime:
            return thumb
        thumb_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    frame = _read_thumbnail_frame(clip)
    if frame is None:
        return None

    if not cv2.imwrite(str(thumb), frame, [cv2.IMWRITE_WEBP_QUALITY, 80]):
        return None
    return thumb


def _read_thumbnail_frame(path: Path) -> np.ndarray | None:
    frame = _read_thumbnail_frame_av(path)
    if frame is not None:
        return frame
    return _read_thumbnail_frame_cv2(path)


def _read_thumbnail_frame_cv2(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    try:
        for _ in range(120):
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
    finally:
        cap.release()
    return None


def _is_faststart_mp4(path: Path) -> bool:
    """MP4 の moov が mdat より前にある(faststart 済み)かを判定する。

    トップレベルのボックスを順に読み、moov と mdat のどちらが先に現れるかで
    判断する。判定できないときは安全側に倒して False(未変換扱い)を返す。
    """
    try:
        with path.open("rb") as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    return False
                size = int.from_bytes(header[:4], "big")
                box_type = header[4:8]
                if box_type == b"moov":
                    return True
                if box_type == b"mdat":
                    return False
                if size == 1:
                    # 64bit の largesize。実サイズはこの 8 バイトに入る。
                    ext = f.read(8)
                    if len(ext) < 8:
                        return False
                    size = int.from_bytes(ext, "big")
                    skip = size - 16
                elif size == 0:
                    # EOF まで続く最後のボックス。moov/mdat 以外なら判定不能。
                    return False
                else:
                    skip = size - 8
                if skip < 0:
                    return False
                f.seek(skip, 1)
    except OSError:
        return False


def _remux_faststart(src: Path, dst: Path) -> bool:
    """MP4 の moov アトムを先頭へ移して dst に書き出す(プログレッシブ再生用)。

    OpenCV の VideoWriter は moov(再生に必要なインデックス)をファイル末尾に
    書くため、低速回線では動画全体を取得し終わるまで再生を開始できない。
    faststart で moov を先頭へ置くと、ブラウザは先頭から順次受信しながら
    再生を始められる。再エンコードはせずパケットをコピーするだけなので、
    画質の劣化や重い処理は発生しない。成功したら True を返す。
    """
    try:
        import av

        with av.open(str(src)) as in_container:
            in_stream = in_container.streams.video[0]
            with av.open(
                str(dst), mode="w", options={"movflags": "faststart"}
            ) as out_container:
                out_stream = out_container.add_stream_from_template(in_stream)
                for packet in in_container.demux(in_stream):
                    # demux は末尾にフラッシュ用の dts=None パケットを返すので除く。
                    if packet.dts is None:
                        continue
                    packet.stream = out_stream
                    out_container.mux(packet)
        return True
    except Exception:
        # 失敗時は中途半端な出力を残さない(呼び出し側がリネームへ退避する)。
        try:
            dst.unlink()
        except OSError:
            pass
        return False


def _read_thumbnail_frame_av(path: Path) -> np.ndarray | None:
    try:
        import av

        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                image = frame.to_ndarray(format="rgb24")
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception:
        return None
    return None


class MotionRecorder:
    """カメラの最新フレームを監視し、動きを検知している間だけ録画する。

    連続フレームの差分で動きを判定し、動き始めたら録画を開始する。静止が
    一定時間続くか上限時間に達したら停止してファイルを確定する。保存数は
    max_clips 件までで、超えた分は古いクリップから削除する(ストレージ節約)。
    瞬間的なノイズ等を避けるため、動きの継続が短いクリップは保存しない。
    """

    def __init__(
        self,
        camera: BaseCameraStream,
        directory: str | Path = "recordings",
        max_clips: int = 100,
        max_seconds: float = 60.0,
        fps: float = 15.0,
        min_area_ratio: float = 0.003,
        diff_threshold: int = 25,
        stop_after_idle: float = 3.0,
        min_motion_seconds: float = 1.0,
        start_after_motion: float = 0.4,
    ) -> None:
        self._camera = camera
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_clips = max_clips
        self._max_seconds = max_seconds
        self._fps = fps
        self._interval = 1.0 / fps
        self._min_area_ratio = min_area_ratio
        self._diff_threshold = diff_threshold
        self._stop_after_idle = stop_after_idle
        self._min_motion_seconds = min_motion_seconds
        # 単発のノイズで録画を始めないよう、動きがこの秒数ぶん連続してから開始する。
        self._start_frames = max(2, round(fps * start_after_motion))
        # ノイズ除去(オープニング)・隙間埋め(膨張)に使う構造要素。
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_gray: np.ndarray | None = None

    # --- ライフサイクル ---
    def start(self) -> None:
        if self._thread is not None:
            return
        self._cleanup_partials()
        # 既存録画の faststart 化は時間がかかりうるため、録画を妨げないよう
        # 別スレッドで進める。
        threading.Thread(target=self._migrate_faststart, daemon=True).start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def list_clips(self) -> list[dict]:
        return list_clips(self._dir)

    # --- 動き検知 ---
    def _is_moving(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        prev = self._prev_gray
        self._prev_gray = gray
        if prev is None or prev.shape != gray.shape:
            return False
        diff = cv2.absdiff(prev, gray)
        _, thresh = cv2.threshold(
            diff, self._diff_threshold, 255, cv2.THRESH_BINARY
        )
        # オープニングで孤立したノイズ点を除去し、膨張で領域の隙間を埋める。
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._kernel)
        thresh = cv2.dilate(thresh, self._kernel, iterations=1)
        # 画面全体に散ったノイズではなく、まとまった動きだけを動体とみなす。
        # 連結領域のうち最大の面積が一定割合を超えたときだけ「動き」と判定する。
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest = max((cv2.contourArea(c) for c in contours), default=0.0)
        return largest / thresh.size >= self._min_area_ratio

    # --- 録画 ---
    def _open_writer(
        self, size: tuple[int, int]
    ) -> tuple[cv2.VideoWriter, Path, Path]:
        # ブラウザ再生を優先して avc1(H.264)→mp4v の順に試す。利用可否は実行環境の
        # OpenCV ビルド依存なので isOpened() で実際に開けたものを採用する。
        # 録画中は隠しファイルへ書き出し、完成時に motion_* へリネームする。
        # VideoWriter は拡張子から形式を判定するため、末尾は .mp4/.avi のままにする。
        # こうすると書き込み途中の再生できないファイルが一覧に出ない。
        name = datetime.now().strftime("motion_%Y%m%d_%H%M%S")
        for fourcc_str, ext in (("avc1", ".mp4"), ("mp4v", ".mp4"), ("MJPG", ".avi")):
            final = self._dir / f"{name}{ext}"
            tmp = self._dir / f".{name}.part{ext}"
            writer = cv2.VideoWriter(
                str(tmp), cv2.VideoWriter_fourcc(*fourcc_str), self._fps, size
            )
            if writer.isOpened():
                return writer, tmp, final
            writer.release()
            try:
                tmp.unlink()
            except OSError:
                pass
        raise RuntimeError("録画用の VideoWriter を開けませんでした")

    def _prune(self) -> None:
        # お気に入りは削除対象から除外し、件数上限は非お気に入りにのみ適用する。
        favorites = list_favorites(self._dir)
        clips = sorted(
            self._dir.glob("motion_*.*"), key=lambda p: p.name, reverse=True
        )
        kept = 0
        for clip in clips:
            if clip.name in favorites:
                continue
            kept += 1
            if kept <= self._max_clips:
                continue
            try:
                clip.unlink()
            except OSError:
                pass

    def _finalize(self, tmp: Path, final: Path) -> None:
        # 録画完了。最終名へ確定して初めて一覧・配信の対象にする。
        # MP4 は faststart で moov を先頭へ移して確定し、低速回線でも素早く再生
        # 開始できるようにする。faststart に失敗した場合や MP4 以外は、書きかけを
        # そのままリネームして確定する(再生はできるので録画を失わない)。
        if final.suffix == ".mp4" and _remux_faststart(tmp, final):
            try:
                tmp.unlink()
            except OSError:
                pass
        else:
            try:
                tmp.rename(final)
            except OSError:
                # リネームできなければ書きかけを残さず捨てる。
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return
        self._prune()

    def _discard(self, tmp: Path) -> None:
        # 動きが短すぎたクリップは保存しない(瞬間的なノイズ等を弾く)。
        try:
            tmp.unlink()
        except OSError:
            pass

    def _close(self, tmp: Path, final: Path, motion_seconds: float) -> None:
        # 動きの継続が短ければ破棄、十分なら確定する。
        if motion_seconds >= self._min_motion_seconds:
            self._finalize(tmp, final)
        else:
            self._discard(tmp)

    def _cleanup_partials(self) -> None:
        # 前回クラッシュ等で残った書きかけ・変換中ファイルを起動時に掃除する。
        for pattern in (".motion_*.part.*", ".motion_*.*.part", ".motion_*.faststart.*"):
            for p in self._dir.glob(pattern):
                try:
                    p.unlink()
                except OSError:
                    pass

    def _migrate_faststart(self) -> None:
        # 既存の MP4 で moov が末尾のものを faststart へ変換する(起動時に一度)。
        # 変換は再エンコードを伴わないため安全だが、念のため一時ファイルへ書き出し、
        # 成功したものだけ置き換える(失敗時は元ファイルをそのまま残す)。
        for clip in self._dir.glob("motion_*.mp4"):
            if self._stop.is_set():
                return
            try:
                if _is_faststart_mp4(clip):
                    continue
            except OSError:
                continue
            tmp = self._dir / f".{clip.stem}.faststart.mp4"
            if not _remux_faststart(clip, tmp):
                continue
            try:
                # prune 等で元が消えていれば変換結果は捨てる(復活させない)。
                if clip.exists():
                    tmp.replace(clip)
                else:
                    tmp.unlink()
            except OSError:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _run(self) -> None:
        writer: cv2.VideoWriter | None = None
        tmp_path: Path | None = None
        final_path: Path | None = None
        started_at = 0.0
        last_motion = 0.0
        motion_frames = 0  # 実際に動きを検知したフレーム数(最低秒数の判定に使う)
        pending_motion = 0  # 録画開始前に動きが連続したフレーム数

        try:
            while not self._stop.is_set():
                tick = time.time()
                frame = self._camera.latest()
                if frame is None:
                    time.sleep(self._interval)
                    continue

                moving = self._is_moving(frame)

                if writer is None:
                    # 動きが連続したときだけ録画を始める(単発ノイズを弾く)。
                    pending_motion = pending_motion + 1 if moving else 0
                    if pending_motion >= self._start_frames:
                        h, w = frame.shape[:2]
                        try:
                            writer, tmp_path, final_path = self._open_writer((w, h))
                        except RuntimeError:
                            writer = None
                        else:
                            started_at = last_motion = tick
                            motion_frames = pending_motion
                            writer.write(frame)
                        pending_motion = 0
                else:
                    writer.write(frame)
                    if moving:
                        last_motion = tick
                        motion_frames += 1
                    over_max = tick - started_at >= self._max_seconds
                    idle = tick - last_motion >= self._stop_after_idle
                    if over_max or idle:
                        writer.release()
                        writer = None
                        self._close(
                            tmp_path, final_path, motion_frames / self._fps
                        )

                # 処理時間を差し引いて約 fps を維持する。
                elapsed = time.time() - tick
                if elapsed < self._interval:
                    time.sleep(self._interval - elapsed)
        finally:
            if writer is not None:
                writer.release()
                self._close(tmp_path, final_path, motion_frames / self._fps)
