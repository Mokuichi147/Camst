from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camst.camera import BaseCameraStream


def list_clips(directory: str | Path) -> list[dict]:
    """保存済みクリップを新しい順に返す(ページ表示用)。"""
    d = Path(directory)
    if not d.is_dir():
        return []
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
            }
        )
    return clips


class MotionRecorder:
    """カメラの最新フレームを監視し、動きを検知している間だけ録画する。

    連続フレームの差分で動きを判定し、動き始めたら録画を開始する。静止が
    一定時間続くか上限時間に達したら停止してファイルを確定する。保存数は
    max_clips 件までで、超えた分は古いクリップから削除する(ストレージ節約)。
    """

    def __init__(
        self,
        camera: BaseCameraStream,
        directory: str | Path = "recordings",
        max_clips: int = 30,
        max_seconds: float = 60.0,
        fps: float = 15.0,
        min_area_ratio: float = 0.005,
        diff_threshold: int = 25,
        stop_after_idle: float = 3.0,
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

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_gray: np.ndarray | None = None

    # --- ライフサイクル ---
    def start(self) -> None:
        if self._thread is not None:
            return
        self._cleanup_partials()
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
        ratio = cv2.countNonZero(thresh) / thresh.size
        return ratio >= self._min_area_ratio

    # --- 録画 ---
    def _open_writer(
        self, size: tuple[int, int]
    ) -> tuple[cv2.VideoWriter, Path, Path]:
        # ブラウザ再生を優先して avc1(H.264)→mp4v の順に試す。利用可否は実行環境の
        # OpenCV ビルド依存なので isOpened() で実際に開けたものを採用する。
        # 録画中は隠しの .part へ書き出し、完成時に motion_* へリネームする。
        # こうすると書き込み途中の再生できないファイルが一覧に出ない。
        name = datetime.now().strftime("motion_%Y%m%d_%H%M%S")
        for fourcc_str, ext in (("avc1", ".mp4"), ("mp4v", ".mp4"), ("MJPG", ".avi")):
            final = self._dir / f"{name}{ext}"
            tmp = self._dir / f".{name}{ext}.part"
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
        clips = sorted(
            self._dir.glob("motion_*.*"), key=lambda p: p.name, reverse=True
        )
        for old in clips[self._max_clips :]:
            try:
                old.unlink()
            except OSError:
                pass

    def _finalize(self, tmp: Path, final: Path) -> None:
        # 録画完了。最終名へリネームして初めて一覧・配信の対象にする。
        # リネームできなければ書きかけを残さず捨てる。
        try:
            tmp.rename(final)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            return
        self._prune()

    def _cleanup_partials(self) -> None:
        # 前回クラッシュ等で残った書きかけ(.part)を起動時に掃除する。
        for p in self._dir.glob(".motion_*.part"):
            try:
                p.unlink()
            except OSError:
                pass

    def _run(self) -> None:
        writer: cv2.VideoWriter | None = None
        tmp_path: Path | None = None
        final_path: Path | None = None
        started_at = 0.0
        last_motion = 0.0

        try:
            while not self._stop.is_set():
                tick = time.time()
                frame = self._camera.latest()
                if frame is None:
                    time.sleep(self._interval)
                    continue

                moving = self._is_moving(frame)

                if writer is None:
                    if moving:
                        h, w = frame.shape[:2]
                        try:
                            writer, tmp_path, final_path = self._open_writer((w, h))
                        except RuntimeError:
                            writer = None
                        else:
                            started_at = last_motion = tick
                            writer.write(frame)
                else:
                    writer.write(frame)
                    if moving:
                        last_motion = tick
                    over_max = tick - started_at >= self._max_seconds
                    idle = tick - last_motion >= self._stop_after_idle
                    if over_max or idle:
                        writer.release()
                        writer = None
                        self._finalize(tmp_path, final_path)

                # 処理時間を差し引いて約 fps を維持する。
                elapsed = time.time() - tick
                if elapsed < self._interval:
                    time.sleep(self._interval - elapsed)
        finally:
            if writer is not None:
                writer.release()
                self._finalize(tmp_path, final_path)
