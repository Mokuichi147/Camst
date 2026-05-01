from __future__ import annotations

import threading
import time

import cv2
import depthai as dai
import numpy as np

_ROTATE_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraStream:
    """OAK-D LITE のRGBフレームをバックグラウンドで取得して共有する。"""

    def __init__(
        self,
        size: tuple[int, int] = (1280, 720),
        fps: int = 30,
        rotate: int = 0,
    ) -> None:
        if rotate not in (0, 90, 180, 270):
            raise ValueError("rotate は 0/90/180/270 のいずれかです")
        self._size = size
        self._fps = fps
        self._rotate = rotate
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._actual_fps = 0.0

    @property
    def fps(self) -> float:
        return self._actual_fps

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def _run(self) -> None:
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            output = cam.requestOutput(
                self._size, dai.ImgFrame.Type.NV12, fps=float(self._fps)
            )
            queue = output.createOutputQueue()
            pipeline.start()

            last = time.time()
            count = 0
            while pipeline.isRunning() and not self._stop.is_set():
                pkt = queue.get()
                frame = pkt.getCvFrame()
                if self._rotate:
                    frame = cv2.rotate(frame, _ROTATE_CODE[self._rotate])
                with self._lock:
                    self._frame = frame
                count += 1
                now = time.time()
                if now - last >= 1.0:
                    self._actual_fps = count / (now - last)
                    count = 0
                    last = now
