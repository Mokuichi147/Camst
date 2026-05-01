from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import depthai as dai


class HLSStream:
    """OAK-D LITE の H.264 出力を ffmpeg 経由で HLS セグメントに変換する。"""

    def __init__(
        self,
        size: tuple[int, int] = (1280, 720),
        fps: int = 30,
        segment_duration: float = 1.0,
        ffmpeg: str = "ffmpeg",
    ) -> None:
        self._size = size
        self._fps = fps
        self._seg = segment_duration
        self._ffmpeg_bin = ffmpeg
        self._dir = Path(tempfile.mkdtemp(prefix="camst-hls-"))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ffmpeg: subprocess.Popen[bytes] | None = None
        self._actual_fps = 0.0

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def playlist_name(self) -> str:
        return "stream.m3u8"

    @property
    def playlist(self) -> Path:
        return self._dir / self.playlist_name

    @property
    def fps(self) -> float:
        return self._actual_fps

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # プレイリストが生成されるのを待つ(最長10秒)
        for _ in range(100):
            if self.playlist.exists():
                break
            time.sleep(0.1)

    def stop(self) -> None:
        self._stop.set()
        if self._ffmpeg is not None:
            try:
                if self._ffmpeg.stdin:
                    self._ffmpeg.stdin.close()
            except Exception:
                pass
            self._ffmpeg.terminate()
            try:
                self._ffmpeg.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._ffmpeg.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)
        shutil.rmtree(self._dir, ignore_errors=True)

    def _start_ffmpeg(self) -> subprocess.Popen[bytes]:
        cmd = [
            self._ffmpeg_bin,
            "-loglevel", "error",
            "-fflags", "+genpts",
            "-f", "h264",
            "-r", str(self._fps),
            "-i", "pipe:0",
            "-c", "copy",
            "-f", "hls",
            "-hls_time", str(self._seg),
            "-hls_list_size", "6",
            "-hls_flags", "delete_segments+independent_segments+omit_endlist",
            "-hls_segment_type", "mpegts",
            "-hls_segment_filename", str(self._dir / "seg_%05d.ts"),
            str(self.playlist),
        ]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _run(self) -> None:
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            output = cam.requestOutput(
                self._size, dai.ImgFrame.Type.NV12, fps=float(self._fps)
            )
            encoder = pipeline.create(dai.node.VideoEncoder)
            encoder.setDefaultProfilePreset(
                self._fps, dai.VideoEncoderProperties.Profile.H264_MAIN
            )
            output.link(encoder.input)
            queue = encoder.bitstream.createOutputQueue()

            self._ffmpeg = self._start_ffmpeg()
            assert self._ffmpeg.stdin is not None
            pipeline.start()

            last = time.time()
            count = 0
            while pipeline.isRunning() and not self._stop.is_set():
                pkt = queue.get()
                data = bytes(pkt.getData())
                try:
                    self._ffmpeg.stdin.write(data)
                    self._ffmpeg.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
                count += 1
                now = time.time()
                if now - last >= 1.0:
                    self._actual_fps = count / (now - last)
                    count = 0
                    last = now
