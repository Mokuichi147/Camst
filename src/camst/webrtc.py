from __future__ import annotations

import asyncio

import av
from aiortc import VideoStreamTrack

from camst.camera import CameraStream


class CameraVideoTrack(VideoStreamTrack):
    """CameraStream の最新フレームを WebRTC ビデオトラックとして配信する。"""

    kind = "video"

    def __init__(self, camera: CameraStream) -> None:
        super().__init__()
        self._camera = camera

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()
        arr = self._camera.latest()
        while arr is None:
            await asyncio.sleep(0.01)
            arr = self._camera.latest()
        frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame
