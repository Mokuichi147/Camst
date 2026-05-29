from __future__ import annotations

import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

_ROTATE_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class BaseCameraStream:
    """カメラフレームをバックグラウンドスレッドで取得して共有する基底クラス。"""

    def __init__(self, rotate: int = 0) -> None:
        if rotate not in (0, 90, 180, 270):
            raise ValueError("rotate は 0/90/180/270 のいずれかです")
        self._rotate = rotate
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._actual_fps = 0.0
        self._fps_last = 0.0
        self._fps_count = 0

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

    def _publish(self, frame: np.ndarray) -> None:
        """取得したフレームに回転を適用して共有し、実測FPSを更新する。"""
        if self._rotate:
            frame = cv2.rotate(frame, _ROTATE_CODE[self._rotate])
        with self._lock:
            self._frame = frame
        self._fps_count += 1
        now = time.time()
        if now - self._fps_last >= 1.0:
            self._actual_fps = self._fps_count / (now - self._fps_last)
            self._fps_count = 0
            self._fps_last = now

    def _run(self) -> None:  # pragma: no cover - サブクラスで実装
        raise NotImplementedError


class OakCameraStream(BaseCameraStream):
    """OAK-D LITE のRGBフレームを depthai 経由で取得する。"""

    def __init__(
        self,
        size: tuple[int, int] = (1280, 720),
        fps: int = 30,
        rotate: int = 0,
    ) -> None:
        super().__init__(rotate)
        self._size = size
        self._fps = fps

    def _run(self) -> None:
        import depthai as dai

        self._fps_last = time.time()
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            output = cam.requestOutput(
                self._size, dai.ImgFrame.Type.NV12, fps=float(self._fps)
            )
            queue = output.createOutputQueue()
            pipeline.start()

            while pipeline.isRunning() and not self._stop.is_set():
                pkt = queue.get()
                self._publish(pkt.getCvFrame())


class UvcCameraStream(BaseCameraStream):
    """UVC（標準USBカメラ。Leap Motion 等）のフレームを OpenCV 経由で取得する。"""

    def __init__(
        self,
        device: int | str = "Leap Motion",
        size: tuple[int, int] | None = None,
        fps: int = 30,
        rotate: int = 0,
    ) -> None:
        super().__init__(rotate)
        # device: 数値ならインデックス、文字列ならデバイス名の一部として検索する。
        self._device = device
        self._size = size
        self._fps = fps

    def _resolve_index(self) -> int:
        if isinstance(self._device, int):
            return self._device
        index = find_camera_index(self._device)
        if index is None:
            raise RuntimeError(
                f"名前に '{self._device}' を含むカメラが見つかりません。"
                "接続状態を確認するか --device に番号を指定してください。"
            )
        return index

    def _run(self) -> None:
        index = self._resolve_index()
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"カメラ(index={index})を開けませんでした")
        if self._size is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        cap.set(cv2.CAP_PROP_FPS, float(self._fps))

        self._fps_last = time.time()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                self._publish(frame)
        finally:
            cap.release()


class LeapCameraStream(BaseCameraStream):
    """Leap Motion のステレオIR映像を PyAV(avfoundation) 経由で取得する。

    Leap は左右2眼のグレースケールを yuvs(YUYV422) の2バイトに交互パッキングして
    出力する。cv2 は macOS で必ず BGR へ変換してしまい生バイトが取れないため、
    PyAV で yuyv422 のまま受け取り、左右に分離する。

    yuyv422 のフレームは (高さ, 幅, 2) で、ch 0 が片眼、ch 1 がもう片眼にあたる。
    """

    # Leap が対応する (幅, 高さ): デバイスが返す正確なネイティブfps。
    _MODES = {
        (640, 480): "57.500014375003595",
        (640, 240): "115.00069000414003",
        (752, 480): "50",
        (752, 240): "100",
    }

    def __init__(
        self,
        device: int | str = "Leap Motion",
        size: tuple[int, int] = (640, 480),
        fps: str | None = None,
        rotate: int = 0,
        eye: str = "left",
        correct: bool = False,
        clahe_clip: float = 2.0,
        denoise: int = 1,
        nlm: bool = False,
        nlm_h: float = 1.0,
        nlm_scale: float = 2.0,
        nlm_template: int = 7,
        nlm_search: int = 21,
    ) -> None:
        if eye not in ("left", "right", "both"):
            raise ValueError("eye は left/right/both のいずれかです")
        if denoise < 1:
            raise ValueError("denoise は1以上です")
        super().__init__(rotate)
        self._device = device
        self._size = size
        self._fps = fps if fps is not None else self._MODES.get(size, "50")
        self._eye = eye
        # 暗い赤外像の明るさ補正: CLAHE(コントラスト制限付き適応ヒストグラム平坦化)
        self._clahe = (
            cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            if correct
            else None
        )
        # 時間方向の移動平均によるノイズ低減(直近 denoise フレーム)
        self._denoise = denoise
        self._buf: deque[np.ndarray] = deque(maxlen=denoise)
        # 空間NLMeansによる強力なノイズ除去(エッジ保存)。
        # 重いので nlm_scale で縮小してから適用し拡大して戻す(コスト約 1/scale^2)。
        self._nlm = nlm
        self._nlm_h = nlm_h
        self._nlm_scale = nlm_scale
        self._nlm_template = nlm_template
        self._nlm_search = nlm_search

    def _resolve_index(self) -> int:
        if isinstance(self._device, int):
            return self._device
        index = find_camera_index(self._device)
        if index is None:
            raise RuntimeError(
                f"名前に '{self._device}' を含むカメラが見つかりません。"
                "接続を確認するか --device に番号を指定してください。"
            )
        return index

    def _split_eyes(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # leapuvc と同じ手順: 生バイトを (高さ, 幅*2) に並べ、左右IRが1バイトずつ
        # 交互に並んでいるので偶数列=左眼・奇数列=右眼として分離する。
        w, h = self._size
        raw = flat.reshape(-1)[: h * w * 2].reshape(h, w * 2)
        return raw[:, 0::2], raw[:, 1::2]

    def _select(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if self._eye == "right":
            return right
        if self._eye == "both":
            # 左右の眼を入れ替えて横並びにする（交差法での立体視向け）。
            return np.hstack([right, left])
        return left

    def _apply_nlm(self, gray: np.ndarray) -> np.ndarray:
        # 重い NLMeans を縮小画像に適用してから元サイズに戻す(高速化)。
        if self._nlm_scale > 1.0:
            h0, w0 = gray.shape[:2]
            small = cv2.resize(
                gray, None,
                fx=1.0 / self._nlm_scale, fy=1.0 / self._nlm_scale,
                interpolation=cv2.INTER_AREA,
            )
            small = cv2.fastNlMeansDenoising(
                small, None, self._nlm_h, self._nlm_template, self._nlm_search
            )
            return cv2.resize(small, (w0, h0), interpolation=cv2.INTER_LINEAR)
        return cv2.fastNlMeansDenoising(
            gray, None, self._nlm_h, self._nlm_template, self._nlm_search
        )

    def _emit(self, flat: np.ndarray) -> None:
        gray = np.ascontiguousarray(self._select(*self._split_eyes(flat)))
        if self._denoise > 1:
            self._buf.append(gray)
            gray = np.mean(self._buf, axis=0).astype(np.uint8)
        # 先にCLAHEで暗い遠方を持ち上げ、その後にNLMeansで増えたノイズを除去する。
        # (順序を逆にすると遠方の微弱なディテールが先に潰れてしまう)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        if self._nlm:
            gray = self._apply_nlm(gray)
        self._publish(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    def _run(self) -> None:
        if sys.platform == "darwin":
            self._run_avfoundation()
        else:
            self._run_v4l2()

    def _run_avfoundation(self) -> None:
        # macOS: cv2 は生バイトを渡さないため PyAV(avfoundation) で取得する。
        import av
        import av.error

        index = self._resolve_index()
        w, h = self._size
        options = {
            "video_size": f"{w}x{h}",
            # ネイティブ形式を指定して ffmpeg による色変換(=映像破壊)を避ける。
            "pixel_format": "uyvy422",
            "framerate": str(self._fps),
        }
        container = av.open(str(index), format="avfoundation", options=options)
        self._fps_last = time.time()
        try:
            while not self._stop.is_set():
                try:
                    for frame in container.decode(video=0):
                        if self._stop.is_set():
                            break
                        self._emit(np.frombuffer(bytes(frame.planes[0]), np.uint8))
                except av.error.BlockingIOError:
                    # まだフレームが届いていない。少し待って再試行する。
                    time.sleep(0.005)
                except av.error.EOFError:
                    break
        finally:
            container.close()

    def _v4l2_candidates(self) -> list[int]:
        # 番号指定ならそれを、名前指定なら一致する全ノードを候補にする。
        # Leap はキャプチャ用とメタデータ用など複数の video ノードを持つため。
        if isinstance(self._device, int):
            return [self._device]
        import glob
        import re

        cands = []
        for path in glob.glob("/sys/class/video4linux/video*/name"):
            try:
                with open(path) as f:
                    name = f.read().strip()
            except OSError:
                continue
            if str(self._device).lower() in name.lower():
                cands.append(int(re.search(r"/video(\d+)/name$", path).group(1)))
        return sorted(cands)

    def _open_v4l2(self) -> cv2.VideoCapture:
        # 候補ノードを順に開き、実際にフレームが取れたものを採用する。
        w, h = self._size
        candidates = self._v4l2_candidates()
        if not candidates:
            raise RuntimeError(
                f"名前に '{self._device}' を含むカメラが見つかりません。"
                "接続を確認するか --device に番号を指定してください。"
            )
        for index in candidates:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # 生のYUYVを取得する
                if cap.read()[0]:
                    return cap
            cap.release()
        raise RuntimeError(
            f"Leap のキャプチャ可能な V4L2 デバイスが見つかりません（試行: {candidates}）"
        )

    def _run_v4l2(self) -> None:
        # Linux(Raspberry Pi等): leapuvc 同様 cv2+V4L2 で生バイトを取得する。
        cap = self._open_v4l2()
        self._fps_last = time.time()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                self._emit(np.asarray(frame, dtype=np.uint8))
        finally:
            cap.release()


def find_camera_index(name_keyword: str) -> int | None:
    """名前にキーワードを含むビデオデバイスのインデックスを返す。"""
    if sys.platform != "darwin":
        return _find_camera_index_v4l2(name_keyword)
    return _find_camera_index_avfoundation(name_keyword)


def _find_camera_index_v4l2(name_keyword: str) -> int | None:
    """Linux: /sys/class/video4linux から名前一致するデバイス番号を返す。"""
    import glob
    import re

    candidates = []
    for path in glob.glob("/sys/class/video4linux/video*/name"):
        try:
            with open(path) as f:
                name = f.read().strip()
        except OSError:
            continue
        if name_keyword.lower() in name.lower():
            num = int(re.search(r"/video(\d+)/name$", path).group(1))
            candidates.append(num)
    # 同名ノードが複数ある場合は、通常キャプチャ可能な最小番号を使う。
    return min(candidates) if candidates else None


def _find_camera_index_avfoundation(name_keyword: str) -> int | None:
    """macOS: libavdevice(PyAV) のデバイス一覧をそのまま解析する。
    PyAV でのキャプチャ時に渡すインデックスと一致する。"""
    import re

    import av
    import av.logging

    av.logging.set_level(av.logging.VERBOSE)
    with av.logging.Capture() as logs:
        try:
            av.open("", format="avfoundation", options={"list_devices": "true"})
        except Exception:
            pass

    in_video = False
    for _level, _name, message in logs:
        if "video devices:" in message:
            in_video = True
            continue
        if "audio devices:" in message:
            in_video = False
            continue
        if not in_video:
            continue
        m = re.match(r"\s*\[(\d+)\]\s+(.*?)\s*$", message)
        if m and name_keyword.lower() in m.group(2).lower():
            return int(m.group(1))
    return None


def create_camera(
    source: str = "oak",
    device: int | str = "Leap Motion",
    rotate: int = 0,
    fps: int = 30,
    eye: str = "left",
    correct: bool = False,
    clahe_clip: float = 2.0,
    denoise: int = 1,
    nlm: bool = False,
    nlm_h: float = 1.0,
    nlm_scale: float = 2.0,
    nlm_template: int = 7,
    nlm_search: int = 21,
) -> BaseCameraStream:
    """source に応じたカメラストリームを生成する。"""
    if source == "oak":
        return OakCameraStream(fps=fps, rotate=rotate)
    dev: int | str = int(device) if str(device).isdigit() else device
    if source == "leap":
        return LeapCameraStream(
            device=dev, rotate=rotate, eye=eye,
            correct=correct, clahe_clip=clahe_clip, denoise=denoise,
            nlm=nlm, nlm_h=nlm_h, nlm_scale=nlm_scale,
            nlm_template=nlm_template, nlm_search=nlm_search,
        )
    if source == "uvc":
        return UvcCameraStream(device=dev, fps=fps, rotate=rotate)
    raise ValueError("source は 'oak' / 'leap' / 'uvc' のいずれかです")


# 後方互換: 既存コードが参照する名前を維持する。
CameraStream = OakCameraStream
