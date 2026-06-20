# image_utils.py
import cv2
import numpy as np
import base64
from typing import Union

try:
    from PIL import Image
except Exception:
    Image = None

class ImageHelper:
    """Helpers to convert frames, encode for APIs, and save to disk."""

    @staticmethod
    def to_cv2(frame: Union[bytes, bytearray, np.ndarray, "Image.Image"], assume_rgb: bool = True) -> np.ndarray:
        if isinstance(frame, (bytes, bytearray)):
            arr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes")
            return img
        if Image and isinstance(frame, Image.Image):
            rgb = np.array(frame)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if assume_rgb else frame.copy()
            return frame.copy()
        raise TypeError(f"Unsupported frame type: {type(frame)}")

    @staticmethod
    def encode(cv_img: np.ndarray, fmt: str = "jpg", quality: int = 90, as_base64: bool = False) -> Union[bytes, str]:
        ext = ".jpg" if fmt.lower() in ("jpg", "jpeg") else ".png"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext == ".jpg" else []
        ok, buf = cv2.imencode(ext, cv_img, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        b = buf.tobytes()
        return base64.b64encode(b).decode("ascii") if as_base64 else b

    @staticmethod
    def save(cv_img: np.ndarray, path: str, quality: int = 90) -> None:
        ext = path.split(".")[-1].lower()
        if ext in ("jpg", "jpeg"):
            ok = cv2.imwrite(path, cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        else:
            ok = cv2.imwrite(path, cv_img)
        if not ok:
            raise IOError(f"Failed to save image to {path}")
        