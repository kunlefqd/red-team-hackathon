# ai_utils.py
import os
from typing import Optional

from google import genai
from google.genai import types

from image_utils import ImageHelper


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: set os.environ from KEY=VALUE lines if not already set."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


class AIUtils:
    """Thin wrapper around the Gemini API for sending camera frames."""

    MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        _load_dotenv()
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Pass api_key= or set GEMINI_API_KEY / GOOGLE_API_KEY."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model or self.MODEL

    def get_api_response(self, frame, prompt: str = "Describe what you see in this image.") -> str:
        """
        Send a frame to Gemini and return the text response.

        `frame` may be any type ImageHelper.to_cv2 understands:
        a cv2/numpy ndarray, raw image bytes, or a PIL Image.
        """
        # Normalise whatever we got into a cv2 BGR ndarray, then JPEG-encode it.
        cv_img = ImageHelper.to_cv2(frame, assume_rgb=False)
        jpeg_bytes = ImageHelper.encode(cv_img, fmt="jpg")

        image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
        )
        return response.text
