# ai_utils.py
import os
from typing import Optional

from google import genai
from google.genai import types

from image_utils import ImageHelper
from schema import VehicleType, VLMResponse


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

    _SYSTEM_PROMPT = """\
        You are the targeting module for an autonomous drone. The drone is in a room
        containing four vehicles: a tank, a boat, a jet, and an ice cream truck. The
        control loop will name one of these as the target.

        YOUR JOB
        Look at the 640x480 frame. Determine whether the named target is clearly visible.
        If yes, return:
        - target_visible: true
        - bbox_center_px: integer (x, y) pixel coordinates of the target's center
        - distance_band: far | medium | close | very_close
        - confidence: how sure you are it's the right vehicle

        If the target is not visible, or you can see it but can't identify it confidently:
        - target_visible: false
        - bbox_center_px: null
        - distance_band: null

        PIXEL COORDINATES
        Origin is top-left. x grows right (0 to 639). y grows down (0 to 479). Frame center
        is (320, 240). Be as precise as you can — the control loop converts these to a yaw
        angle and small errors get corrected on the next frame.

        CONFIDENCE
        Honest reporting beats false positives. Motion blur, partial occlusion, or a vehicle
        that looks similar to the target should drop confidence below 0.6.
        """

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

    def get_api_response(self, frame, target) -> VLMResponse:
        """
        Send a frame to Gemini and return a parsed VLMResponse.

        `frame` may be any type ImageHelper.to_cv2 understands:
        a cv2/numpy ndarray, raw image bytes, or a PIL Image.
        `target` is the vehicle to look for — a VehicleType or its string value
        (e.g. "tank", "boat", "jet", "ice_cream_truck").
        """
        target_name = target.value if isinstance(target, VehicleType) else str(target)
        prompt = f"Target: {target_name}. Locate it in this frame and report per the schema."

        # Normalise whatever we got into a cv2 BGR ndarray, then JPEG-encode it.
        cv_img = ImageHelper.to_cv2(frame, assume_rgb=False)
        jpeg_bytes = ImageHelper.encode(cv_img, fmt="jpg")

        image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                system_instruction=self._SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=VLMResponse,
                temperature=0.1,
            ),
        )
        return response.parsed
