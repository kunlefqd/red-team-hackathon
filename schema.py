"""Pydantic models for the vehicle-targeting VLM call."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VehicleType(str, Enum):
    TANK = "tank"
    BOAT = "boat"
    JET = "jet"
    ICE_CREAM_TRUCK = "ice_cream_truck"


class DistanceBand(str, Enum):
    FAR = "far"
    MEDIUM = "medium"
    CLOSE = "close"
    VERY_CLOSE = "very_close"


class VLMResponse(BaseModel):
    scene_description: str = Field(
        description="One sentence describing what's visible in the frame."
    )

    target_visible: bool = Field(
        description="True only if the requested target vehicle is clearly identifiable "
                    "in the frame. False if absent, occluded, or ambiguous."
    )

    bbox_center_px: Optional[tuple[int, int]] = Field(
        default=None,
        description="(x, y) integer pixel coordinates of the target's center in a "
                    "640x480 frame, origin at top-left. Null if target_visible is false.",
    )

    distance_band: Optional[DistanceBand] = Field(
        default=None,
        description="Categorical distance estimate based on how much of the frame "
                    "the target occupies. Null if target_visible is false.",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the identification, 0.0 to 1.0. Below 0.6 the "
                    "control loop will hover and re-query.",
    )

    notes: Optional[str] = Field(
        default=None,
        description="Optional free text — occlusion, ambiguity, motion blur, etc.",
    )