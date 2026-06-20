"""Pydantic models for the VLM perception output."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class VehicleType(str, Enum):
    TANK = "tank"
    BOAT = "boat"
    JET = "jet"
    ICE_CREAM_TRUCK = "ice_cream_truck"


class ScreenPosition(str, Enum):
    FAR_LEFT = "far_left"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    FAR_RIGHT = "far_right"


class VerticalPosition(str, Enum):
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


class DistanceBand(str, Enum):
    FAR = "far"
    MEDIUM = "medium"
    CLOSE = "close"
    VERY_CLOSE = "very_close"


class ApparentSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ChosenAction(str, Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    FORWARD = "forward"
    HOVER = "hover"
    LAND = "land"
    APPROACH_TANK = "approach_tank"
    APPROACH_BOAT = "approach_boat"
    APPROACH_JET = "approach_jet"
    APPROACH_TRUCK = "approach_truck"


class MemoryKey(str, Enum):
    FIRST_TURN = "first_turn"
    SECOND_TURN = "second_turn"
    SPHERE_COUNT = "sphere_count"
    TARGET_VEHICLE = "target_vehicle"


# ─── Observation blocks ─────────────────────────────────────────

class ArrowObservation(BaseModel):
    visible: bool = Field(description="Whether any arrows are in the current frame.")
    red_arrow_direction: Optional[Direction] = Field(
        default=None,
        description="Direction the RED arrow points. Null if not visible.",
    )
    green_arrow_direction: Optional[Direction] = Field(
        default=None,
        description="Direction the GREEN arrow points. Determines the first turn.",
    )


class SphereObservation(BaseModel):
    visible: bool
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Number of blue spheres, 1-5. Null if uncountable.",
    )


class VehicleSighting(BaseModel):
    type: VehicleType
    screen_position: ScreenPosition
    vertical_position: VerticalPosition = VerticalPosition.MIDDLE
    apparent_size: ApparentSize
    distance_band: DistanceBand = Field(
        description="Categorical distance estimate based on how much of the frame "
                    "the vehicle occupies."
    )


class VehicleObservation(BaseModel):
    visible: bool
    items: list[VehicleSighting] = Field(default_factory=list)


class MemoryUpdate(BaseModel):
    key: MemoryKey
    value: str = Field(description="Value to store. Stringified — control loop parses.")


# ─── Top-level response ─────────────────────────────────────────

class VLMResponse(BaseModel):
    scene_description: str = Field(
        description="One sentence describing what's in the frame."
    )
    arrows: ArrowObservation
    spheres: SphereObservation
    vehicles: VehicleObservation
    chosen_action: ChosenAction
    memory_updates: list[MemoryUpdate] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: Optional[str] = None