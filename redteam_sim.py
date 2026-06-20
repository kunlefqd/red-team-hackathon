"""redteam_sim.py - helper library for the Red Team Hack Sim drone challenge.

Everything goes through one ProjectAirSim client connection -- control,
telemetry, and camera. No Docker, no PX4, no MAVLink.

    from redteam_sim import connect, reset, read_frame

    client, world, drone = connect()         # connect + spawn Drone1 at the start line
    drone.enable_api_control()
    drone.arm()
    await (await drone.takeoff_async())       # *_async() returns a task; await it to finish

    frame = read_frame(drone)                 # BGR numpy image -> your vision input
    state = drone.get_estimated_kinematics()  # onboard state estimate (allowed in HARD)

    # ... your autonomy: read frames + telemetry, call drone.move_to_position_async(...)

    reset(drone)                              # teleport back to the start line to retry
    client.disconnect()                       # when you're done

Coordinates are NED metres: +north, +east, +DOWN (so climbing is NEGATIVE z).
"""
from pathlib import Path

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.types import Pose
from projectairsim.utils import unpack_image

SIM_CONFIG_DIR = str(Path(__file__).resolve().parent / "sim_config")
SCENE = "sf_scene.jsonc"          # bundled challenge scene (SimpleFlight drone)
DRONE_NAME = "Drone1"
CAMERA = "FPV"                     # nose cam; scene (RGB) image is type 0

# The start line == the drone's spawn origin in sf_scene.jsonc (NED metres).
START_POSE = Pose(
    {
        "translation": {"x": 0.0, "y": -35.0, "z": -0.1},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
)


def connect(address: str = "127.0.0.1"):
    """Connect to a running sim, load the scene (spawns Drone1 at the start line),
    and return (client, world, drone). Loading the scene is what registers the
    flight-control services, so always go through here."""
    client = ProjectAirSimClient(address=address)
    client.connect()
    world = World(client, SCENE, delay_after_load_sec=2, sim_config_path=SIM_CONFIG_DIR)
    drone = Drone(client, world, DRONE_NAME)
    return client, world, drone


def reset(drone) -> None:
    """Teleport the drone back to the start line (velocity zeroed) to retry a run.
    Returning within the RaceManager's start radius re-arms the race clock.

    This teleport is the ONE sanctioned use of set_pose -- teleporting anywhere
    else is off-limits (see HARD mode in the README)."""
    drone.set_pose(START_POSE, reset_kinematics=True)


def read_frame(drone, camera: str = CAMERA):
    """Grab the latest camera frame as a BGR numpy array (your vision input).
    Blocks until the next captured frame is available; call it in your loop.
    Returns None if no frame is ready yet."""
    images = drone.get_images(camera, [0])   # 0 = scene (RGB) image type
    img = images.get(0)
    return unpack_image(img) if img is not None else None
