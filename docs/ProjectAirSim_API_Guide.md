# Red Team Hack Sim — Python API Guide

The deeper API reference behind the **How to Play** briefing (`README.md`). Read the
briefing first for the rules; this is the detail on flying, sensing, and navigating from
Python.

You pilot a drone (flown by the in-sim **SimpleFlight** controller) through the level using
two data sources — **telemetry** and the **camera** — all over one ProjectAirSim client
connection. No Docker, no PX4, no MAVLink.

You work through the `redteam_sim` helper library, which wraps the raw ProjectAirSim client.
The raw `drone.*` API is still there for movement and sensing — this guide shows the parts
you'll actually use.

---

## 1. Setup (one time)

Requires **Python 3.12** — the bundled wheels are built for it. From this directory,
install **offline** from `./wheels` (no internet needed):

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --no-index --find-links wheels -r requirements.txt
```

If you're online and on a different Python, drop `--no-index` to let pip fall back to PyPI.

---

## 2. Run it

**Launch the game** (it hosts the sim server on ports 8989 / 8990 while running). Pass the
map as the first argument:

```bash
./Red_Team_Hack_Sim.sh RedRoad -windowed -ResX=1280 -ResY=720      # Linux
Red_Team_Hack_Sim.exe RedRoad -windowed -ResX=1280 -ResY=720       # Windows (cmd)
```

See the briefing for the full option list (`-headless` equivalents, quality, `RedRoad2`).
If a launch fails because a previous run is still holding the port, kill the stale process:
`pkill -f Red_Team_Hack_Sim/Binaries`.

**Then, in another terminal (same venv):**

```bash
python fly.py                        # starter: takeoff, read camera + telemetry, land
```

Build your autonomy by editing `fly.py` (look for `YOUR AUTONOMY GOES HERE`) or start fresh
with the helper library below.

---

## 3. The helper library (`redteam_sim`)

```python
from redteam_sim import connect, reset, read_frame
```

| Function | Returns | What it does |
|----------|---------|--------------|
| `connect(address="127.0.0.1")` | `(client, world, drone)` | Connects and loads `sf_scene.jsonc`, which spawns `Drone1` at the start line. Loading the scene registers the flight-control services, so **always go through `connect()`**. |
| `reset(drone)` | `None` | Teleports the drone back to the start line (velocity zeroed) to retry. This is the **one sanctioned use of `set_pose`**. |
| `read_frame(drone, camera="FPV")` | BGR numpy array, or `None` | Latest frame from the nose camera as an OpenCV-ready BGR image. Blocks until the next frame is available — call it in your loop. |

---

## 4. Flight lifecycle

```python
import asyncio
from redteam_sim import connect, reset, read_frame

async def main():
    client, world, drone = connect()        # connect + spawn at the start line
    try:
        drone.enable_api_control()
        drone.arm()

        await (await drone.takeoff_async())             # see double-await note below
        await (await drone.move_to_position_async(0.0, -35.0, -5.0, 3.0))  # climb to 5 m

        # --- your autonomy loop ---
        frame = read_frame(drone)                       # vision input (BGR)
        state = drone.get_estimated_kinematics()        # telemetry (HARD-safe)
        # read the clues, decide the next waypoint, then move...

        await (await drone.land_async())
        drone.disarm()
    finally:
        client.disconnect()

asyncio.run(main())
```

### Double-await — read this

`*_async()` commands **return a task immediately**; awaiting that task is what blocks until
the maneuver finishes. So you await twice:

```python
await (await drone.move_to_position_async(n, e, d, speed))
```

A single `await` only retrieves the task — your code races ahead of the drone. The starter
wraps this in a helper:

```python
async def do(cmd):
    await (await cmd)

await do(drone.move_to_position_async(5.0, -35.0, -5.0, 3.0))
```

---

## 5. Coordinate system & units

- **NED frame** — North-East-Down. `+X = North`, `+Y = East`, **`+Z = Down`**.
  Climbing is **negative Z**. The drone spawns at the start line `(0, -35, -0.1)`.
- **Distances** in meters, **velocities** in m/s.
- **Angles in radians** unless the parameter name ends in `-deg`.
- **Quaternions** are `{w, x, y, z}`, scalar-first.

---

## 6. Movement

All `*_async`, all NED meters / m/s. Remember the double-await. `move_to_position_async` is
your workhorse.

| Method | Key params | What it does |
|--------|-----------|--------------|
| `takeoff_async(timeout_sec=20)` | — | Auto-takeoff to ~3 m. Drone must be still first. |
| `land_async()` | — | Auto-land at current position. |
| `hover_async()` | — | Hold current position. |
| `move_to_position_async(north, east, down, velocity)` | NED m + m/s | Fly to an absolute NED point. |
| `move_on_path_async(path, velocity)` | `[[n,e,d], ...]` | Follow a list of NED waypoints. |
| `move_by_velocity_async(v_north, v_east, v_down, duration=...)` | world-frame m/s | Fly a velocity vector for `duration` seconds. |
| `move_by_velocity_body_frame_async(v_forward, v_right, v_down, duration=...)` | body-frame m/s | Velocity relative to where the nose points. |
| `move_by_heading_async(heading, speed, v_down=0.0, duration=...)` | heading rad, m/s | Fly along a compass heading. |
| `rotate_to_yaw_async(yaw)` | yaw rad | Turn to face an absolute yaw. |
| `rotate_by_yaw_rate_async(yaw_rate, duration)` | rad/s, s | Spin at a rate — handy for scanning the scene. |
| `set_controls(roll_rate, pitch_rate, yaw_rate, throttle)` | rad/s + 0..1 | Low-level / acro (synchronous, no await). |

Cancel an in-progress maneuver: `drone.cancel_last_task()` (synchronous).

---

## 7. Telemetry

Synchronous — call and get an immediate value, no `await`. Pass the sensor **id** from the
config (e.g. `"IMU1"`), not the type.

```python
state = drone.get_estimated_kinematics()  # onboard fused estimate — allowed in HARD
pos = state["pose"]["position"]           # {"x":..., "y":..., "z":...}  NED meters
ori = state["pose"]["orientation"]        # {"w":..., "x":..., "y":..., "z":...}
twist = state["twist"]                    # linear / angular velocity

imu  = drone.get_imu_data("IMU1")             # accel + gyro
baro = drone.get_barometer_data("Barometer1") # pressure / altitude
mag  = drone.get_magnetometer_data("Magnetometer1")  # heading reference
batt = drone.get_battery_state("Battery")     # battery level
landed = drone.get_landed_state()             # 0 = LANDED, 1 = FLYING
```

### Easy vs Hard mode

Same course — the difference is **which APIs you may use**. It's on your honour; the level
shows `LEVEL: EASY` / `LEVEL: HARD` on the result popup.

- **🟢 EASY — anything goes**, including the "where am I, exactly?" oracles:
  `get_ground_truth_kinematics()`, `get_ground_truth_pose()`, `get_gps_data("GPS1")`,
  `get_ground_truth_geo_location()`, and GPS-waypoint moves
  (`move_to_geo_position_async`, `move_on_geo_path_async`).
- **🔴 HARD — fly like a real drone.** Those oracles are **off-limits**. Navigate with
  `get_estimated_kinematics()`, the onboard sensors (`IMU1`, `Barometer1`, `Magnetometer1`,
  `Battery`), and the **camera**. `set_pose` / `set_ground_truth_kinematics` (teleporting)
  are banned in **both** modes — the `reset()` is the only exception.

---

## 8. Vision

The level is a vision challenge — you read arrows, count spheres, and identify the target
vehicle from the camera. The nose camera `FPV` (640×480, 120° FOV) is your eye; a `Chase`
camera (1280×720) is also available.

```python
import cv2
frame = read_frame(drone)                # (H, W, 3) BGR uint8, or None
if frame is not None:
    # your detection: arrow color, sphere count, vehicle type...
    cv2.imwrite("frame.png", frame)

chase = read_frame(drone, camera="Chase")    # the follow cam
```

`read_frame` returns BGR (OpenCV's native order) so `cv2` functions work directly. If a
frame isn't ready yet it returns `None` — guard for it.

### Live preview window

Watch the feed in an OpenCV window while you develop — `view_camera.py` is a ready-to-run
version of this:

```python
import cv2
from redteam_sim import connect, read_frame

client, _world, drone = connect()
try:
    while True:
        frame = read_frame(drone)            # BGR, or None if not ready yet
        if frame is None:
            continue
        cv2.imshow("FPV", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):   # q or ESC to quit
            break
finally:
    cv2.destroyAllWindows()
    client.disconnect()
```

`cv2.imshow` / `cv2.waitKey` must run on the main thread, and `waitKey` is what actually
pumps the window's event loop — without it the window never paints. Run `python
view_camera.py` (`--camera Chase` for the follow cam).

### Advanced: raw image API

`read_frame` grabs the scene (RGB) image. For other capture types, call the raw API:

```python
from projectairsim.types import ImageType
from projectairsim.utils import unpack_image

images = drone.get_images("FPV", [ImageType.SCENE])
img = unpack_image(images[ImageType.SCENE])
```

| ImageType | Value |
|-----------|-------|
| `SCENE` | 0 (what `read_frame` uses) |
| `DEPTH_PLANAR` | 1 |
| `SEGMENTATION` | 3 |
| `SURFACE_NORMALS` | 6 |

---

## 9. Did I win?

The authoritative result is the **on-screen popup** — ✅ `MISSION PASSED` / ❌ `FAILED` —
shown on the FPV OSD when you reach a vehicle. The briefing also shows a `RaceManager` poll
(`get_object_float_property("RaceManager", "MissionState")`); that depends on the build
exposing those properties — if it returns `NaN`, rely on the on-screen result.

---

## 10. Resetting between attempts

```python
reset(drone)             # teleport back to the start line (velocity zeroed)
```

Returning within the start radius re-arms the race clock. This is the only sanctioned
`set_pose`.

---

## 11. Gotchas

- **Forgot the double-await?** `await drone.move_to_position_async(...)` only gets the task;
  your code runs ahead of the drone. Use `await (await cmd)` or the `do()` helper.
- **Won't take off?** Call `enable_api_control()` **and** `arm()` first, drone stationary.
- **Flying into the ground?** `+Z is down` — to climb, go to a *negative* `down`.
- **Connection refused?** The game must be running before you connect. Stale port:
  `pkill -f Red_Team_Hack_Sim/Binaries`.
- **Game crash/stall when connecting?** Close the active Python connection
  (`client.disconnect()`) before relaunching or starting another script — only one client
  has command authority at a time.
```
