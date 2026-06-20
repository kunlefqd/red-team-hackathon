#!/usr/bin/env python3
"""
Autonomous drone for the Red Team Hack Sim.

First-stage flow:
    Takeoff → hover scan → find the green/red arrow sign → move near it
    → turn 90° toward the green arrow.

Coordinate system: NED metres  (+X north, +Y east, +Z DOWN, climb = negative Z)
Drone spawns at (0, -35, -0.1).
"""
import argparse
import asyncio
import io
import json
import logging
import math
import os
import time

import cv2
import numpy as np

from redteam_sim import connect, reset, read_frame  # noqa: F401
from vehicle import identify_vehicle_opencv

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/run.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Flight constants ──────────────────────────────────────────────────────────
ALT = 5.0        # cruise altitude (m above ground → z = -5)
SPEED = 5.0      # default move speed (m/s)

GEMINI_API_KEY = ""  # set via: export GEMINI_API_KEY=your_key

# ── Async helpers ─────────────────────────────────────────────────────────────

async def do(cmd):
    """Await a *_async() command to completion (double-await pattern)."""
    await (await cmd)


def _pos(drone):
    state = drone.get_estimated_kinematics()
    p = state.get("pose", {}).get("position", {})
    return p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)


def _yaw_deg(drone):
    state = drone.get_estimated_kinematics()
    ori = state.get("pose", {}).get("orientation", {})
    w = float(ori.get("w", 1.0))
    x = float(ori.get("x", 0.0))
    y = float(ori.get("y", 0.0))
    z = float(ori.get("z", 0.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


async def goto(drone, n, e, d, speed=None, label=""):
    spd = speed or SPEED
    if label:
        log.info(f"   fly -> ({n}, {e}, {d})  [{label}]")
    await do(drone.move_to_position_async(n, e, d, spd))
    x, y, z = _pos(drone)
    if label:
        log.info(f"   arrived ~ ({x:.1f}, {y:.1f}, {z:.1f})")


async def yaw_to(drone, deg):
    await do(drone.rotate_to_yaw_async(math.radians(deg)))
    log.info(f"   yaw -> {deg}°")


async def yaw_by(drone, delta_deg):
    target = _yaw_deg(drone) + delta_deg
    await do(drone.rotate_to_yaw_async(math.radians(target)))
    log.info(f"   yaw +{delta_deg}° -> target {target:.1f}°")


def _save(frame, prefix):
    ts = int(time.time() * 1000)
    path = f"logs/{prefix}_{ts}.jpg"
    cv2.imwrite(path, frame)
    log.info(f"   saved frame -> {path}")


def _green_red_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (20, 35, 25), (95, 255, 255))
    red = (
        cv2.inRange(hsv, (0, 35, 25), (15, 255, 255)) |
        cv2.inRange(hsv, (150, 35, 25), (180, 255, 255))
    )
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kern)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kern)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kern)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kern)
    return green, red


def _debug_arrow_view(frame, status=""):
    """Show the live camera feed with green/red contour boxes for debugging."""
    if frame is None:
        return

    display = frame.copy()
    green, red = _green_red_masks(frame)

    for mask, color, label in ((green, (0, 255, 0), "GREEN"), (red, (0, 0, 255), "RED")):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 120:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, f"{label} {int(area)}", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if status:
        cv2.putText(display, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Arrow Debug", display)
    cv2.waitKey(1)


def _arrow_sign_area(frame):
    """Return a simple area score for the green/red sign; larger means closer."""
    if frame is None:
        return 0.0
    g, r = _green_red_masks(frame)
    return float(cv2.countNonZero(g) + cv2.countNonZero(r))


# ── Vision: arrow direction ───────────────────────────────────────────────────

def _arrow_from_frame(frame):
    """
    Return "LEFT", "RIGHT", or None.

    Strategy:
      1. Build an HSV mask for the green arrow.
      2. Find the largest green blob.
      3. Scan each column of the bounding box; the column-density profile
         tells us which end is the pointed tip.
         - Density rises right-to-left  →  tip is on LEFT  →  arrow points LEFT
         - Density rises left-to-right  →  tip is on RIGHT →  arrow points RIGHT
      4. Fallback: if the green blob centroid is left of the red blob centroid
         the green arrow is on the LEFT side of the sign  →  go LEFT.
    """
    h, w = frame.shape[:2]

    # ── masks ──────────────────────────────────────────────────────────────
    g_mask, r_mask = _green_red_masks(frame)

    # ── find green contour ─────────────────────────────────────────────────
    cnts, _ = cv2.findContours(g_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    gc = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(gc) < 180:
        return None

    bx, by, bw, bh = cv2.boundingRect(gc)

    # ── column-density profile ─────────────────────────────────────────────
    cols = [
        cv2.countNonZero(g_mask[by:by+bh, bx+c:bx+c+1])
        for c in range(bw)
    ]
    if max(cols) == 0:
        return None

    # Smooth the profile
    smooth = np.convolve(cols, np.ones(max(1, bw//10)) / max(1, bw//10), mode="same")

    margin = max(2, bw // 6)
    left_density  = float(np.mean(smooth[:margin]))
    right_density = float(np.mean(smooth[-margin:]))

    if abs(left_density - right_density) > 0.5:
        # The tip has FEWER pixels; the broad shaft/body has MORE
        # tip on left (fewer left pixels) → arrow points LEFT
        # tip on right (fewer right pixels) → arrow points RIGHT
        if left_density < right_density:
            return "LEFT"
        else:
            return "RIGHT"

    # ── fallback: green blob position vs red blob position ────────────────
    r_cnts, _ = cv2.findContours(r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not r_cnts:
        return None
    rc = max(r_cnts, key=cv2.contourArea)

    Mg = cv2.moments(gc)
    Mr = cv2.moments(rc)
    if Mg["m00"] == 0 or Mr["m00"] == 0:
        return None

    gcx = Mg["m10"] / Mg["m00"]
    rcx = Mr["m10"] / Mr["m00"]

    # Green arrow is to the LEFT of red arrow → green points LEFT (or IS left)
    if gcx < rcx - w * 0.02:
        return "LEFT"
    if gcx > rcx + w * 0.02:
        return "RIGHT"

    # ── final fallback: if red is weak, use the green contour skew alone ──
    pts = gc[:, 0, :]
    left_edge = float(np.mean(pts[pts[:, 0] <= gcx][:, 0])) if np.any(pts[:, 0] <= gcx) else gcx
    right_edge = float(np.mean(pts[pts[:, 0] >= gcx][:, 0])) if np.any(pts[:, 0] >= gcx) else gcx
    if (gcx - left_edge) > (right_edge - gcx):
        return "LEFT"
    if (right_edge - gcx) > (gcx - left_edge):
        return "RIGHT"
    return None


def has_arrow_blobs(frame, min_area=400):
    """True if frame has both a green and a red blob of meaningful size."""
    g, r = _green_red_masks(frame)
    gc, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rc, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    g_ok = any(cv2.contourArea(c) >= min_area * 0.5 for c in gc)
    r_ok = any(cv2.contourArea(c) >= min_area * 0.5 for c in rc)
    return g_ok and r_ok


async def spin_find_arrows(drone):
    """
    Sweep 360° in slow steps while hovering, return the current yaw when the
    green/red arrow sign becomes visible.
    """
    log.info(">> spin-scan: searching for arrow room …")
    for step in range(60):
        hits = 0
        areas = []
        for _ in range(3):
            f = read_frame(drone)
            if f is not None:
                areas.append(_arrow_sign_area(f))
                if has_arrow_blobs(f):
                    hits += 1
                _debug_arrow_view(f, f"scan step {step:02d} hits {hits}/3")
            time.sleep(0.1)

        log.info(f"   step={step:02d}  arrow_hits={hits}/3  area={max(areas) if areas else 0:.0f}")
        if hits >= 2:
            log.info("   ✓ arrow sign confirmed")
            f = read_frame(drone)
            if f is not None:
                _save(f, "scan_found")
            return _yaw_deg(drone)

        await yaw_by(drone, 6.0)
        await do(drone.hover_async())
        await asyncio.sleep(0.4)

    return None


async def approach_arrow_sign(drone, max_steps=12):
    """Move closer to the sign in small body-frame increments."""
    log.info(">> approaching the arrow sign …")
    best_area = 0.0
    for step in range(max_steps):
        frame = read_frame(drone)
        if frame is not None:
            best_area = max(best_area, _arrow_sign_area(frame))
            _save(frame, f"approach_{step}")
            _debug_arrow_view(frame, f"approach step {step} area {int(best_area)}")
            if best_area >= 600.0:
                log.info(f"   sign is close enough (area={best_area:.0f})")
                return

        await do(drone.move_by_velocity_body_frame_async(1.0, 0.0, 0.0, 1.0))
        await do(drone.hover_async())
        await asyncio.sleep(0.5)

    log.info(f"   stop approach at area={best_area:.0f}")


def detect_arrow_reliable(drone, n=20):
    """
    Sample n frames, vote on direction.
    Returns "LEFT" or "RIGHT" (defaults "RIGHT" on failure).
    """
    votes = {"LEFT": 0, "RIGHT": 0}
    for _ in range(n):
        f = read_frame(drone)
        if f is None:
            time.sleep(0.05)
            continue
        _save(f, "arrow")
        _debug_arrow_view(f, "reading arrow direction")
        d = _arrow_from_frame(f)
        if d:
            votes[d] += 1
        time.sleep(0.05)

    log.info(f"   Arrow votes: LEFT={votes['LEFT']}, RIGHT={votes['RIGHT']}")
    if votes["LEFT"] == 0 and votes["RIGHT"] == 0:
        log.warning("!! Arrow detection failed — defaulting to RIGHT")
        return "RIGHT"
    return "LEFT" if votes["LEFT"] > votes["RIGHT"] else "RIGHT"


# ── Vision: sphere count ──────────────────────────────────────────────────────

def _count_spheres(frame):
    """
    Count blue spheres in a single frame using HSV + contour circularity.
    """
    blurred = cv2.GaussianBlur(frame, (9, 9), 2)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Require high saturation to exclude sphere shadows and ambient blue tones
    mask = cv2.inRange(hsv, (85, 100, 60), (135, 255, 255))

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    min_area = 500   # spheres must be reasonably large — filters shadow fragments
    max_area = (h * w) // 5

    spheres = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circ = 4 * math.pi * area / (perim ** 2)
        if circ >= 0.55:   # must be round — shadows and edges won't pass
            spheres += 1
    return min(spheres, 5)  # game rule: max 5 spheres


async def count_spheres_reliable(drone, n=24):
    """
    Rotate ±30° looking for the best sphere count, then vote across frames.
    Returns the modal nonzero count (defaults to 1 = ODD on failure).
    """
    best_count = 0
    best_yaw = 0

    # Scan yaw range to find the angle with the most spheres
    for yaw in (0, 45, 90, 135, 180, -135, -90, -45, -30, 30, -15, 15):
        await yaw_to(drone, yaw)
        await asyncio.sleep(0.5)
        counts = []
        for _ in range(4):
            f = read_frame(drone)
            if f is not None:
                counts.append(_count_spheres(f))
            time.sleep(0.08)
        avg = sum(counts) / len(counts) if counts else 0
        log.info(f"   sphere scan yaw={yaw}°: counts={counts} avg={avg:.1f}")
        if avg > best_count:
            best_count = avg
            best_yaw = yaw

    # Lock onto the best yaw and vote across more frames
    await yaw_to(drone, best_yaw)
    await asyncio.sleep(0.5)

    all_counts = []
    for _ in range(n):
        f = read_frame(drone)
        if f is None:
            time.sleep(0.05)
            continue
        _save(f, "spheres")
        all_counts.append(_count_spheres(f))
        time.sleep(0.05)

    log.info(f"   Sphere raw counts: {all_counts}")

    nonzero = [c for c in all_counts if c > 0]
    if not nonzero:
        log.warning("!! Sphere detection failed — defaulting to 1 (ODD)")
        return 1

    # Modal value among nonzero counts
    mode = max(set(nonzero), key=nonzero.count)
    return mode


# ── Stage 3: vehicle detection + approach ─────────────────────────────────────

VEHICLE_LOOKUP = {
    ("LEFT",  "LEFT"):  "tank",
    ("LEFT",  "RIGHT"): "boat",
    ("RIGHT", "LEFT"):  "jet",
    ("RIGHT", "RIGHT"): "ice_cream_truck",
}


async def find_and_approach_vehicle(drone, target):
    """
    Creep forward toward vehicles, scan 360° with OpenCV to find the target.
    Falls back to flying straight if not found.
    """
    log.info(f">> CV scanning for target vehicle: {target}")

    for approach in range(6):
        log.info(f"   approach pass {approach+1}/6 — flying forward 3s")
        await do(drone.move_by_velocity_body_frame_async(SPEED, 0.0, 0.0, 3.0))
        await do(drone.hover_async())

        found_yaw = None
        consecutive = 0
        for step in range(8):
            frame = read_frame(drone)
            if frame is not None:
                _save(frame, f"vehicle_scan_{approach}_{step}")
                vehicle_type, info = identify_vehicle_opencv(frame)
                log.info(f"   approach={approach} step={step} — CV: {vehicle_type} info={info}")
                if vehicle_type == target:
                    consecutive += 1
                    if consecutive >= 2:
                        found_yaw = _yaw_deg(drone)
                        log.info(f"   ✓ confirmed {target} at yaw={found_yaw:.1f}°")
                        break
                    # rotate a little more to get 2nd confirmation frame
                    await yaw_by(drone, 15.0)
                    await asyncio.sleep(0.3)
                    continue
                else:
                    consecutive = 0
            await yaw_by(drone, 45.0)
            await asyncio.sleep(0.5)

        if found_yaw is not None:
            break

    if found_yaw is None:
        log.warning(f"   {target} not found after all passes — flying straight")

    if target == "ice_cream_truck":
        log.info(">> ice cream truck: creeping closer until it fills the frame, then landing")
        for step in range(8):
            await do(drone.move_by_velocity_body_frame_async(SPEED, 0.0, 0.0, 2.0))
            await do(drone.hover_async())
            frame = read_frame(drone)
            if frame is not None:
                _save(frame, f"ice_cream_approach_{step}")
                vt, info = identify_vehicle_opencv(frame)
                log.info(f"   creep step {step}: {vt} area={info['area'] if info else None}")
                if vt == "ice_cream_truck" and info and info["area"] > 80000:
                    log.info("   >> close enough — landing")
                    break
        await do(drone.land_async())
        log.info(">> landed beside ice cream truck ✓")
    else:
        log.info(f">> {target}: flying into it at speed")
        await do(drone.move_by_velocity_body_frame_async(SPEED, 0.0, 0.0, 5.0))


# ── Main flight ───────────────────────────────────────────────────────────────

async def fly(address: str):
    log.info(f"Connected | alt={ALT}m speed={SPEED}m/s")

    client, world, drone = connect(address)
    try:
        # ── Arm & take off ────────────────────────────────────────────────
        log.info(">> arming")
        drone.enable_api_control()
        drone.arm()

        log.info(f">> takeoff to {ALT} m")
        await do(drone.takeoff_async())
        await do(drone.move_to_position_async(0.0, -35.0, -ALT, SPEED))
        await do(drone.hover_async())
        
        
        log.info(">> hovering and scanning for the green/red arrow sign")
        cv2.namedWindow("Arrow Debug", cv2.WINDOW_NORMAL)
        scan_yaw = await spin_find_arrows(drone)
        if scan_yaw is None:
            log.warning(">> no arrow sign found after a full slow scan")
            await do(drone.land_async())
            drone.disarm()
            return

        log.info(">> flying near the arrow sign")
        await approach_arrow_sign(drone)

        log.info(">> reading arrow direction")
        turn1 = detect_arrow_reliable(drone)
        log.info(f">> green arrow points {turn1}")

        # Turn 90 degrees toward the side with the green arrow.
        delta = -90.0 if turn1 == "LEFT" else 90.0
        log.info(f">> turning {delta:+.0f}° toward the green arrow")
        await yaw_by(drone, delta)
        await do(drone.hover_async())

        log.info(">> stage 1 complete — flying to sphere room")
        await do(drone.move_to_position_async(93.0, -13.0, -ALT, SPEED))
        await do(drone.hover_async())

        # ── Stage 2: count blue spheres ───────────────────────────────────
        log.info(">> stage 2: counting blue spheres")
        sphere_count = await count_spheres_reliable(drone)
        parity = "EVEN" if sphere_count % 2 == 0 else "ODD"
        turn2 = "LEFT" if sphere_count % 2 == 0 else "RIGHT"
        log.info(f">> sphere count={sphere_count} ({parity}) → turn {turn2}")

        delta2 = -90.0 if turn2 == "LEFT" else 90.0
        log.info(f">> turning {delta2:+.0f}° for sphere result")
        await yaw_by(drone, delta2)
        await do(drone.hover_async())

        log.info(f">> stage 2 complete — turn1={turn1} turn2={turn2}")
        log.info(">> flying toward vehicle area")
        await do(drone.move_by_velocity_body_frame_async(SPEED, 0.0, 0.0, 4.0))
        await do(drone.hover_async())

        # ── Stage 3: vehicle selection ────────────────────────────────────
        target = VEHICLE_LOOKUP[(turn1, turn2)]
        log.info(f">> target vehicle: {target}")

        # ── Stage 3: find and approach target vehicle ─────────────────────
        await find_and_approach_vehicle(drone, target)

        # ── Poll race manager for result ──────────────────────────────────
        log.info(">> waiting for mission result...")
        for _ in range(150):  # up to 30s
            state = world.get_object_float_property("RaceManager", "MissionState")
            if state in (2.0, 3.0):
                elapsed = world.get_object_float_property("RaceManager", "ElapsedSeconds")
                result = "PASSED ✅" if state == 2.0 else "FAILED ❌"
                log.info(f">> {result} — time={elapsed:.1f}s")
                break
            time.sleep(0.2)
        else:
            log.warning(">> no result from RaceManager after 30s")

        drone.disarm()

    finally:
        cv2.destroyAllWindows()
        client.disconnect()
        log.info(">> disconnected")


async def test_spheres(address: str):
    """Teleport directly to the sphere room and test detection only."""
    from redteam_sim import reset
    from projectairsim.types import Pose

    client, world, drone = connect(address)
    try:
        drone.enable_api_control()
        drone.arm()
        await do(drone.takeoff_async())
        await do(drone.move_to_position_async(0.0, -35.0, -ALT, SPEED))

        # Teleport to estimated sphere room position — tune these if needed
        SPHERE_POSE = Pose({"translation": {"x": 93.0, "y": -13.0, "z": -ALT},
                            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}})
        log.info(">> teleporting to sphere room")
        drone.set_pose(SPHERE_POSE, reset_kinematics=True)
        await asyncio.sleep(1.0)
        await do(drone.hover_async())

        sphere_count = await count_spheres_reliable(drone)
        parity = "EVEN" if sphere_count % 2 == 0 else "ODD"
        turn = "LEFT" if sphere_count % 2 == 0 else "RIGHT"
        log.info(f">> result: count={sphere_count} ({parity}) → {turn}")
    finally:
        client.disconnect()


async def test_ice_cream(address: str):
    """Spin until ice cream truck is visible, fly straight at it, land."""
    client, world, drone = connect(address)
    try:
        drone.enable_api_control()
        drone.arm()
        await do(drone.takeoff_async())
        await do(drone.move_to_position_async(0.0, -35.0, -ALT, SPEED))
        await do(drone.hover_async())

        # ── Turn 45° right and fly straight to the truck ─────────────────────
        log.info(">> turning 45° right toward ice cream truck")
        await yaw_by(drone, 45.0)
        await do(drone.hover_async())

        log.info(">> flying to ice cream truck")
        await do(drone.move_by_velocity_body_frame_async(SPEED, 0.0, 0.0, 36.0))
        await do(drone.hover_async())

        x, y, _ = _pos(drone)
        await do(drone.move_to_position_async(x, y, -ALT, SPEED))
        await do(drone.hover_async())
        await do(drone.land_async())
        log.info(">> landed ✓")
        drone.disarm()
    finally:
        cv2.destroyAllWindows()
        client.disconnect()


def main():
    ap = argparse.ArgumentParser(description="Red Team Hack Sim — autonomous flight")
    ap.add_argument("--address", default="127.0.0.1")
    ap.add_argument("--test-spheres", action="store_true",
                    help="teleport to sphere room and test detection only")
    ap.add_argument("--test-ice-cream", action="store_true",
                    help="skip stages 1+2, go straight to ice cream truck scan+land")
    args = ap.parse_args()
    if args.test_spheres:
        asyncio.run(test_spheres(args.address))
    elif args.test_ice_cream:
        asyncio.run(test_ice_cream(args.address))
    else:
        asyncio.run(fly(args.address))


if __name__ == "__main__":
    main()
