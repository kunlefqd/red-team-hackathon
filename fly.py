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
import logging
import math
import os
import time

import cv2
import numpy as np

from redteam_sim import connect, reset, read_frame  # noqa: F401

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
ALT = 5.0        # cruise altitude (m above ground → z = -3)
SPEED = 5.0      # default move speed (m/s)

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
    # Permissive ranges to handle Unreal's gamma/tone-mapping (gamma=2.5)
    green = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    red = (
        cv2.inRange(hsv, (0,  25, 25), (15, 255, 255)) |
        cv2.inRange(hsv, (155, 25, 25), (180, 255, 255))
    )
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN,  kern)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kern)
    red   = cv2.morphologyEx(red,   cv2.MORPH_OPEN,  kern)
    red   = cv2.morphologyEx(red,   cv2.MORPH_CLOSE, kern)
    return green, red


def _save_masks(frame, tag="mask"):
    """Save raw HSV masks side-by-side for offline tuning."""
    g, r = _green_red_masks(frame)
    h, w = frame.shape[:2]
    panel = np.zeros((h, w * 3, 3), dtype=np.uint8)
    panel[:, :w]        = frame
    panel[:, w:2*w]     = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    panel[:, 2*w:3*w]   = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    ts = int(time.time() * 1000)
    cv2.imwrite(f"logs/{tag}_{ts}.jpg", panel)
    log.info(f"   mask saved -> logs/{tag}_{ts}.jpg  (left=frame | mid=green | right=red)")


def _debug_arrow_view(frame, status=""):
    """Live debug window: bounding boxes, centroid line, and status text."""
    if frame is None:
        return

    display = frame.copy()
    h, w = display.shape[:2]
    g_mask, r_mask = _green_red_masks(frame)

    for mask, color, lbl in ((g_mask, (0, 200, 0), "G"), (r_mask, (0, 0, 200), "R")):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 120:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(display, f"{lbl} {int(cv2.contourArea(c))}", (bx, max(14, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Draw the combined centroid as a vertical line
    norm_cx = _arrows_centroid_norm(frame)
    if norm_cx is not None:
        cx_px = int(norm_cx * w)
        cv2.line(display, (cx_px, 0), (cx_px, h), (0, 255, 255), 2)
        cv2.line(display, (w // 2, 0), (w // 2, h), (128, 128, 128), 1)  # centre reference

    if status:
        cv2.putText(display, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

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


def _arrows_centroid_norm(frame):
    """
    Normalised horizontal centroid of the combined green+red mask.
    Merging into one mask before computing moments gives a single centroid
    that sits at the physical centre of the sign regardless of colour order.
    Returns None when there aren't enough coloured pixels.
    """
    g_mask, r_mask = _green_red_masks(frame)
    combined = cv2.bitwise_or(g_mask, r_mask)
    if cv2.countNonZero(combined) < 100:
        return None
    M = cv2.moments(combined)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"]) / frame.shape[1]


async def spin_find_arrows(drone, min_each=15):
    """
    Sweep in 10° steps.  For each heading, compute the individual centroids
    of the green and red blobs.  Stop when the camera centre (0.5) falls
    strictly between them — i.e. one blob is left of centre and the other
    is right of centre.  This fires exactly when the drone is looking into
    the gap between the two arrows, regardless of which colour is on which side.
    Then snap yaw to the nearest 45° via live telemetry.
    Returns the snapped yaw, or None after a full revolution.
    """
    log.info(">> spin-scan: searching for gap between arrows …")
    start_yaw = _yaw_deg(drone)
    mask_saved = False

    for step in range(36):
        target_yaw = start_yaw + step * 10.0
        await yaw_to(drone, target_yaw)
        await asyncio.sleep(0.2)

        f = read_frame(drone)
        if f is None:
            continue

        g_mask, r_mask = _green_red_masks(f)
        g_px = cv2.countNonZero(g_mask)
        r_px = cv2.countNonZero(r_mask)

        cx_g, cx_r = None, None
        if g_px >= min_each:
            Mg = cv2.moments(g_mask)
            if Mg["m00"] > 0:
                cx_g = (Mg["m10"] / Mg["m00"]) / f.shape[1]
        if r_px >= min_each:
            Mr = cv2.moments(r_mask)
            if Mr["m00"] > 0:
                cx_r = (Mr["m10"] / Mr["m00"]) / f.shape[1]

        # camera centre is in the gap when one blob is left of 0.5
        # and the other is right of 0.5
        in_gap = (
            cx_g is not None and cx_r is not None
            and min(cx_g, cx_r) < 0.5 < max(cx_g, cx_r)
        )

        g_str = f"{cx_g:.2f}" if cx_g is not None else "----"
        r_str = f"{cx_r:.2f}" if cx_r is not None else "----"
        log.info(
            f"   step={step:02d}  yaw={target_yaw:.1f}°  "
            f"green_px={g_px} cx_g={g_str}  "
            f"red_px={r_px} cx_r={r_str}  in_gap={in_gap}"
        )
        _debug_arrow_view(f, f"scan {target_yaw:.0f}° | g={g_str} r={r_str} | {'IN GAP ✓' if in_gap else 'scanning…'}")

        if not mask_saved and g_px + r_px > 30:
            _save_masks(f, "spin_mask")
            mask_saved = True

        if in_gap:
            log.info("   ✓ camera centred in gap between arrows")
            _save(f, "scan_found")

            actual_yaw = _yaw_deg(drone)
            snapped_yaw = round(actual_yaw / 45.0) * 45.0
            if abs(actual_yaw - snapped_yaw) > 1.0:
                log.info(f"   snapping {actual_yaw:.1f}° → {snapped_yaw:.0f}°")
                await yaw_to(drone, snapped_yaw)
                await asyncio.sleep(0.3)
            log.info(f"   final yaw = {_yaw_deg(drone):.1f}°")
            return snapped_yaw

    f = read_frame(drone)
    if f is not None:
        _save_masks(f, "spin_failed_mask")
    log.warning("   full 360° sweep — gap between arrows not found")
    return None


async def approach_arrow_sign(drone, max_steps=40, bottom_frac=0.90):
    """
    Fly forward (body-frame) 1 m at a time.
    Stop when the green arrow blob centroid is in the bottom bottom_frac
    of the frame — meaning the sign is directly below/ahead of us.
    """
    approach_yaw = _yaw_deg(drone)
    log.info(f">> approaching arrow sign … (heading {approach_yaw:.1f}°)")

    for step in range(max_steps):
        frame = read_frame(drone)
        if frame is not None:
            g_mask, r_mask = _green_red_masks(frame)
            h = frame.shape[0]

            g_px = cv2.countNonZero(g_mask)
            r_px = cv2.countNonZero(r_mask)
            active = g_mask if g_px >= r_px else r_mask
            active_px = max(g_px, r_px)

            norm_cy = 0.0
            if active_px > 50:
                M = cv2.moments(active)
                if M["m00"] > 0:
                    norm_cy = (M["m01"] / M["m00"]) / h

            _save(frame, f"approach_{step:02d}")
            _debug_arrow_view(frame, f"approach {step} | cy={norm_cy*100:.1f}% / {bottom_frac*100:.0f}%  g={g_px} r={r_px}")
            log.info(f"   step {step:02d}: cy={norm_cy*100:.1f}%  g_px={g_px}  r_px={r_px}")

            if active_px > 50 and norm_cy >= bottom_frac:
                log.info("   ✓ arrows near bottom of screen — stopping approach")
                return frame

        await do(drone.move_by_velocity_body_frame_async(1.0, 0.0, 0.0, 1.0))
        await do(drone.hover_async())
        await asyncio.sleep(0.4)

    log.info("   approach max steps reached")
    return read_frame(drone)


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

    # Wide blue range (accounts for Unreal gamma/tone-mapping)
    mask = cv2.inRange(hsv, (85, 60, 60), (135, 255, 255))

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    min_area = 150
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
        if circ >= 0.3:   # reasonably round
            spheres += 1
    return spheres


async def count_spheres_reliable(drone, n=24):
    """
    Rotate ±30° looking for the best sphere count, then vote across frames.
    Returns the modal nonzero count (defaults to 1 = ODD on failure).
    """
    best_count = 0
    best_yaw = 0

    # Scan yaw range to find the angle with the most spheres
    for yaw in (0, 15, -15, 30, -30):
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
        close_frame = await approach_arrow_sign(drone)
        if close_frame is None:
            log.warning(">> did not reach the close-view threshold, continuing to direction readout")

        log.info(">> reading arrow direction")
        turn1 = _arrow_from_frame(close_frame) if close_frame is not None else None
        if turn1 is None:
            turn1 = detect_arrow_reliable(drone)
        log.info(f">> green arrow points {turn1}")

        # Turn 90 degrees toward the side with the green arrow.
        delta = -90.0 if turn1 == "LEFT" else 90.0
        log.info(f">> turning {delta:+.0f}° toward the green arrow")
        await yaw_by(drone, delta)
        await do(drone.hover_async())

        log.info(">> first stage complete")
        await do(drone.land_async())
        drone.disarm()

    finally:
        cv2.destroyAllWindows()
        client.disconnect()
        log.info(">> disconnected")


def main():
    ap = argparse.ArgumentParser(description="Red Team Hack Sim — autonomous flight")
    ap.add_argument("--address", default="127.0.0.1")
    args = ap.parse_args()
    asyncio.run(fly(args.address))


if __name__ == "__main__":
    main()