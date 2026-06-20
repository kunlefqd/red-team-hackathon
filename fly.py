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

# ── Decision log (machine-readable timestamped action trace) ──────────────────
_dlog_path = f"logs/decisions_{int(time.time())}.log"
_dlog = logging.getLogger("decisions")
_dlog.setLevel(logging.INFO)
_dlog.addHandler(logging.FileHandler(_dlog_path))
_dlog.addHandler(logging.StreamHandler())
_dlog.propagate = False
_dlog_fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
for h in _dlog.handlers:
    h.setFormatter(_dlog_fmt)


def decision(stage: str, cue: str, action: str, **kwargs):
    """Log a single timestamped control decision with its visual trigger."""
    extras = "  ".join(f"{k}={v}" for k, v in kwargs.items())
    _dlog.info(f"STAGE={stage:<18s}  CUE={cue:<40s}  ACTION={action:<35s}  {extras}")

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


def _debug_sphere_view(frame, status=""):
    """Live debug window: blue/purple sphere contours, below-horizon only."""
    if frame is None:
        return

    display = frame.copy()
    h, w = frame.shape[:2]
    horizon = h // 2

    # Draw horizon line so it's obvious where the cutoff is
    cv2.line(display, (0, horizon), (w, horizon), (0, 255, 255), 1)

    roi = frame[horizon:, :]
    blurred = cv2.GaussianBlur(roi, (9, 9), 2)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 120, 80), (140, 255, 255))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

    rh, rw = roi.shape[:2]
    min_area = 150
    max_area = (rh * rw) // 5
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circ = 4 * math.pi * area / (perim ** 2)
        bx, by, bw, bh = cv2.boundingRect(c)
        # offset back into full-frame coordinates
        by += horizon
        color = (0, 200, 255) if circ >= 0.3 else (60, 60, 60)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.putText(display, f"c={circ:.2f}", (bx, max(horizon + 14, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        if circ >= 0.3:
            count += 1

    label = f"spheres={count}  {status}"
    cv2.putText(display, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Sphere Debug", display)
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
            decision("1_ARROW_SCAN", f"gap detected: cx_g={g_str} cx_r={r_str}",
                     f"stop spin at yaw={target_yaw:.1f}°",
                     green_px=g_px, red_px=r_px, step=step)

            actual_yaw = _yaw_deg(drone)
            snapped_yaw = round(actual_yaw / 45.0) * 45.0
            if abs(actual_yaw - snapped_yaw) > 1.0:
                log.info(f"   snapping {actual_yaw:.1f}° → {snapped_yaw:.0f}°")
                decision("1_ARROW_SCAN", f"yaw drift {actual_yaw:.1f}° vs nearest 45°",
                         f"snap to {snapped_yaw:.0f}°")
                await yaw_to(drone, snapped_yaw)
                await asyncio.sleep(0.3)
            log.info(f"   final yaw = {_yaw_deg(drone):.1f}°")
            return snapped_yaw

    f = read_frame(drone)
    if f is not None:
        _save_masks(f, "spin_failed_mask")
    decision("1_ARROW_SCAN", "full 360° — no gap found", "abort scan, return None")
    log.warning("   full 360° sweep — gap between arrows not found")
    return None


async def approach_arrow_sign(drone, bottom_frac=0.90):
    """
    Issue one long-duration forward velocity command so the drone moves
    smoothly without stop/start.  Poll frames every 0.1 s in the background.
    Override with hover_async() the moment the stop condition fires.
    """
    approach_yaw = _yaw_deg(drone)
    log.info(f">> approaching arrow sign … (heading {approach_yaw:.1f}°)")

    # Start moving — don't double-await, let it run in the background
    await drone.move_by_velocity_body_frame_async(3.0, 0.0, 0.0, 60.0)

    step = 0
    while step < 200:
        await asyncio.sleep(0.1)
        frame = read_frame(drone)
        if frame is not None:
            g_mask, r_mask = _green_red_masks(frame)
            g_px = cv2.countNonZero(g_mask)
            r_px = cv2.countNonZero(r_mask)
            active = g_mask if g_px >= r_px else r_mask
            active_px = max(g_px, r_px)

            norm_cy = 0.0
            if active_px > 50:
                M = cv2.moments(active)
                if M["m00"] > 0:
                    norm_cy = (M["m01"] / M["m00"]) / frame.shape[0]

            log.info(f"   step {step:02d}: cy={norm_cy*100:.1f}%  g_px={g_px}  r_px={r_px}")
            _debug_arrow_view(frame, f"approach {step} | cy={norm_cy*100:.1f}% / {bottom_frac*100:.0f}%")

            if active_px > 50 and norm_cy >= bottom_frac:
                log.info("   ✓ arrows near bottom — stopping")
                decision("1_ARROW_APPROACH",
                         f"norm_cy={norm_cy*100:.1f}% >= {bottom_frac*100:.0f}%",
                         "hover — sign is close enough",
                         active_px=active_px, g_px=g_px, r_px=r_px, step=step)
                await do(drone.hover_async())
                return frame

        step += 1

    decision("1_ARROW_APPROACH", "max iterations reached", "hover and continue")
    log.info("   approach max iterations reached")
    await do(drone.hover_async())
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
    Count blue/purple spheres in a single frame using HSV + contour circularity.
    Only searches the bottom half of the frame to avoid false positives from sky.
    """
    h, w = frame.shape[:2]
    horizon = h // 2
    roi = frame[horizon:, :]          # bottom half only

    blurred = cv2.GaussianBlur(roi, (9, 9), 2)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (100, 120, 80), (140, 255, 255))

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rh, rw = roi.shape[:2]
    min_area = 150
    max_area = (rh * rw) // 5

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


async def approach_blue_spheres(drone, bottom_frac=0.80):
    """
    Move forward smoothly until blue spheres are in the bottom 20% of frame.
    Uses the same continuous-velocity pattern as approach_arrow_sign.
    """
    log.info(">> moving forward to find blue spheres …")
    await drone.move_by_velocity_body_frame_async(3.0, 0.0, 0.0, 60.0)

    step = 0
    while step < 200:
        await asyncio.sleep(0.1)
        f = read_frame(drone)
        if f is not None:
            h = f.shape[0]
            horizon = h // 2
            roi = f[horizon:, :]

            blurred = cv2.GaussianBlur(roi, (9, 9), 2)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (100, 120, 80), (140, 255, 255))
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

            # centroid of all blue pixels in the ROI, mapped back to full frame
            norm_cy = 0.0
            if cv2.countNonZero(mask) > 50:
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    roi_cy = M["m01"] / M["m00"]           # y within ROI
                    full_cy = (horizon + roi_cy) / h       # y in full frame
                    norm_cy = full_cy

            count = _count_spheres(f)
            _debug_sphere_view(f, f"step={step:03d} cy={norm_cy*100:.1f}%")
            log.info(f"   step={step:03d}  spheres={count}  cy={norm_cy*100:.1f}%")

            if count > 0 and norm_cy >= bottom_frac:
                log.info("   ✓ spheres in bottom 20% — stopping")
                decision("2_SPHERE_APPROACH",
                         f"norm_cy={norm_cy*100:.1f}% >= {bottom_frac*100:.0f}%  spheres={count}",
                         "hover — spheres close enough",
                         step=step)
                await do(drone.hover_async())
                return

        step += 1

    decision("2_SPHERE_APPROACH", "max iterations reached", "hover and continue")
    log.info("   approach_spheres max iterations reached")
    await do(drone.hover_async())


async def count_spheres_reliable(drone, n=24):
    """
    Scan ±30° relative to current heading for the angle showing the most
    spheres, then vote across n frames at that angle.
    Returns the modal nonzero count (defaults to 1 = ODD on failure).
    """
    base_yaw = _yaw_deg(drone)
    best_count = 0
    best_yaw = base_yaw

    for offset in (0, 15, -15, 30, -30):
        target = base_yaw + offset
        await yaw_to(drone, target)
        await asyncio.sleep(0.4)
        counts = []
        for _ in range(4):
            f = read_frame(drone)
            if f is not None:
                counts.append(_count_spheres(f))
            await asyncio.sleep(0.08)
        avg = sum(counts) / len(counts) if counts else 0
        log.info(f"   sphere scan offset={offset:+d}°: counts={counts} avg={avg:.1f}")
        if avg > best_count:
            best_count = avg
            best_yaw = target

    await yaw_to(drone, best_yaw)
    await asyncio.sleep(0.4)

    all_counts = []
    for _ in range(n):
        f = read_frame(drone)
        if f is None:
            await asyncio.sleep(0.05)
            continue
        _save(f, "spheres")
        c = _count_spheres(f)
        _debug_sphere_view(f, f"voting n={len(all_counts)}")
        all_counts.append(c)
        await asyncio.sleep(0.05)

    log.info(f"   sphere raw counts: {all_counts}")

    nonzero = [c for c in all_counts if c > 0]
    if not nonzero:
        decision("2_SPHERE_COUNT", "all frames returned 0 spheres", "default to 1 (ODD→RIGHT)",
                 raw_counts=str(all_counts))
        log.warning("!! sphere detection failed — defaulting to 1 (ODD)")
        return 1

    mode = max(set(nonzero), key=nonzero.count)
    turn = "ODD→RIGHT" if mode % 2 else "EVEN→LEFT"
    decision("2_SPHERE_COUNT",
             f"modal count={mode}  raw={all_counts}",
             f"turn {turn}",
             nonzero_frames=len(nonzero), total_frames=len(all_counts))
    log.info(f"   sphere count = {mode} ({turn})")
    return mode


# ── Stage 3: fly to ambulance and land ────────────────────────────────────────

def _red_vehicle_mask(frame):
    """Return a mask of the red ambulance body (broad red HSV range)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, (0,  80, 80), (10, 255, 255)) |
        cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    )
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)
    return mask


def _debug_vehicle_view(frame, status=""):
    """Live debug window showing red vehicle detection."""
    if frame is None:
        return
    display = frame.copy()
    h, w = frame.shape[:2]
    mask = _red_vehicle_mask(frame)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 200:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
    # horizon line for reference
    cv2.line(display, (0, h // 2), (w, h // 2), (128, 128, 128), 1)
    if status:
        cv2.putText(display, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Vehicle Debug", display)
    cv2.waitKey(1)


async def fly_to_ambulance_and_land(drone, bottom_frac=0.85):
    """
    Fly forward continuously.  Poll frames every 0.1 s.
    Stop and land when the red ambulance centroid reaches bottom_frac of frame.
    """
    log.info(">> stage 3: flying toward ambulance …")
    await drone.move_by_velocity_body_frame_async(5.0, 0.0, 0.0, 60.0)

    step = 0
    while step < 300:
        await asyncio.sleep(0.1)
        f = read_frame(drone)
        if f is None:
            step += 1
            continue

        mask = _red_vehicle_mask(f)
        red_px = cv2.countNonZero(mask)
        norm_cy = 0.0
        if red_px > 200:
            M = cv2.moments(mask)
            if M["m00"] > 0:
                norm_cy = (M["m01"] / M["m00"]) / f.shape[0]

        _debug_vehicle_view(f, f"step={step:03d}  red_px={red_px}  cy={norm_cy*100:.1f}%")
        log.info(f"   vehicle step={step:03d}  red_px={red_px}  cy={norm_cy*100:.1f}%")

        if red_px > 200 and norm_cy >= bottom_frac:
            decision("3_VEHICLE_APPROACH",
                     f"red_px={red_px}  norm_cy={norm_cy*100:.1f}% >= {bottom_frac*100:.0f}%",
                     "hover and land beside ambulance", step=step)
            log.info("   ✓ ambulance at bottom of frame — landing")
            await do(drone.hover_async())
            await do(drone.land_async())
            return

        step += 1

    decision("3_VEHICLE_APPROACH", "max iterations — ambulance not detected", "land anyway")
    log.info("   max iterations reached — landing")
    await do(drone.hover_async())
    await do(drone.land_async())


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
        cv2.namedWindow("Sphere Debug", cv2.WINDOW_NORMAL)
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
        decision("1_ARROW_READ", f"green arrow direction={turn1}",
                 f"yaw {'-90°' if turn1 == 'LEFT' else '+90°'} toward green arrow")

        # Turn 90 degrees toward the side with the green arrow.
        delta = -90.0 if turn1 == "LEFT" else 90.0
        log.info(f">> turning {delta:+.0f}° toward the green arrow")
        await yaw_by(drone, delta)
        await do(drone.hover_async())

        # ── Stage 2: fly forward until blue spheres visible, then count ──
        log.info(">> stage 2: approaching blue spheres")
        await approach_blue_spheres(drone)

        log.info(">> stage 2: counting blue spheres")
        sphere_count = await count_spheres_reliable(drone)
        log.info(f">> sphere count = {sphere_count}")

        # Even → LEFT, Odd → RIGHT
        turn2 = "LEFT" if sphere_count % 2 == 0 else "RIGHT"
        delta2 = -90.0 if turn2 == "LEFT" else 90.0
        decision("2_SPHERE_TURN", f"sphere_count={sphere_count} ({'even' if sphere_count%2==0 else 'odd'})",
                 f"yaw {delta2:+.0f}° → {turn2}", sphere_count=sphere_count)
        log.info(f">> stage 2 complete: spheres={sphere_count} → turning {turn2} ({delta2:+.0f}°)")
        await yaw_by(drone, delta2)
        # Snap to nearest 90° for accuracy
        actual_yaw = _yaw_deg(drone)
        snapped_yaw = round(actual_yaw / 90.0) * 90.0
        if abs(actual_yaw - snapped_yaw) > 1.0:
            log.info(f"   snapping {actual_yaw:.1f}° → {snapped_yaw:.0f}°")
            decision("2_SPHERE_TURN", f"yaw drift {actual_yaw:.1f}° vs nearest 90°",
                     f"snap to {snapped_yaw:.0f}°")
            await yaw_to(drone, snapped_yaw)
        await do(drone.hover_async())

        # ── Stage 3: fly to ambulance and land ───────────────────────────
        log.info(f">> stage 3: flying to ambulance  (path={turn1}+{turn2})")
        cv2.namedWindow("Vehicle Debug", cv2.WINDOW_NORMAL)
        await fly_to_ambulance_and_land(drone)

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