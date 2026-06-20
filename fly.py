#!/usr/bin/env python3
import argparse
import asyncio
import cv2
import numpy as np

from redteam_sim import connect, reset, read_frame  # noqa: F401


async def do(cmd):
    """Send a *_async() flight command and wait until it finishes."""
    await (await cmd)


# HSV range for green.
# If detection is weak, tune these while watching the Green Mask window.
GREEN_LOWER = np.array([35, 60, 60], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)


def estimate_arrow_direction(contour, x, y, w, h, cx, cy):
    """
    Estimate if the green arrow points LEFT or RIGHT.

    Main idea:
    - The arrow tip is usually the pointier side.
    - The pointier side has less vertical spread near the extreme edge.
    """
    pts = contour.reshape(-1, 2)

    left_edge_pts = pts[pts[:, 0] <= x + 0.18 * w]
    right_edge_pts = pts[pts[:, 0] >= x + 0.82 * w]

    def vertical_spread(p):
        if len(p) < 3:
            return None
        return float(p[:, 1].max() - p[:, 1].min())

    left_spread = vertical_spread(left_edge_pts)
    right_spread = vertical_spread(right_edge_pts)

    if left_spread is not None and right_spread is not None:
        if left_spread < right_spread:
            return "LEFT"
        return "RIGHT"

    # Fallback: use farthest contour point from center
    center = np.array([cx, cy])
    dists = np.linalg.norm(pts - center, axis=1)
    tip = pts[np.argmax(dists)]

    return "RIGHT" if tip[0] > cx else "LEFT"


def detect_green_arrow(frame, min_area=500):
    """
    Detect green arrow using HSV color segmentation.
    Returns detection dictionary and the green mask.
    """
    H, W = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    # Clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        # Ignore huge green regions if any exist
        if area > 0.45 * W * H:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 20 or h < 20:
            continue

        aspect = w / float(h)

        # Keep loose because perspective can distort the arrow
        if aspect < 0.4 or aspect > 6.0:
            continue

        candidates.append((area, contour))

    if not candidates:
        return None, mask

    _, contour = max(candidates, key=lambda item: item[0])

    x, y, w, h = cv2.boundingRect(contour)

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = x + w // 2
        cy = y + h // 2

    direction = estimate_arrow_direction(contour, x, y, w, h, cx, cy)

    return {
        "contour": contour,
        "bbox": (x, y, w, h),
        "cx": cx,
        "cy": cy,
        "area": cv2.contourArea(contour),
        "direction": direction,
    }, mask


def draw_debug(frame, detection):
    out = frame.copy()

    H, W = out.shape[:2]
    center_x = W // 2

    cv2.line(out, (center_x, 0), (center_x, H), (255, 255, 255), 2)

    if detection is None:
        cv2.putText(
            out,
            "NO GREEN ARROW",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        return out

    x, y, w, h = detection["bbox"]
    cx = detection["cx"]
    cy = detection["cy"]

    cv2.drawContours(out, [detection["contour"]], -1, (0, 255, 0), 2)
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(out, (cx, cy), 6, (255, 0, 255), -1)

    cv2.putText(
        out,
        f"GREEN ARROW: {detection['direction']}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        out,
        f"cx={cx} area={detection['area']:.0f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return out


async def scan_until_green_arrow_centered(drone, center_tol_px=55):
    """
    Rotate slowly until green arrow is found and centered.
    Returns "LEFT" or "RIGHT".
    """
    stable_count = 0
    last_direction = None

    while True:
        frame = read_frame(drone)

        if frame is None:
            print("No frame yet")
            await asyncio.sleep(0.03)
            continue

        detection, mask = detect_green_arrow(frame)

        debug = draw_debug(frame, detection)
        cv2.imshow("FPV Green Arrow Detector", debug)
        cv2.imshow("Green Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

        # No green found: spin slowly and keep searching
        if detection is None:
            print("No green arrow. Searching...")
            await do(drone.rotate_by_yaw_rate_async(20.0, 0.15))
            continue

        frame_center_x = frame.shape[1] / 2
        error_x = detection["cx"] - frame_center_x

        print(
            f"Green arrow found | "
            f"cx={detection['cx']} "
            f"error_x={error_x:+.1f} "
            f"direction={detection['direction']} "
            f"area={detection['area']:.0f}"
        )

        # Green arrow is visible but not centered.
        # Rotate toward the green blob.
        if abs(error_x) > center_tol_px:
            yaw_rate = 15.0 if error_x > 0 else -15.0
            await do(drone.rotate_by_yaw_rate_async(yaw_rate, 0.12))
            stable_count = 0
            continue

        # Arrow is centered. Confirm same reading for a few frames.
        if detection["direction"] == last_direction:
            stable_count += 1
        else:
            last_direction = detection["direction"]
            stable_count = 1

        if stable_count >= 3:
            print(f">> LOCKED GREEN ARROW DIRECTION: {detection['direction']}")
            return detection["direction"]

        await asyncio.sleep(0.05)


async def fly(address: str, alt: float, speed: float):
    client, world, drone = connect(address)

    try:
        print(">> arming")
        drone.enable_api_control()
        drone.arm()

        print(f">> takeoff, climb to {alt:.0f} m")
        await do(drone.takeoff_async())

        # Move to a stable start position
        await do(drone.move_to_position_async(0.0, -35.0, -alt, speed))

        state = drone.get_estimated_kinematics()
        pos = state.get("pose", {}).get("position", {})
        print(
            f"estimated pos: "
            f"x={pos.get('x', 0):.1f} "
            f"y={pos.get('y', 0):.1f} "
            f"z={pos.get('z', 0):.1f}"
        )

        print(">> scanning for green arrow")
        first_turn = await scan_until_green_arrow_centered(drone)

        if first_turn is None:
            print(">> stopped before choosing a turn")
            return

        print(f">> first turn selected: {first_turn}")

        # Turn based on green arrow direction.
        # If the drone turns the wrong way in your sim, swap the signs.
        if first_turn == "LEFT":
            print(">> rotating LEFT")
            await do(drone.rotate_by_yaw_rate_async(-45.0, 2.0))
        else:
            print(">> rotating RIGHT")
            await do(drone.rotate_by_yaw_rate_async(45.0, 2.0))

        print(">> flying forward after first turn")
        await do(drone.move_by_velocity_body_frame_async(3.0, 0.0, 0.0, 4.0))

        print(">> holding hover")
        await do(drone.hover_async())

        # Keep camera window open for debugging
        while True:
            frame = read_frame(drone)

            if frame is None:
                continue

            detection, mask = detect_green_arrow(frame)
            debug = draw_debug(frame, detection)

            cv2.imshow("FPV Green Arrow Detector", debug)
            cv2.imshow("Green Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print(">> landing")
        await do(drone.land_async())
        drone.disarm()
        print(">> done")

    finally:
        client.disconnect()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Green arrow autonomous detector.")
    ap.add_argument("--address", default="127.0.0.1")
    ap.add_argument("--alt", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=3.0)
    args = ap.parse_args()

    asyncio.run(fly(args.address, args.alt, args.speed))


if __name__ == "__main__":
    main()