#!/usr/bin/env python3
"""fly.py - autonomous-flight starter (SimpleFlight). Copy this and build on it.

Shows the two data sources your autonomy needs plus how to command the drone,
using the redteam_sim helper library -- all over one client connection:
  * CONTROL + TELEMETRY  via the ProjectAirSim client (arm / takeoff / move / state)
  * VIDEO                via read_frame(drone) -> BGR numpy image

Launch the game first (see README), then run this:
    python fly.py

Retry from the start line anytime from your own code:  reset(drone)
Coordinates are NED metres: +north, +east, +DOWN -- climbing is NEGATIVE z.
"""
import argparse
import asyncio
import math
import cv2
import numpy as np

from redteam_sim import connect, reset, read_frame  # noqa: F401  (reset for your use)


async def do(cmd):
    """Send a *_async() flight command and wait until the maneuver finishes.
    (They return a task immediately; awaiting the task blocks until it's done.)"""
    await (await cmd)
def frame_difference_data(prev_frame, frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, gray)

    mean_diff = diff.mean()
    max_diff = diff.max()

    _, threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    changed_pixels = cv2.countNonZero(threshold)
    total_pixels = diff.shape[0] * diff.shape[1]
    percent_changed = 100 * changed_pixels / total_pixels

    return {
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "changed_pixels": changed_pixels,
        "percent_changed": percent_changed,
    }

async def fly(address: str, alt: float, speed: float):
    client, world, drone = connect(address)   # connects + spawns the drone at the start line
    try:
        print(">> arming")
        drone.enable_api_control()
        drone.arm()

        print(f">> takeoff, climb to {alt:.0f} m")
        await do(drone.takeoff_async())
        await do(drone.move_to_position_async(0.0, -35.0, -alt, speed))

        # ---- your two data sources --------------------------------------
        # EASY mode: get_ground_truth_kinematics() gives true position.
        # HARD mode: that's off-limits -- use get_estimated_kinematics() + camera.
        state = drone.get_estimated_kinematics()              # telemetry (allowed in both modes)
        pos = state.get("pose", {}).get("position", {})
        print(f"   estimated pos (NED): "
              f"x={pos.get('x', 0):.1f} y={pos.get('y', 0):.1f} z={pos.get('z', 0):.1f}")
        frame = read_frame(drone)                              # vision
        print(f"   camera frame: {None if frame is None else frame.shape}")
        print(f"Frame seen: {frame}")
        print(">> waypoint: 5 m north of the start line")
        hover_target = (5.0, -35.0, -alt)
        await do(drone.move_to_position_async(*hover_target, speed))

        def log_hover_difference() -> None:
            state = drone.get_estimated_kinematics()
            pos = state.get("pose", {}).get("position", {})
            north = float(pos.get("x", 0.0))
            east = float(pos.get("y", 0.0))
            down = float(pos.get("z", 0.0))
            diff_n = north - hover_target[0]
            diff_e = east - hover_target[1]
            diff_d = down - hover_target[2]
            distance = math.sqrt(diff_n * diff_n + diff_e * diff_e + diff_d * diff_d)
            print(
                "   hover diff (m): "
                f"dn={diff_n:+.2f} de={diff_e:+.2f} dd={diff_d:+.2f} "
                f"|d|={distance:.2f}"
            )

        # ================================================================
        # YOUR AUTONOMY GOES HERE: loop on frame = read_frame(drone) (+ telemetry),
        # read the clues, decide the next waypoint, call move_to_position_async(...)
        # until you reach the target. To retry from the start line:  reset(drone)
        # ================================================================

        prev = None
        while True:
            frame = read_frame(drone)

            if frame is None:
                print("No frame yet")
                continue
            
            log_hover_difference()
            if prev is not None:
                data = frame_difference_data(prev, frame)

                print(
                    f"mean_diff={data['mean_diff']:.2f} "
                    f"max_diff={data['max_diff']} "
                    f"changed={data['percent_changed']:.2f}%"
                )
            prev = frame.copy()
            print("frame shape:", frame.shape)

            cv2.imshow("FPV Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print(">> landing")
        await do(drone.land_async())
        drone.disarm()
        print(">> done")
    finally:
        client.disconnect()


def main():
    ap = argparse.ArgumentParser(description="SimpleFlight autonomous-flight starter.")
    ap.add_argument("--address", default="127.0.0.1")
    ap.add_argument("--alt", type=float, default=5.0, help="climb altitude (m)")
    ap.add_argument("--speed", type=float, default=3.0, help="move speed (m/s)")
    args = ap.parse_args()
    asyncio.run(fly(args.address, args.alt, args.speed))


if __name__ == "__main__":
    main()
