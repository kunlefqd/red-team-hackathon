#!/usr/bin/env python3
"""smoke_test.py - end-to-end check that your setup works against a live sim.

Exercises the whole pipeline once -- connect, sensors, camera, arm, takeoff,
move, telemetry, land, reset -- and prints PASS/FAIL per step. Run this first
when you get a new build or set up a new machine, before debugging your autonomy.

Launch the game first (see README), then:
    python smoke_test.py
    python smoke_test.py --easy     # also check the EASY-only ground-truth APIs

Exit code is 0 only if every step passed.
"""
import argparse
import asyncio
import inspect
import sys

from redteam_sim import connect, reset, read_frame

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


async def step(name, action):
    """Run a check and record PASS, or FAIL with the exception. Never raises.

    `action` is either a no-arg callable or a coroutine. Flight commands
    (*_async) return a task when awaited, so we await twice to run to completion.
    """
    try:
        result = action() if callable(action) else action
        for _ in range(2):                      # coroutine -> Task -> completion
            if inspect.isawaitable(result):
                result = await result
        record(name, True)
        return result
    except Exception as e:  # noqa: BLE001 - smoke test reports everything
        record(name, False, f"{type(e).__name__}: {e}")
        return None


async def run(address: str, check_easy: bool):
    print(f">> connecting to {address} ...")
    try:
        client, world, drone = connect(address)
        record("connect + load scene (spawn Drone1)", True)
    except Exception as e:  # noqa: BLE001
        record("connect + load scene (spawn Drone1)", False, f"{type(e).__name__}: {e}")
        return  # nothing else can run

    try:
        await step("world.get_sim_time()", lambda: world.get_sim_time())
        record("drone.sensors populated", bool(drone.sensors),
               ", ".join(drone.sensors) if drone.sensors else "empty")

        frame = read_frame(drone)
        record("read_frame('FPV')", frame is not None,
               "" if frame is None else f"shape={frame.shape}")

        await step("enable_api_control", lambda: drone.enable_api_control())
        await step("arm", lambda: drone.arm())

        await step("takeoff_async", drone.takeoff_async())
        await step("move_to_position_async (climb 5 m)",
                   drone.move_to_position_async(0.0, -35.0, -5.0, 3.0))

        kin = await step("get_estimated_kinematics", lambda: drone.get_estimated_kinematics())
        if kin:
            p = kin.get("pose", {}).get("position", {})
            print(f"      estimated NED pos: x={p.get('x', 0):.1f} "
                  f"y={p.get('y', 0):.1f} z={p.get('z', 0):.1f}")

        if check_easy:
            await step("get_ground_truth_kinematics (EASY)",
                       lambda: drone.get_ground_truth_kinematics())
            await step("get_gps_data('GPS1') (EASY)", lambda: drone.get_gps_data("GPS1"))

        await step("land_async", drone.land_async())
        await step("disarm", lambda: drone.disarm())
        await step("reset (teleport to start)", lambda: reset(drone))
    finally:
        try:
            client.disconnect()
            record("disconnect", True)
        except Exception as e:  # noqa: BLE001
            record("disconnect", False, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description="End-to-end smoke test against a live sim.")
    ap.add_argument("--address", default="127.0.0.1")
    ap.add_argument("--easy", action="store_true",
                    help="also exercise the EASY-only ground-truth / GPS APIs")
    args = ap.parse_args()

    asyncio.run(run(args.address, args.easy))

    passed = sum(RESULTS)
    print(f"\n>> {passed}/{len(RESULTS)} checks passed")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
