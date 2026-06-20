#!/usr/bin/env python3
import argparse
import asyncio
import json
import socket
import time

import cv2
import numpy as np

from redteam_sim import connect, read_frame


FEATURE_PARAMS = dict(
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=8,
    blockSize=7,
)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


async def do(cmd):
    await (await cmd)


class OpticalHold:
    def __init__(
        self,
        gain_forward=0.0010,
        gain_side=0.0020,
        gain_vertical=0.0015,
        max_vx=0.6,
        max_vy=0.6,
        max_vz=0.35,
    ):
        self.prev_gray = None
        self.prev_pts = None
        self.prev_t = None
        self.seq = 0

        self.fx_s = 0.0
        self.fy_s = 0.0
        self.exp_s = 0.0

        self.alpha = 0.25

        self.gain_forward = gain_forward
        self.gain_side = gain_side
        self.gain_vertical = gain_vertical

        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_vz = max_vz

    def detect_features(self, gray):
        return cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)

    def reset(self, gray):
        self.prev_gray = gray
        self.prev_pts = self.detect_features(gray)
        self.prev_t = time.time()

    def update(self, frame):
        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 30:
            self.reset(gray)
            return None, frame

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            **LK_PARAMS,
        )

        if next_pts is None or status is None:
            self.reset(gray)
            return None, frame

        old = self.prev_pts.reshape(-1, 2)
        new = next_pts.reshape(-1, 2)
        valid = status.flatten() == 1

        old = old[valid]
        new = new[valid]

        if len(new) < 25:
            self.reset(gray)
            return None, frame

        flow = new - old

        # Remove wild outliers.
        med_flow = np.median(flow, axis=0)
        dist = np.linalg.norm(flow - med_flow, axis=1)
        med_dist = np.median(dist)
        keep = dist < max(2.5 * med_dist, 1.0)

        old_f = old[keep]
        new_f = new[keep]
        flow_f = flow[keep]

        if len(flow_f) < 15:
            old_f = old
            new_f = new
            flow_f = flow

        dx_px, dy_px = np.median(flow_f, axis=0)

        h, w = gray.shape
        center = np.array([w / 2.0, h / 2.0])

        # Positive expansion = scene expands outward, usually forward motion.
        rel = old_f - center
        rel_norm = np.linalg.norm(rel, axis=1) + 1e-6
        rel_unit = rel / rel_norm[:, None]
        radial_flow = np.sum(flow_f * rel_unit, axis=1)
        expansion_px = np.median(radial_flow)

        dt = max(now - self.prev_t, 1e-3)

        flow_x_s = dx_px / dt
        flow_y_s = dy_px / dt
        expansion_s = expansion_px / dt

        self.fx_s = (1.0 - self.alpha) * self.fx_s + self.alpha * flow_x_s
        self.fy_s = (1.0 - self.alpha) * self.fy_s + self.alpha * flow_y_s
        self.exp_s = (1.0 - self.alpha) * self.exp_s + self.alpha * expansion_s

        # Body-frame correction guesses:
        # vx = forward/back, vy = left/right, vz = down/up.
        vx_body = -self.exp_s * self.gain_forward
        vy_body = -self.fx_s * self.gain_side
        vz_body = -self.fy_s * self.gain_vertical

        vx_body = float(np.clip(vx_body, -self.max_vx, self.max_vx))
        vy_body = float(np.clip(vy_body, -self.max_vy, self.max_vy))
        vz_body = float(np.clip(vz_body, -self.max_vz, self.max_vz))

        confidence = float(min(len(flow_f) / 120.0, 1.0))

        payload = {
            "type": "optical_hold",
            "seq": self.seq,
            "t": now,
            "dt": dt,
            "dx_px": float(dx_px),
            "dy_px": float(dy_px),
            "expansion_px": float(expansion_px),
            "flow_x_px_s": float(self.fx_s),
            "flow_y_px_s": float(self.fy_s),
            "expansion_px_s": float(self.exp_s),
            "vx_body": vx_body,
            "vy_body": vy_body,
            "vz_body": vz_body,
            "confidence": confidence,
            "tracked_points": int(len(flow_f)),
        }

        self.seq += 1

        # Debug drawing.
        debug = frame.copy()
        for p0, p1 in zip(old_f[::5], new_f[::5]):
            x0, y0 = p0.astype(int)
            x1, y1 = p1.astype(int)
            cv2.line(debug, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.circle(debug, (x1, y1), 2, (0, 0, 255), -1)

        cv2.putText(
            debug,
            f"vx={vx_body:+.2f} vy={vy_body:+.2f} vz={vz_body:+.2f} conf={confidence:.2f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # Redetect every frame for robustness.
        self.prev_gray = gray
        self.prev_pts = self.detect_features(gray)
        self.prev_t = now

        return payload, debug


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default="127.0.0.1")
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=5005)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--direct-hold", action="store_true")
    ap.add_argument("--duration", type=float, default=0.08)

    ap.add_argument("--gain-forward", type=float, default=0.0010)
    ap.add_argument("--gain-side", type=float, default=0.0020)
    ap.add_argument("--gain-vertical", type=float, default=0.0015)

    ap.add_argument("--sign-x", type=float, default=1.0)
    ap.add_argument("--sign-y", type=float, default=1.0)
    ap.add_argument("--sign-z", type=float, default=1.0)

    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = (args.udp_host, args.udp_port)

    hold = OpticalHold(
        gain_forward=args.gain_forward,
        gain_side=args.gain_side,
        gain_vertical=args.gain_vertical,
    )

    client, world, drone = connect(args.address)

    try:
        if args.direct_hold:
            print(">> DIRECT HOLD MODE: taking API control")
            drone.enable_api_control()
            drone.arm()
            await do(drone.takeoff_async())
            await do(drone.move_to_position_async(0.0, -35.0, -5.0, 3.0))
        else:
            print(">> OBSERVE/UDP MODE: not commanding drone")

        print(f">> sending UDP optical flow to {udp_target}")

        while True:
            frame = read_frame(drone)
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            payload, debug = hold.update(frame)

            if payload is not None:
                # Sign flips are for fast live tuning.
                payload["vx_body"] *= args.sign_x
                payload["vy_body"] *= args.sign_y
                payload["vz_body"] *= args.sign_z

                if payload["confidence"] < 0.25:
                    payload["vx_body"] = 0.0
                    payload["vy_body"] = 0.0
                    payload["vz_body"] = 0.0

                sock.sendto(json.dumps(payload).encode("utf-8"), udp_target)

                print(
                    f"seq={payload['seq']:04d} "
                    f"dx={payload['dx_px']:+.2f} dy={payload['dy_px']:+.2f} "
                    f"vx={payload['vx_body']:+.2f} "
                    f"vy={payload['vy_body']:+.2f} "
                    f"vz={payload['vz_body']:+.2f} "
                    f"conf={payload['confidence']:.2f}"
                )

                if args.direct_hold:
                    await do(
                        drone.move_by_velocity_body_frame_async(
                            payload["vx_body"],
                            payload["vy_body"],
                            payload["vz_body"],
                            args.duration,
                        )
                    )

            if args.show:
                cv2.imshow("optical_hold debug", debug)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print(">> stopping")

    finally:
        if args.direct_hold:
            try:
                await do(drone.land_async())
                drone.disarm()
            except Exception:
                pass
        client.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())