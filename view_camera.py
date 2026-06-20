#!/usr/bin/env python3
"""view_camera.py - live OpenCV window of the drone's camera feed.

The simplest possible vision demo: connect, then stream the nose camera into an
OpenCV window so you can see exactly what your autonomy sees (arrows, spheres,
vehicles). No flight -- the camera renders whether or not the drone is armed.

Launch the game first (see README), then run:
    python view_camera.py                 # FPV nose cam
    python view_camera.py --camera Chase  # follow cam

Press q or ESC in the window to quit.
"""
import argparse

import cv2

from redteam_sim import connect, read_frame


def main():
    ap = argparse.ArgumentParser(description="Live OpenCV view of the drone camera.")
    ap.add_argument("--address", default="127.0.0.1")
    ap.add_argument("--camera", default="FPV", help='camera id: "FPV" or "Chase"')
    args = ap.parse_args()

    client, _world, drone = connect(args.address)
    win = f"Red Team Hack Sim - {args.camera}"
    print(f">> streaming {args.camera} -- press q or ESC in the window to quit")

    frames = 0
    try:
        while True:
            frame = read_frame(drone, camera=args.camera)  # BGR numpy image, or None
            if frame is None:
                continue
            frames += 1

            # Overlay a frame counter so you can confirm the feed is live.
            cv2.putText(frame, f"{args.camera}  frame {frames}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, frame)

            # waitKey drives the window's event loop; 1 ms = run as fast as frames arrive.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        client.disconnect()
        print(">> done")


if __name__ == "__main__":
    main()
