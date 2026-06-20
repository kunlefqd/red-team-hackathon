"""
vehicle.py - OpenCV-based vehicle identification.

Actual HSV observations from sim frames:
  ice_cream_truck: white body (S<40, V>200) + red accent (H:0-10 or 165-180, S>120)
  tank:            sandy/tan (H:15-35, S:20-180, V:100-220), ~1926 px at medium range
  boat:            small dark object (V<130, S<60), smaller area than jet
  jet:             large dark object (V<130, S<60), larger area than boat, elongated

Key insight: boat and jet are BOTH dark low-saturation blobs.
Distinguish by area — jet is much larger than boat.
"""
import cv2
import numpy as np


def _largest_contour_info(mask, min_area=200, frame_w=640, frame_h=480):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Skip full-frame bands (sky strip, ground strip)
        if w / frame_w > 0.75 or h / frame_h > 0.75:
            continue
        # Skip sky (top 15%) and very near ground (bottom 8%)
        cy = y + h / 2
        if cy < frame_h * 0.15 or cy > frame_h * 0.92:
            continue
        # Skip contours pinned to left/right frame edge (background bands)
        if x <= 2 or (x + w) >= frame_w - 2:
            continue
        valid.append((area, c))
    if not valid:
        return None
    area, c = max(valid, key=lambda t: t[0])
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / h if h > 0 else 0
    return {"area": area, "x": x, "y": y, "w": w, "h": h,
            "aspect_ratio": aspect,
            "center_x": x + w / 2, "center_y": y + h / 2}


def identify_vehicle_opencv(frame):
    """
    Returns (vehicle_type, info_dict) or (None, None).
    vehicle_type: tank | boat | jet | ice_cream_truck
    """
    fh, fw = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def clean(m):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kern)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
        return m

    # ── Ice cream truck: white body + red accent ──────────────────────────
    kern_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    def clean_sm(m):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kern_sm)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern_sm)
        return m
    white   = clean(cv2.inRange(hsv, (0,   0,  200), (180, 40,  255)))
    red1    = clean_sm(cv2.inRange(hsv, (0,  100,  60), (12,  255, 255)))
    red2    = clean_sm(cv2.inRange(hsv, (160, 100,  60), (180, 255, 255)))
    red     = cv2.bitwise_or(red1, red2)

    white_info = _largest_contour_info(white, min_area=300, frame_w=fw, frame_h=fh)
    red_info   = _largest_contour_info(red,   min_area=20,  frame_w=fw, frame_h=fh)

    if white_info and red_info:
        if abs(white_info["center_x"] - red_info["center_x"]) < white_info["w"] * 1.5:
            return "ice_cream_truck", white_info

    # ── Tank: sandy/tan camo ──────────────────────────────────────────────
    # S>=40 excludes low-sat ground floor (S≈18) and near-gray backgrounds
    tan      = clean(cv2.inRange(hsv, (15, 40, 80), (38, 200, 230)))
    tan_info = _largest_contour_info(tan, min_area=300, frame_w=fw, frame_h=fh)

    if tan_info and 0.3 <= tan_info["aspect_ratio"] <= 3.0:
        return "tank", tan_info

    # ── Boat & jet: both dark blobs — distinguish by area ─────────────────
    # Boat H≈67, Jet H≈86, both S≈20, V≈83-93
    dark     = clean(cv2.inRange(hsv, (0, 0, 20), (180, 70, 135)))
    dark_info = _largest_contour_info(dark, min_area=200, frame_w=fw, frame_h=fh)

    if dark_info:
        aspect = dark_info["aspect_ratio"]
        area   = dark_info["area"]
        # Jet: large elongated dark shape (wings)
        if area > 8000 and 1.3 <= aspect <= 6.0:
            return "jet", dark_info
        # Boat: smaller compact dark shape (aspect cap excludes horizon strips)
        if area <= 12000 and 0.4 <= aspect <= 4.5:
            return "boat", dark_info

    return None, None


def debug_vehicle_view(frame):
    display = frame.copy()
    vehicle, info = identify_vehicle_opencv(frame)
    if vehicle and info:
        x, y, ww, hh = info["x"], info["y"], info["w"], info["h"]
        cv2.rectangle(display, (x, y), (x + ww, y + hh), (0, 255, 255), 2)
        cv2.putText(display, vehicle, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(display, "no vehicle detected", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Vehicle Debug", display)
    cv2.waitKey(1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        vehicle, info = identify_vehicle_opencv(frame)
        print(f"Detected: {vehicle}")
        print(f"Info: {info}")
        debug_vehicle_view(frame)
        cv2.waitKey(0)
    else:
        print("Usage: python vehicle.py logs/vehicle_scan_*.jpg")
