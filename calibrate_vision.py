"""Camera and color-mask calibration helper for the PiCar MVP."""

import cv2
import numpy as np

from autonomous_picar_mvp import CAMERA_ID, Camera, detect_lane


def make_sign_masks(frame):
    """Return red and blue masks using the same thresholds as the MVP."""
    h, _, _ = frame.shape
    roi = frame[0:int(h * 0.70), :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
    red_mask = red1 | red2
    blue_mask = cv2.inRange(hsv, np.array([90, 80, 60]), np.array([130, 255, 255]))

    return red_mask, blue_mask


def main():
    """Show camera, lane mask, red mask, and blue mask until q is pressed."""
    cap = Camera(CAMERA_ID)
    if not cap.is_opened():
        print(f"Could not open camera with CAMERA_ID={CAMERA_ID}.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: camera opened but no frame was received.")
                continue

            lane_error, lane_mask = detect_lane(frame)
            red_mask, blue_mask = make_sign_masks(frame)
            red_area = cv2.countNonZero(red_mask)
            blue_area = cv2.countNonZero(blue_mask)

            print(
                f"red_area={red_area} | blue_area={blue_area} | "
                f"lane_error={lane_error}"
            )

            cv2.imshow("camera", frame)
            cv2.imshow("lane mask", lane_mask)
            cv2.imshow("red sign mask", red_mask)
            cv2.imshow("blue sign mask", blue_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("q pressed: exiting calibration.")
                break
    except KeyboardInterrupt:
        print("CTRL+C received: exiting calibration.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and OpenCV windows closed.")


if __name__ == "__main__":
    main()
