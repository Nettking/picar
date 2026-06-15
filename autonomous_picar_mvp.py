"""Autonomous PiCar MVP.

This script is intentionally small and readable for students. It combines:

* OpenCV camera input for lane and sign detection
* Simple HSV color masks for STOP and RIGHT TURN signs
* Ultrasonic obstacle stopping through the PiCar library
* A proportional controller for basic lane following

The code expects the hardware-specific ``picarx`` package to be installed on
Raspberry Pi hardware. See README.md for setup and safety notes.
"""

import argparse
import time

import cv2
import numpy as np

try:
    from picarx import Picarx
except ImportError:
    Picarx = None


# -----------------------------------------------------------------------------
# Hardware setup
# -----------------------------------------------------------------------------

px = None
DRY_RUN = False


def init_robot(dry_run=False):
    """Initialize PiCar hardware unless dry-run mode is requested."""
    global px, DRY_RUN
    DRY_RUN = dry_run

    if DRY_RUN:
        print("Dry-run mode: motor and steering commands will be printed only.")

    if Picarx is None:
        if DRY_RUN:
            print(
                "Warning: picarx is not installed, so dry-run mode will use "
                f"{UNKNOWN_DISTANCE_CM} cm as the ultrasonic fallback."
            )
            return
        raise RuntimeError(
            "Could not import picarx. Install the PiCar/PiCar-X hardware "
            "library on the Raspberry Pi, or use --dry-run for camera and "
            "vision testing without motor movement."
        )

    try:
        px = Picarx()
    except Exception as exc:
        if DRY_RUN:
            print(
                "Warning: could not initialize Picarx in dry-run mode, so "
                f"ultrasonic readings will use {UNKNOWN_DISTANCE_CM} cm. "
                f"Original error: {exc}"
            )
            return
        raise RuntimeError(
            "Could not initialize Picarx. Make sure the PiCar/PiCar-X hardware "
            "library is installed, the script is running on the Raspberry Pi, and "
            "the robot hardware is connected. Original error: " + str(exc)
        ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# OpenCV camera index. Use 0 for the default camera; try 1 if another camera is
# selected or the camera does not open.
CAMERA_ID = 0

# Normal forward driving speed. Start low while testing and increase gradually.
BASE_SPEED = 22

# Forward speed used during the timed right-turn maneuver.
TURN_SPEED = 20

# Maximum steering servo angle in degrees. This keeps commands within a safe
# range even if the lane error is large.
MAX_STEERING = 30

# Proportional steering gain. Larger values turn more sharply; smaller values
# make smoother but weaker corrections.
KP = 0.08

# Stop immediately when the ultrasonic sensor reports an obstacle closer than
# this distance in centimeters.
OBSTACLE_LIMIT_CM = 15

# Minimum number of seconds between repeated reactions to the same type of sign.
STOP_COOLDOWN = 4
RIGHT_COOLDOWN = 4

# Minimum number of colored pixels required before a red or blue region counts
# as a sign. Increase this to reduce false positives; decrease it if signs are
# being missed.
SIGN_AREA_THRESHOLD = 1800

# Fallback distance used when the ultrasonic sensor has no usable reading. A
# large value lets the car continue instead of stopping for a sensor glitch.
UNKNOWN_DISTANCE_CM = 999


# -----------------------------------------------------------------------------
# Robot control helpers
# -----------------------------------------------------------------------------

def stop():
    """Stop the drive motors."""
    if DRY_RUN:
        print("DRY RUN motor command: stop()")
        return
    px.stop()


def drive(speed, steering_angle):
    """Drive with a requested speed and steering angle.

    Args:
        speed: Positive values drive forward, negative values drive backward,
            and zero stops the motors.
        steering_angle: Desired steering angle in degrees. The value is clipped
            to ``MAX_STEERING`` so the servo is not commanded beyond the MVP's
            expected steering range.
    """
    steering_angle = max(-MAX_STEERING, min(MAX_STEERING, steering_angle))

    if DRY_RUN:
        print(
            "DRY RUN motor command: "
            f"speed={speed}, steering_angle={steering_angle:.1f}"
        )
        return

    px.set_dir_servo_angle(steering_angle)

    if speed > 0:
        px.forward(speed)
    elif speed < 0:
        px.backward(abs(speed))
    else:
        px.stop()


def get_distance_cm():
    """Read the ultrasonic distance sensor.

    Returns:
        Distance in centimeters. If the sensor read fails or returns ``None``,
        ``UNKNOWN_DISTANCE_CM`` is returned so a temporary bad reading does not
        crash the program.
    """
    if px is None:
        print(
            "Warning: ultrasonic hardware is not available; "
            f"using {UNKNOWN_DISTANCE_CM} cm fallback."
        )
        return UNKNOWN_DISTANCE_CM

    try:
        distance = px.ultrasonic.read()
        if distance is None:
            print(
                "Warning: ultrasonic sensor returned no reading; "
                f"using {UNKNOWN_DISTANCE_CM} cm fallback."
            )
            return UNKNOWN_DISTANCE_CM
        return distance
    except Exception as exc:
        print(
            "Warning: could not read ultrasonic sensor; "
            f"using {UNKNOWN_DISTANCE_CM} cm fallback. Error: {exc}"
        )
        return UNKNOWN_DISTANCE_CM


# -----------------------------------------------------------------------------
# Vision: lane detection
# -----------------------------------------------------------------------------

def detect_lane(frame):
    """Find the horizontal lane offset using a white color mask.

    The function looks only at the lower part of the camera image because that
    is where nearby lane markings should appear. It thresholds for bright white
    pixels, computes the mask's centroid, and compares that centroid with the
    image center.

    Args:
        frame: BGR image from OpenCV.

    Returns:
        A tuple ``(error, mask)``. ``error`` is positive when the lane center is
        to the right of the image center and negative when it is to the left. If
        no lane is found, ``error`` is ``None``. ``mask`` is returned for display
        and debugging.
    """
    h, w, _ = frame.shape

    # Look at the bottom part of the image and ignore the horizon/background.
    roi = frame[int(h * 0.55):h, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # White tape or bright lane markings. Tune these values if your lane color
    # or lighting is different.
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 90, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Remove tiny specks, then expand the remaining lane pixels slightly.
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return None, mask

    cx = int(moments["m10"] / moments["m00"])
    error = cx - (w // 2)

    return error, mask


# -----------------------------------------------------------------------------
# Vision: traffic sign detection
# -----------------------------------------------------------------------------

def detect_sign(frame):
    """Detect simple red STOP and blue RIGHT TURN signs by color area.

    This MVP does not recognize shapes or text. It only counts red and blue
    pixels in the upper/middle part of the image. Keep the test area simple and
    avoid red or blue clutter in the background.

    Args:
        frame: BGR image from OpenCV.

    Returns:
        ``(sign, red_area, blue_area)`` where ``sign`` is ``"STOP"``,
        ``"RIGHT"``, or ``None``. The area values are useful for tuning
        ``SIGN_AREA_THRESHOLD``.
    """
    h, _, _ = frame.shape

    # Look at the upper/middle part, not the floor directly below the car.
    roi = frame[0:int(h * 0.70), :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Red wraps around the HSV hue range, so two masks are combined.
    red1 = cv2.inRange(
        hsv,
        np.array([0, 100, 80]),
        np.array([10, 255, 255])
    )

    red2 = cv2.inRange(
        hsv,
        np.array([170, 100, 80]),
        np.array([180, 255, 255])
    )

    red_mask = red1 | red2

    # Blue mask for a simple right-arrow sign.
    blue_mask = cv2.inRange(
        hsv,
        np.array([90, 80, 60]),
        np.array([130, 255, 255])
    )

    red_area = cv2.countNonZero(red_mask)
    blue_area = cv2.countNonZero(blue_mask)

    if red_area > SIGN_AREA_THRESHOLD:
        return "STOP", red_area, blue_area

    if blue_area > SIGN_AREA_THRESHOLD:
        return "RIGHT", red_area, blue_area

    return None, red_area, blue_area


# -----------------------------------------------------------------------------
# Sign behaviors
# -----------------------------------------------------------------------------

def handle_stop_sign():
    """Stop briefly after detecting a STOP sign."""
    print("STOP sign detected: stopping for 2 seconds.")
    stop()
    time.sleep(2.0)


def handle_right_sign():
    """Perform a simple timed right turn after detecting a RIGHT sign."""
    print("RIGHT sign detected: performing timed right turn.")

    # Small forward movement before turn.
    drive(18, 0)
    time.sleep(0.25)

    # Turn right. This is intentionally simple and time-based for the MVP.
    drive(TURN_SPEED, 30)
    time.sleep(1.0)

    # Straighten after the turn.
    drive(BASE_SPEED, 0)
    time.sleep(0.4)


# -----------------------------------------------------------------------------
# Main program loop
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the autonomous PiCar MVP.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run camera, lane detection, sign detection, and ultrasonic fallback "
            "while printing intended motor commands without moving motors."
        ),
    )
    return parser.parse_args()


def main():
    """Run the autonomous driving loop until the user stops the program."""
    args = parse_args()
    init_robot(dry_run=args.dry_run)

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(
            f"Could not open camera with CAMERA_ID={CAMERA_ID}. "
            "Check the camera connection, enable the Raspberry Pi camera if "
            "needed, or try changing CAMERA_ID to 1."
        )
        return

    last_stop_time = 0
    last_right_time = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print(
                    "Warning: camera opened but no frame was received. "
                    "Stopping motors and trying again."
                )
                stop()
                continue

            distance = get_distance_cm()

            if distance < OBSTACLE_LIMIT_CM:
                print(
                    f"Obstacle detected at {distance:.1f} cm; stopping. "
                    f"Threshold is {OBSTACLE_LIMIT_CM} cm."
                )
                stop()
                time.sleep(0.2)
                continue

            sign, red_area, blue_area = detect_sign(frame)
            now = time.time()

            if sign == "STOP" and now - last_stop_time > STOP_COOLDOWN:
                handle_stop_sign()
                last_stop_time = now
                continue

            if sign == "RIGHT" and now - last_right_time > RIGHT_COOLDOWN:
                handle_right_sign()
                last_right_time = now
                continue

            error, lane_mask = detect_lane(frame)

            if error is None:
                print(
                    "Lane lost: driving slowly straight. Check lane tape, "
                    "lighting, camera angle, or white HSV thresholds."
                )
                drive(14, 0)
                continue

            steering = KP * error
            steering = max(-MAX_STEERING, min(MAX_STEERING, steering))

            drive(BASE_SPEED, steering)

            print(
                f"distance={distance:.1f} cm | "
                f"error={error} | steering={steering:.1f} | "
                f"red={red_area} | blue={blue_area}"
            )

            cv2.imshow("camera", frame)
            cv2.imshow("lane mask", lane_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("q pressed: exiting safely.")
                break

    except KeyboardInterrupt:
        print("CTRL+C received: stopping safely.")

    finally:
        stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Motors stopped, camera released, and OpenCV windows closed.")


if __name__ == "__main__":
    main()
