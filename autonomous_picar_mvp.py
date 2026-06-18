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
import math
import threading
import time

import cv2
import numpy as np

try:
    from picarx import Picarx
except ImportError:
    Picarx = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

try:
    from gpiozero import Button
except ImportError:
    Button = None


# -----------------------------------------------------------------------------
# Hardware setup
# -----------------------------------------------------------------------------

px = None
DRY_RUN = False
gpio_stop_latched = threading.Event()
gpio_stop_button = None
motor_lock = threading.RLock()


def init_robot(dry_run=False, stop_gpio=None):
    """Initialize PiCar hardware unless dry-run mode is requested."""
    global px, DRY_RUN, gpio_stop_button
    DRY_RUN = dry_run
    gpio_stop_latched.clear()

    if DRY_RUN:
        print("Dry-run mode: motor and steering commands will be printed only.")

    if Picarx is None:
        if DRY_RUN:
            print(
                "Warning: picarx is not installed, so dry-run mode will use "
                "an unavailable ultrasonic reading."
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
                "ultrasonic readings will be unavailable. "
                f"Original error: {exc}"
            )
            return
        raise RuntimeError(
            "Could not initialize Picarx. Make sure the PiCar/PiCar-X hardware "
            "library is installed, the script is running on the Raspberry Pi, and "
            "the robot hardware is connected. Original error: " + str(exc)
        ) from exc

    if stop_gpio is not None:
        if Button is None:
            raise RuntimeError(
                "A hardware stop GPIO was requested, but gpiozero is not "
                "installed. Install python3-gpiozero and python3-lgpio."
            )
        try:
            gpio_stop_button = Button(stop_gpio, pull_up=True, bounce_time=0.05)
            gpio_stop_button.when_pressed = trigger_gpio_stop
        except Exception as exc:
            stop()
            raise RuntimeError(
                f"Could not configure hardware stop on BCM GPIO {stop_gpio}. "
                "Connect a normally-open button between that pin and ground. "
                f"Original error: {exc}"
            ) from exc
        print(
            f"GPIO emergency stop button armed on BCM GPIO {stop_gpio} "
            "(button to ground)."
        )


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

# Stop after this many invalid ultrasonic readings in a row. One or two bad
# reads can be transient; repeated failures mean obstacle detection cannot be
# trusted and autonomous driving must fail safe.
MAX_CONSECUTIVE_INVALID_DISTANCE_READS = 3

# Minimum number of seconds between repeated reactions to the same type of sign.
STOP_COOLDOWN = 4
RIGHT_COOLDOWN = 4

# Minimum number of colored pixels required before a red or blue region counts
# as a sign. Increase this to reduce false positives; decrease it if signs are
# being missed.
SIGN_AREA_THRESHOLD = 1800

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# -----------------------------------------------------------------------------
# Robot control helpers
# -----------------------------------------------------------------------------

def stop():
    """Stop both drive motors using redundant hardware commands."""
    if DRY_RUN:
        print("DRY RUN motor command: stop()")
        return
    if px is None:
        return

    errors = []
    with motor_lock:
        try:
            px.stop()
        except Exception as exc:
            errors.append(f"stop(): {exc}")

        set_motor_speed = getattr(px, "set_motor_speed", None)
        if callable(set_motor_speed):
            for motor in (1, 2):
                try:
                    set_motor_speed(motor, 0)
                except Exception as exc:
                    errors.append(f"motor {motor}: {exc}")

    if errors:
        print("Warning: one or more motor stop commands failed: " + "; ".join(errors))


def trigger_gpio_stop():
    """Latch the GPIO stop and immediately command the motor controller off."""
    gpio_stop_latched.set()
    print("GPIO EMERGENCY STOP BUTTON PRESSED: motors disabled.")
    stop()


def release_gpio_stop():
    """Release GPIO resources used by the emergency stop button."""
    global gpio_stop_button
    if gpio_stop_button is not None:
        gpio_stop_button.close()
        gpio_stop_button = None


class Camera:
    """Raspberry Pi 5 camera wrapper with an OpenCV fallback."""

    def __init__(self, camera_id=0, backend="auto"):
        self.backend = None
        self.camera = None

        if backend in ("auto", "picamera2") and Picamera2 is not None:
            camera = None
            try:
                camera = Picamera2(camera_id)
                config = camera.create_preview_configuration(
                    main={
                        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                        "format": "BGR888",
                    }
                )
                camera.configure(config)
                camera.start()
                self.camera = camera
                self.backend = "picamera2"
                time.sleep(0.5)
                return
            except Exception as exc:
                if camera is not None:
                    try:
                        camera.close()
                    except Exception as close_exc:
                        print(
                            "Warning: failed to close Picamera2 after "
                            f"initialization error: {close_exc}"
                        )
                if backend == "picamera2":
                    raise
                print(
                    "Warning: Picamera2 initialization failed; falling back "
                    f"to OpenCV VideoCapture. Error: {exc}"
                )

        if backend == "picamera2":
            raise RuntimeError(
                "Picamera2 is not installed. Install python3-picamera2 on "
                "Raspberry Pi OS."
            )

        camera = cv2.VideoCapture(camera_id)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera = camera
        self.backend = "opencv"

    def is_opened(self):
        """Return whether the selected camera backend initialized."""
        if self.backend == "picamera2":
            return self.camera is not None
        return self.camera.isOpened()

    def read(self):
        """Return an OpenCV-style ``(success, BGR frame)`` pair."""
        if self.backend == "picamera2":
            try:
                return True, self.camera.capture_array("main")
            except Exception as exc:
                print(f"Warning: Picamera2 frame capture failed: {exc}")
                return False, None
        return self.camera.read()

    def release(self):
        """Release the selected camera backend."""
        if self.camera is None:
            return
        if self.backend == "picamera2":
            self.camera.stop()
            self.camera.close()
        else:
            self.camera.release()
        self.camera = None


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

    with motor_lock:
        if gpio_stop_latched.is_set():
            stop()
            return

        px.set_dir_servo_angle(steering_angle)

        if speed > 0:
            px.forward(speed)
        elif speed < 0:
            px.backward(abs(speed))
        else:
            stop()


def get_distance_cm():
    """Read the ultrasonic distance sensor.

    Returns:
        A positive, finite distance in centimeters. If the sensor read fails or
        returns an invalid value (including the PiCar-X ``-1``/``-2`` error
        sentinels), ``None`` is returned. The main loop tolerates brief glitches
        but stops after repeated invalid readings.
    """
    if px is None:
        print("Warning: ultrasonic hardware is not available.")
        return None

    try:
        distance = px.ultrasonic.read()
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            print(
                f"Warning: ultrasonic sensor returned invalid reading "
                f"{distance!r}."
            )
            return None

        if not math.isfinite(distance) or distance <= 0:
            print(
                f"Warning: ultrasonic sensor returned invalid reading "
                f"{distance:g} cm."
            )
            return None

        return distance
    except Exception as exc:
        print(f"Warning: could not read ultrasonic sensor. Error: {exc}")
        return None


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
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "picamera2", "opencv"),
        default="auto",
        help="Use Picamera2 on Raspberry Pi 5, or OpenCV for a USB camera.",
    )
    parser.add_argument(
        "--stop-gpio",
        type=int,
        default=None,
        metavar="BCM_PIN",
        help=(
            "BCM GPIO connected to a normally-open GPIO emergency stop button "
            "whose other terminal is connected to ground."
        ),
    )
    return parser.parse_args()


def main():
    """Run the autonomous driving loop until the user stops the program."""
    args = parse_args()
    init_robot(dry_run=args.dry_run, stop_gpio=args.stop_gpio)

    cap = None
    last_stop_time = 0
    last_right_time = 0
    consecutive_invalid_distance_reads = 0

    try:
        cap = Camera(CAMERA_ID, backend=args.camera_backend)
        if not cap.is_opened():
            print(
                f"Could not open camera with CAMERA_ID={CAMERA_ID}. "
                "Check the camera connection, enable the Raspberry Pi camera if "
                "needed, or try changing CAMERA_ID to 1."
            )
            return

        while True:
            if gpio_stop_latched.is_set():
                stop()
                print("GPIO emergency stop is latched; exiting autonomous mode.")
                break

            ret, frame = cap.read()

            if not ret:
                print(
                    "Warning: camera opened but no frame was received. "
                    "Stopping motors and trying again."
                )
                stop()
                continue

            distance = get_distance_cm()

            if distance is None:
                if DRY_RUN:
                    print(
                        "Dry-run: ultrasonic reading unavailable; continuing "
                        "vision processing without obstacle sensing."
                    )
                else:
                    consecutive_invalid_distance_reads += 1
                    print(
                        "Ultrasonic reading unavailable "
                        f"({consecutive_invalid_distance_reads}/"
                        f"{MAX_CONSECUTIVE_INVALID_DISTANCE_READS} consecutive)."
                    )
                    if (
                        consecutive_invalid_distance_reads
                        >= MAX_CONSECUTIVE_INVALID_DISTANCE_READS
                    ):
                        print(
                            "Ultrasonic sensor failure limit reached; stopping "
                            "for safety. Check sensor power and trigger/echo "
                            "wiring."
                        )
                        stop()
                        time.sleep(0.2)
                        continue
            else:
                consecutive_invalid_distance_reads = 0

            if distance is not None and distance < OBSTACLE_LIMIT_CM:
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

            distance_text = (
                f"{distance:.1f} cm" if distance is not None else "unavailable"
            )
            print(
                f"distance={distance_text} | "
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
        release_gpio_stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Motors stopped, camera released, and OpenCV windows closed.")


if __name__ == "__main__":
    main()
