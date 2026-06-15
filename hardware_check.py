"""Safe hardware check for a Raspberry Pi PiCar/PiCar-X robot.

Run this before autonomous driving. The script asks for Enter before every
motor or steering movement so students can keep the car lifted and clear.
"""

import time

import cv2
from picarx import Picarx

CAMERA_ID = 0
LOW_TEST_SPEED = 10
STEERING_LEFT = -25
STEERING_CENTER = 0
STEERING_RIGHT = 25


def wait_for_enter(message):
    """Pause until the user is ready for the next hardware action."""
    input(f"\n{message}\nPress Enter when ready...")


def test_camera():
    """Open the camera and capture one frame."""
    print("Testing camera...")
    cap = cv2.VideoCapture(CAMERA_ID)
    try:
        if not cap.isOpened():
            print(f"Camera FAILED: could not open CAMERA_ID={CAMERA_ID}.")
            return

        ret, frame = cap.read()
        if not ret:
            print("Camera FAILED: opened camera but could not read a frame.")
            return

        print(f"Camera OK: captured frame {frame.shape[1]}x{frame.shape[0]}.")
    finally:
        cap.release()


def test_ultrasonic(px):
    """Read and print the ultrasonic sensor distance."""
    print("\nTesting ultrasonic sensor...")
    try:
        distance = px.ultrasonic.read()
        print(f"Ultrasonic reading: {distance} cm")
    except Exception as exc:
        print(f"Ultrasonic FAILED: {exc}")


def main():
    """Run camera, ultrasonic, steering, and low-speed motor checks."""
    print("Safe PiCar hardware check")
    print("Keep the car on blocks with wheels off the ground.")
    print("Keep hands, hair, cables, and loose clothing away from wheels.")

    px = Picarx()

    try:
        test_camera()
        test_ultrasonic(px)

        wait_for_enter("Steering will move LEFT.")
        px.set_dir_servo_angle(STEERING_LEFT)
        time.sleep(0.5)

        wait_for_enter("Steering will move CENTER.")
        px.set_dir_servo_angle(STEERING_CENTER)
        time.sleep(0.5)

        wait_for_enter("Steering will move RIGHT.")
        px.set_dir_servo_angle(STEERING_RIGHT)
        time.sleep(0.5)

        wait_for_enter("Steering will return to CENTER.")
        px.set_dir_servo_angle(STEERING_CENTER)
        time.sleep(0.5)

        wait_for_enter(f"Motors will move FORWARD at low speed {LOW_TEST_SPEED}.")
        px.forward(LOW_TEST_SPEED)
        time.sleep(0.5)

        wait_for_enter("Motors will STOP.")
        px.stop()
        print("Hardware check complete.")
    except KeyboardInterrupt:
        print("\nCTRL+C received: stopping safely.")
    finally:
        px.stop()
        px.set_dir_servo_angle(STEERING_CENTER)
        print("Motors stopped and steering centered.")


if __name__ == "__main__":
    main()
