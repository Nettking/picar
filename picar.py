from picarx import Picarx
import cv2
import numpy as np
import time

px = Picarx()

# ---------------- CONFIG ----------------

CAMERA_ID = 0

BASE_SPEED = 22
TURN_SPEED = 20
MAX_STEERING = 30

KP = 0.08

OBSTACLE_LIMIT_CM = 15

STOP_COOLDOWN = 4
RIGHT_COOLDOWN = 4

SIGN_AREA_THRESHOLD = 1800

# ---------------- ROBOT CONTROL ----------------

def stop():
    px.stop()

def drive(speed, steering_angle):
    steering_angle = max(-MAX_STEERING, min(MAX_STEERING, steering_angle))
    px.set_dir_servo_angle(steering_angle)

    if speed > 0:
        px.forward(speed)
    elif speed < 0:
        px.backward(abs(speed))
    else:
        px.stop()

def get_distance_cm():
    try:
        distance = px.ultrasonic.read()
        if distance is None:
            return 999
        return distance
    except Exception:
        return 999

# ---------------- VISION: LANE ----------------

def detect_lane(frame):
    h, w, _ = frame.shape

    # Look at bottom part of image
    roi = frame[int(h * 0.55):h, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # White tape
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 90, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return None, mask

    cx = int(moments["m10"] / moments["m00"])
    error = cx - (w // 2)

    return error, mask

# ---------------- VISION: SIGNS ----------------

def detect_sign(frame):
    h, w, _ = frame.shape

    # Look at upper/middle part, not the floor directly below car
    roi = frame[0:int(h * 0.70), :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Red STOP sign
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

    # Blue right-arrow sign
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

# ---------------- BEHAVIOURS ----------------

def handle_stop_sign():
    print("STOP sign detected")
    stop()
    time.sleep(2.0)

def handle_right_sign():
    print("RIGHT sign detected")

    # Small forward movement before turn
    drive(18, 0)
    time.sleep(0.25)

    # Turn right
    drive(TURN_SPEED, 30)
    time.sleep(1.0)

    # Straighten
    drive(BASE_SPEED, 0)
    time.sleep(0.4)

# ---------------- MAIN ----------------

def main():
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("Could not open camera")
        return

    last_stop_time = 0
    last_right_time = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("No camera frame")
                stop()
                continue

            distance = get_distance_cm()

            if distance < OBSTACLE_LIMIT_CM:
                print(f"Obstacle: {distance} cm")
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
                print("Lane lost")
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
                break

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()