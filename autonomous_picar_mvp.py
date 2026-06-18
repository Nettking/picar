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
import importlib
import math
import shutil
import subprocess
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
state_lock = threading.RLock()
web_server_failed = threading.Event()
flask = None
speech_recognition = None

DEFAULT_THRESHOLDS = {
    "h_min": 0,
    "s_min": 0,
    "v_min": 150,
    "h_max": 180,
    "s_max": 90,
    "v_max": 255,
}

shared_state = {
    "autonomous_active": False,
    "distance_cm": None,
    "detected_sign": None,
    "lane_error": None,
    "steering": 0.0,
    "fault": None,
    "raw_jpeg": None,
    "mask_jpeg": None,
    "latest_frame": None,
    "thresholds": DEFAULT_THRESHOLDS.copy(),
}


def init_robot(dry_run=False, stop_gpio=None):
    """Initialize PiCar hardware unless dry-run mode is requested."""
    global px, DRY_RUN, gpio_stop_button
    DRY_RUN = dry_run
    gpio_stop_latched.clear()
    web_server_failed.clear()
    with state_lock:
        shared_state.update({
            "autonomous_active": False,
            "distance_cm": None,
            "detected_sign": None,
            "lane_error": None,
            "steering": 0.0,
            "fault": None,
            "raw_jpeg": None,
            "mask_jpeg": None,
            "latest_frame": None,
            "thresholds": DEFAULT_THRESHOLDS.copy(),
        })

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

# Minimum number of red pixels required before a STOP sign is detected.
SIGN_AREA_THRESHOLD = 1800

# Blue RIGHT sign detection settings. These are deliberately stricter because a
# false RIGHT detection starts a timed turn. Tune the HSV values first, then the
# size and shape limits if needed.
BLUE_HSV_LOWER = np.array([95, 100, 80])
BLUE_HSV_UPPER = np.array([125, 255, 255])
BLUE_MIN_CONTOUR_AREA = 800
# Reject masks covering more than 8% of the ROI. At 640x480 this rejects the
# previously observed background-like blue area of about 19,600 pixels.
BLUE_MAX_ROI_RATIO = 0.08
BLUE_MIN_FILL_RATIO = 0.25
BLUE_MIN_ASPECT_RATIO = 0.4
BLUE_MAX_ASPECT_RATIO = 2.5
BLUE_MIN_WIDTH = 25
BLUE_MIN_HEIGHT = 25
BLUE_EDGE_MARGIN_RATIO = 0.05
BLUE_MORPH_KERNEL_SIZE = 5

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
    latch_emergency_stop("GPIO emergency stop button pressed")
    print("GPIO EMERGENCY STOP BUTTON PRESSED: motors disabled.")


def latch_emergency_stop(reason):
    """Latch the single emergency-stop state used by GPIO, web, and faults."""
    gpio_stop_latched.set()
    with state_lock:
        shared_state["autonomous_active"] = False
        shared_state["steering"] = 0.0
        shared_state["fault"] = reason
    stop()


def speak(message):
    """Speak a short message without blocking, or print it if espeak is absent."""
    if shutil.which("espeak") is None:
        print(f"Speech: {message}")
        return
    try:
        subprocess.Popen(
            ["espeak", message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"Speech: {message} (could not start espeak: {exc})")


def request_start(source):
    """Enable autonomous driving when all current safety checks allow it."""
    with state_lock:
        if gpio_stop_latched.is_set():
            reason = "Emergency stop is active"
        elif shared_state["fault"]:
            reason = f"Cannot start: {shared_state['fault']}"
        else:
            shared_state["autonomous_active"] = True
            reason = None

    if reason is not None:
        print(f"{source} start rejected: {reason}")
        if source == "voice":
            speak(reason)
        return False, reason

    print(f"Autonomous driving started by {source}.")
    speak("Starting")
    return True, None


def request_stop(source):
    """Disable autonomous driving and immediately stop the motors."""
    with state_lock:
        shared_state["autonomous_active"] = False
        shared_state["steering"] = 0.0
    stop()
    print(f"Autonomous driving stopped by {source}.")
    speak("Stopping")
    return True, None


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

    with motor_lock:
        with state_lock:
            controls_enabled = shared_state["autonomous_active"]
        if gpio_stop_latched.is_set() or not controls_enabled:
            return

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

def detect_lane(frame, thresholds=None):
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
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    lower_white = np.array([
        thresholds["h_min"], thresholds["s_min"], thresholds["v_min"]
    ])
    upper_white = np.array([
        thresholds["h_max"], thresholds["s_max"], thresholds["v_max"]
    ])
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


def calibrate_lane_thresholds(frame):
    """Estimate white-lane HSV thresholds from the lower camera ROI."""
    h, _, _ = frame.shape
    roi = frame[int(h * 0.55):h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    candidate_mask = (hsv[:, :, 2] > 120) & (hsv[:, :, 1] < 120)
    candidate_pixels = hsv[candidate_mask]
    minimum_pixels = max(100, int(roi.shape[0] * roi.shape[1] * 0.001))
    if len(candidate_pixels) < minimum_pixels:
        raise ValueError(
            "Could not find enough bright low-saturation lane pixels. "
            "Improve lighting or reposition the car."
        )

    hue = candidate_pixels[:, 0]
    saturation = candidate_pixels[:, 1]
    value = candidate_pixels[:, 2]

    h_min = max(0, int(np.percentile(hue, 5)) - 10)
    h_max = min(180, int(np.percentile(hue, 95)) + 10)
    if h_max - h_min < 20:
        h_min, h_max = 0, 180

    thresholds = {
        "h_min": h_min,
        "h_max": h_max,
        "s_min": max(0, int(np.percentile(saturation, 5)) - 20),
        "s_max": min(255, int(np.percentile(saturation, 95)) + 20),
        "v_min": max(0, int(np.percentile(value, 5)) - 20),
        "v_max": 255,
    }

    thresholds["h_min"] = min(thresholds["h_min"], thresholds["h_max"])
    thresholds["s_min"] = min(thresholds["s_min"], thresholds["s_max"])
    thresholds["v_min"] = min(thresholds["v_min"], thresholds["v_max"])
    return thresholds


# -----------------------------------------------------------------------------
# Vision: traffic sign detection
# -----------------------------------------------------------------------------

def make_blue_mask(hsv):
    """Create a cleaned mask for saturated blue sign pixels."""
    blue_mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    kernel = np.ones(
        (BLUE_MORPH_KERNEL_SIZE, BLUE_MORPH_KERNEL_SIZE),
        dtype=np.uint8,
    )
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    return blue_mask


def detect_sign(frame):
    """Detect a red STOP sign or a plausible blue RIGHT TURN sign.

    Red detection uses mask area. Blue detection uses the largest cleaned blue
    contour and rejects tiny, oddly shaped, sparse, edge-cut, or
    background-sized regions.

    Args:
        frame: BGR image from OpenCV.

    Returns:
        ``(sign, red_area, blue_area, blue_debug)``. ``blue_debug`` contains
        the mask ratio, largest contour measurements, and validation result.
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

    red_area = cv2.countNonZero(red_mask)
    blue_mask = make_blue_mask(hsv)
    blue_area = cv2.countNonZero(blue_mask)
    roi_area = roi.shape[0] * roi.shape[1]
    blue_roi_ratio = blue_area / roi_area if roi_area else 0.0

    blue_debug = {
        "ratio": blue_roi_ratio,
        "contour_area": 0.0,
        "width": 0,
        "height": 0,
        "aspect_ratio": 0.0,
        "fill_ratio": 0.0,
        "center_ok": False,
        "valid": False,
    }

    contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        x, _, width, height = cv2.boundingRect(largest)
        box_area = width * height
        aspect_ratio = width / height if height else 0.0
        fill_ratio = contour_area / box_area if box_area else 0.0
        center_x = x + width / 2
        edge_margin = roi.shape[1] * BLUE_EDGE_MARGIN_RATIO
        center_ok = edge_margin <= center_x <= roi.shape[1] - edge_margin

        blue_valid = (
            contour_area > BLUE_MIN_CONTOUR_AREA
            and blue_roi_ratio < BLUE_MAX_ROI_RATIO
            and width >= BLUE_MIN_WIDTH
            and height >= BLUE_MIN_HEIGHT
            and BLUE_MIN_ASPECT_RATIO <= aspect_ratio <= BLUE_MAX_ASPECT_RATIO
            and fill_ratio >= BLUE_MIN_FILL_RATIO
            and center_ok
        )
        blue_debug.update({
            "contour_area": contour_area,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "fill_ratio": fill_ratio,
            "center_ok": center_ok,
            "valid": blue_valid,
        })

    if red_area > SIGN_AREA_THRESHOLD:
        return "STOP", red_area, blue_area, blue_debug

    if blue_debug["valid"]:
        return "RIGHT", red_area, blue_area, blue_debug

    return None, red_area, blue_area, blue_debug


# -----------------------------------------------------------------------------
# Optional Flask dashboard
# -----------------------------------------------------------------------------

DASHBOARD_HTML = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PiCar Dashboard</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111827;color:#f9fafb}main{max-width:1050px;margin:auto;padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.card{background:#1f2937;padding:16px;border-radius:10px}
img{width:100%;background:#000;border-radius:6px}.status{display:grid;grid-template-columns:1fr 1fr;gap:8px}.status div{background:#374151;padding:8px;border-radius:5px}
button{padding:12px 16px;margin:4px;border:0;border-radius:5px;font-weight:700;cursor:pointer}.start{background:#22c55e}.stop{background:#f59e0b}.emergency{background:#ef4444;color:white}.calibrate{background:#60a5fa}
label{display:grid;grid-template-columns:80px 1fr 45px;gap:8px;margin:8px 0;align-items:center}input{width:100%}.latched,.message.error{color:#f87171;font-weight:700}.message.success{color:#4ade80;font-weight:700}
</style></head><body><main><h1>PiCar Live Dashboard</h1>
<p>Driving starts disabled. Test with the car lifted on blocks.</p><div class="grid">
<section class="card"><h2>Camera</h2><img src="/video_feed" alt="Live camera"></section>
<section class="card"><h2>Lane mask</h2><img src="/mask_feed" alt="Lane mask"></section>
<section class="card"><h2>Status</h2><div class="status">
<div>Distance: <b id="distance">--</b></div><div>Sign: <b id="sign">--</b></div>
<div>Lane error: <b id="error">--</b></div><div>Steering: <b id="steering">--</b></div>
<div>Autonomous: <b id="active">OFF</b></div><div>Emergency: <b id="emergency">clear</b></div>
<div>Fault: <b id="fault">none</b></div></div>
<p><button class="start" onclick="post('/api/start')">Start autonomous driving</button>
<button class="stop" onclick="post('/api/stop')">Stop autonomous driving</button>
<button class="emergency" onclick="post('/api/emergency_stop')">EMERGENCY STOP</button></p></section>
<section class="card"><h2>Lane HSV thresholds</h2><div id="sliders"></div>
<p><button class="calibrate" onclick="calibrateLane()">Calibrate lane threshold</button></p>
<p id="calibration_status" class="message" role="status" aria-live="polite"></p></section></div>
<script>
const defs=[['h_min','H min',0,180],['h_max','H max',0,180],['s_min','S min',0,255],['s_max','S max',0,255],['v_min','V min',0,255],['v_max','V max',0,255]];
const sliders=document.getElementById('sliders');
for(const [key,name,min,max] of defs){sliders.insertAdjacentHTML('beforeend',`<label>${name}<input id="${key}" type="range" min="${min}" max="${max}"><output id="${key}_out"></output></label>`);const el=document.getElementById(key);el.oninput=()=>{document.getElementById(key+'_out').value=el.value; updateConfig();};}
let timer; function updateConfig(){clearTimeout(timer);timer=setTimeout(()=>post('/api/config',Object.fromEntries(defs.map(([k])=>[k,Number(document.getElementById(k).value)]))),100);}
async function post(url,data){await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:data?JSON.stringify(data):null});refresh();}
async function calibrateLane(){const status=document.getElementById('calibration_status');status.className='message';status.textContent='Calibrating from current frame...';
try{const response=await fetch('/api/calibrate_lane',{method:'POST'});const result=await response.json();
if(!response.ok||!result.ok){throw new Error(result.error||'Calibration failed.');}
for(const [key] of defs){document.getElementById(key).value=result.thresholds[key];document.getElementById(key+'_out').value=result.thresholds[key];}
status.className='message success';status.textContent=result.message;
}catch(err){status.className='message error';status.textContent=err.message;}}
let initialized=false;async function refresh(){const s=await fetch('/api/status').then(r=>r.json());
distance.textContent=s.distance_cm==null?'unavailable':s.distance_cm.toFixed(1)+' cm';sign.textContent=s.detected_sign||'none';error.textContent=s.lane_error==null?'none':s.lane_error;steering.textContent=s.steering.toFixed(1);active.textContent=s.autonomous_active?'ON':'OFF';emergency.textContent=s.emergency_stop?'LATCHED':'clear';emergency.className=s.emergency_stop?'latched':'';fault.textContent=s.fault||'none';
if(!initialized){for(const [k] of defs){document.getElementById(k).value=s.thresholds[k];document.getElementById(k+'_out').value=s.thresholds[k];}initialized=true;}}
refresh();setInterval(refresh,500);
</script></main></body></html>
"""


def create_web_app():
    """Create the small dashboard application around the shared robot state."""
    if flask is None:
        raise RuntimeError("Flask has not been loaded")

    app = flask.Flask(__name__)

    @app.get("/")
    def dashboard():
        return flask.render_template_string(DASHBOARD_HTML)

    def stream_frames(key):
        while not web_server_failed.is_set():
            with state_lock:
                jpeg = shared_state[key]
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.03)

    @app.get("/video_feed")
    def video_feed():
        return flask.Response(stream_frames("raw_jpeg"), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/mask_feed")
    def mask_feed():
        return flask.Response(stream_frames("mask_jpeg"), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/start")
    def start_autonomous():
        ok, error = request_start("web")
        if not ok:
            return flask.jsonify(ok=False, error=error), 409
        return flask.jsonify(ok=True)

    @app.post("/api/stop")
    def stop_autonomous():
        request_stop("web")
        return flask.jsonify(ok=True)

    @app.post("/api/emergency_stop")
    def emergency_stop():
        latch_emergency_stop("Web emergency stop pressed")
        return flask.jsonify(ok=True)

    @app.post("/api/config")
    def update_config():
        values = flask.request.get_json(silent=True) or {}
        limits = {"h_min": 180, "h_max": 180, "s_min": 255, "s_max": 255, "v_min": 255, "v_max": 255}
        try:
            updated = {key: max(0, min(limit, int(values[key]))) for key, limit in limits.items()}
        except (KeyError, TypeError, ValueError):
            return flask.jsonify(ok=False, error="All six HSV threshold values are required"), 400
        if updated["h_min"] > updated["h_max"] or updated["s_min"] > updated["s_max"] or updated["v_min"] > updated["v_max"]:
            return flask.jsonify(ok=False, error="Minimum thresholds cannot exceed maximums"), 400
        with state_lock:
            shared_state["thresholds"] = updated
        return flask.jsonify(ok=True, thresholds=updated)

    @app.post("/api/calibrate_lane")
    def calibrate_lane():
        with state_lock:
            if gpio_stop_latched.is_set():
                return flask.jsonify(
                    ok=False,
                    error="Emergency stop is latched. Calibration is disabled.",
                ), 409
            if shared_state["autonomous_active"]:
                return flask.jsonify(
                    ok=False,
                    error="Stop autonomous driving before calibrating.",
                ), 409
            frame = shared_state["latest_frame"]
            if frame is None:
                return flask.jsonify(
                    ok=False,
                    error="No camera frame is available yet. Wait for the live feed.",
                ), 503
            frame = frame.copy()

        try:
            thresholds = calibrate_lane_thresholds(frame)
        except (ValueError, cv2.error) as exc:
            return flask.jsonify(ok=False, error=str(exc)), 422

        error, lane_mask = detect_lane(frame, thresholds)
        mask_ok, mask_buffer = cv2.imencode(".jpg", lane_mask)
        with state_lock:
            # Recheck safety state before applying a result calculated outside
            # the lock, in case driving or the emergency stop began meanwhile.
            if gpio_stop_latched.is_set():
                return flask.jsonify(
                    ok=False,
                    error="Emergency stop was latched during calibration.",
                ), 409
            if shared_state["autonomous_active"]:
                return flask.jsonify(
                    ok=False,
                    error="Autonomous driving started during calibration.",
                ), 409
            shared_state["thresholds"] = thresholds
            shared_state["lane_error"] = error
            if mask_ok:
                shared_state["mask_jpeg"] = mask_buffer.tobytes()

        return flask.jsonify(
            ok=True,
            thresholds=thresholds,
            message="Lane threshold calibrated from current frame.",
        )

    @app.get("/api/status")
    def status():
        with state_lock:
            data = {
                key: value
                for key, value in shared_state.items()
                if key not in ("raw_jpeg", "mask_jpeg", "latest_frame")
            }
            data["emergency_stop"] = gpio_stop_latched.is_set()
        return flask.jsonify(data)

    return app


def run_web_server():
    """Run Flask in a background thread and fail safe if it exits."""
    try:
        create_web_app().run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
    finally:
        web_server_failed.set()
        with state_lock:
            shared_state["autonomous_active"] = False
        stop()


def load_flask():
    """Load Flask only when the optional web dashboard is requested."""
    global flask
    try:
        flask = importlib.import_module("flask")
    except ImportError as exc:
        raise RuntimeError(
            "The --web option requires Flask. Install it with "
            "'python3 -m pip install -r requirements.txt'."
        ) from exc


# -----------------------------------------------------------------------------
# Optional voice control
# -----------------------------------------------------------------------------

def load_voice_dependencies():
    """Load SpeechRecognition only when microphone features are requested."""
    global speech_recognition
    try:
        speech_recognition = importlib.import_module("speech_recognition")
        importlib.import_module("pyaudio")
    except ImportError as exc:
        raise RuntimeError(
            "Voice control requires SpeechRecognition and PyAudio. Install "
            "them with 'sudo apt install python3-pyaudio portaudio19-dev' and "
            "'python3 -m pip install SpeechRecognition pyaudio'."
        ) from exc


def list_microphones():
    """Print microphone device indexes understood by SpeechRecognition."""
    load_voice_dependencies()
    try:
        names = speech_recognition.Microphone.list_microphone_names()
    except (AttributeError, OSError, ValueError) as exc:
        raise RuntimeError(
            "Could not list microphones. Install PyAudio and PortAudio, then "
            f"check that an audio input device is connected. Original error: {exc}"
        ) from exc
    if not names:
        print("No microphones were found.")
        return
    for index, name in enumerate(names):
        print(f"{index}: {name}")


def run_voice_listener(device_index=None):
    """Listen for the minimal wake-word commands in a background thread."""
    recognizer = speech_recognition.Recognizer()
    try:
        microphone = speech_recognition.Microphone(device_index=device_index)
        with microphone as source:
            print("Voice control: adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
    except (AttributeError, OSError, ValueError) as exc:
        print(
            "Voice control could not open the microphone. Check --list-mics, "
            "install PyAudio/PortAudio, and select --voice-device-index if "
            f"needed. Original error: {exc}"
        )
        return

    print("Voice control ready. Say 'picar start' or 'picar stop'.")
    while True:
        try:
            with microphone as source:
                audio = recognizer.listen(
                    source,
                    timeout=1,
                    phrase_time_limit=3,
                )
            command = recognizer.recognize_google(audio).lower().strip()
            print(f"Voice heard: {command}")
            words = command.split()
            if "picar" not in words:
                continue
            wake_index = words.index("picar")
            command_words = words[wake_index + 1:]
            if command_words == ["start"]:
                request_start("voice")
            elif command_words == ["stop"]:
                request_stop("voice")
        except speech_recognition.WaitTimeoutError:
            continue
        except speech_recognition.UnknownValueError:
            continue
        except speech_recognition.RequestError as exc:
            print(
                "Voice recognition service error. The default Google "
                f"recognizer requires internet access. Error: {exc}"
            )
            time.sleep(2)
        except OSError as exc:
            print(f"Voice microphone error: {exc}")
            time.sleep(2)


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
        "--web",
        action="store_true",
        help="Run the live Flask dashboard on port 5000; driving starts stopped.",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable background voice control for 'picar start' and 'picar stop'.",
    )
    parser.add_argument(
        "--voice-device-index",
        type=int,
        default=None,
        metavar="INDEX",
        help="Select the microphone index used by --voice.",
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List microphone device indexes and exit.",
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
    if args.list_mics:
        list_microphones()
        return
    if args.voice:
        load_voice_dependencies()
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

        if args.web:
            load_flask()
            threading.Thread(target=run_web_server, name="picar-web", daemon=True).start()
            print("Web dashboard: http://<raspberry-pi-ip>:5000")
        if args.voice:
            threading.Thread(
                target=run_voice_listener,
                args=(args.voice_device_index,),
                name="picar-voice",
                daemon=True,
            ).start()
        if not args.web and not args.voice:
            with state_lock:
                shared_state["autonomous_active"] = True

        while True:
            if gpio_stop_latched.is_set():
                if not args.web:
                    print(
                        "Emergency stop is latched; exiting autonomous mode."
                    )
                    break
            if args.web and web_server_failed.is_set():
                print("Web server stopped unexpectedly; exiting safely.")
                break

            ret, frame = cap.read()

            if not ret:
                print(
                    "Warning: camera opened but no frame was received. "
                    "Stopping motors and trying again."
                )
                latch_emergency_stop("Camera capture failed")
                continue

            distance = get_distance_cm()
            sign, red_area, blue_area, blue_debug = detect_sign(frame)
            with state_lock:
                thresholds = shared_state["thresholds"].copy()
                autonomous_active = shared_state["autonomous_active"]
                emergency_stop = gpio_stop_latched.is_set()
            error, lane_mask = detect_lane(frame, thresholds)
            raw_ok, raw_buffer = cv2.imencode(".jpg", frame)
            mask_ok, mask_buffer = cv2.imencode(".jpg", lane_mask)
            with state_lock:
                shared_state.update({
                    "distance_cm": distance,
                    "detected_sign": sign,
                    "lane_error": error,
                    "raw_jpeg": raw_buffer.tobytes() if raw_ok else None,
                    "mask_jpeg": mask_buffer.tobytes() if mask_ok else None,
                    "latest_frame": frame.copy(),
                })

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
                        with state_lock:
                            shared_state["autonomous_active"] = False
                            shared_state["steering"] = 0.0
                            shared_state["fault"] = (
                                "Ultrasonic sensor unavailable after "
                                f"{consecutive_invalid_distance_reads} readings"
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

            if not autonomous_active or emergency_stop:
                if not args.web:
                    cv2.imshow("camera", frame)
                    cv2.imshow("lane mask", lane_mask)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(0.03)
                continue

            now = time.time()

            if sign == "STOP" and now - last_stop_time > STOP_COOLDOWN:
                handle_stop_sign()
                last_stop_time = now
                continue

            if sign == "RIGHT" and now - last_right_time > RIGHT_COOLDOWN:
                handle_right_sign()
                last_right_time = now
                continue

            if error is None:
                print(
                    "Lane lost: driving slowly straight. Check lane tape, "
                    "lighting, camera angle, or white HSV thresholds."
                )
                drive(14, 0)
                continue

            steering = KP * error
            steering = max(-MAX_STEERING, min(MAX_STEERING, steering))

            with state_lock:
                shared_state["steering"] = steering
            drive(BASE_SPEED, steering)

            distance_text = (
                f"{distance:.1f} cm" if distance is not None else "unavailable"
            )
            print(
                f"distance={distance_text} | "
                f"error={error} | steering={steering:.1f} | "
                f"red={red_area} | blue_area={blue_area} | "
                f"blue_ratio={blue_debug['ratio']:.3f} | "
                f"blue_contour={blue_debug['contour_area']:.0f} | "
                f"blue_box={blue_debug['width']}x{blue_debug['height']} | "
                f"blue_aspect={blue_debug['aspect_ratio']:.2f} | "
                f"blue_fill={blue_debug['fill_ratio']:.2f} | "
                f"blue_center={blue_debug['center_ok']} | "
                f"blue_valid={blue_debug['valid']}"
            )

            if not args.web:
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
