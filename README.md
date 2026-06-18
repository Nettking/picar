# Autonomous PiCar MVP

A minimum viable autonomous driving project for a Raspberry Pi-based PiCar. The goal is to give students a readable starting point for experimenting with simple autonomous driving behaviors without machine learning, cloud services, or a complex software architecture.

The main script, `autonomous_picar_mvp.py`, uses:

- OpenCV camera input for lane and traffic sign detection
- A PiCar/PiCar-X-compatible `picarx` Python library for motor, steering, and ultrasonic sensor control
- Simple proportional steering for lane following
- HSV color thresholding for STOP and RIGHT TURN sign detection
- An ultrasonic distance threshold for obstacle stopping

> **Important:** This project assumes a `picarx` Python package is available on the Raspberry Pi. Different PiCar kits use different libraries, so verify your hardware and software before running the car on the floor.

## Quick Start

Run these commands from the repository directory on your Raspberry Pi:

```bash
sudo apt update
sudo apt install python3-picamera2 python3-gpiozero python3-lgpio python3-opencv python3-numpy
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -c "from picarx import Picarx; print('picarx is installed')"
python3 autonomous_picar_mvp.py --dry-run
python3 hardware_check.py
python3 autonomous_picar_mvp.py
```

Stop the program at any time with **CTRL+C**. If a display window is open, you can also press **q** while the OpenCV window is focused.

## Raspberry Pi 5 Setup

Raspberry Pi 5 uses the libcamera camera stack. Install the OS-provided
Picamera2 and GPIO packages before creating the virtual environment:

```bash
sudo apt update
sudo apt install python3-picamera2 python3-gpiozero python3-lgpio python3-opencv python3-numpy
python3 -m venv --system-site-packages .venv
```

The default `--camera-backend auto` mode prefers Picamera2 for a CSI-connected
Raspberry Pi camera and falls back to OpenCV for USB cameras. A backend can also
be selected explicitly:

```bash
python3 autonomous_picar_mvp.py --camera-backend picamera2
python3 autonomous_picar_mvp.py --camera-backend opencv
```

## GPIO Emergency Stop Button

For a software-mediated stop that does not depend on the camera loop, connect a
normally-open momentary switch between an unused BCM GPIO pin and ground. For
example, with BCM GPIO 17:

```bash
python3 autonomous_picar_mvp.py --stop-gpio 17
```

Pressing the switch invokes a GPIO callback immediately, latches the stop state,
and sends the PiCar-X `stop()` command. If the installed PiCar-X library exposes
`set_motor_speed`, the callback also sends explicit zero-speed commands to both
motor channels. The process then exits; restart it only after checking the cause
of the stop.

This GPIO button is still software-mediated. It supplements, but does not
replace, a physical power-disconnect switch that directly removes motor power.


## Safe First-Run Procedure

Use this order the first time you set up a car or after changing wiring, camera angle, lane tape, signs, or tuning values:

1. Put the car on a stable stand or blocks so the wheels cannot touch the floor.
2. Keep hands, hair, cables, and loose clothing away from the wheels and steering linkage.
3. Install dependencies and confirm the PiCar hardware library imports.
4. Run the vision-only dry run:

   ```bash
   python3 autonomous_picar_mvp.py --dry-run
   ```

5. Run the guided hardware check and press **Enter** only when the car is lifted and the area is clear:

   ```bash
   python3 hardware_check.py
   ```

6. Run the vision calibration tool and adjust lighting, lane tape, signs, and thresholds before driving:

   ```bash
   python3 calibrate_vision.py
   ```

7. Start autonomous driving at low speed in a clear test area. Stay close enough to press **CTRL+C** immediately.

## Dry-Run Mode

Dry-run mode runs the camera loop, lane detection, sign detection, and ultrasonic distance read when the PiCar library and sensor are available. It prints the motor and steering commands it would send, but it does not move the motors or steering servo.

If ultrasonic hardware is unavailable in dry-run mode, the script reports the
missing distance and continues the vision pipeline. The repeated-read fail-safe
stop applies only when the car is running with real motor control.

```bash
python3 autonomous_picar_mvp.py --dry-run
```

Use this before every classroom run to check that the camera sees the lane and signs and that the intended commands look reasonable. Press **q** in an OpenCV window or **CTRL+C** in the terminal to stop.

## Hardware Check Script

`hardware_check.py` is a guided, low-speed hardware test. It checks the camera and ultrasonic sensor, then asks for **Enter** before each steering or motor movement:

```bash
python3 hardware_check.py
```

Expected movement sequence:

1. Steering left.
2. Steering center.
3. Steering right.
4. Steering center.
5. Forward at very low speed.
6. Stop.

Keep the car on blocks for this test. If anything moves the wrong way, stop with **CTRL+C** and fix the hardware or library configuration before autonomous driving.

## Vision Calibration Script

`calibrate_vision.py` shows four OpenCV windows:

- `camera`: raw camera feed
- `lane mask`: white lane pixels used for lane following
- `red sign mask`: red pixels used for STOP detection
- `blue sign mask`: blue pixels used for RIGHT TURN detection

Run it with:

```bash
python3 calibrate_vision.py
```

The terminal prints `red_area`, `blue_area`, and `lane_error`. Use those values to tune sign size, lighting, camera angle, lane tape, and HSV thresholds. Press **q** in an OpenCV window or **CTRL+C** in the terminal to stop.

## What the Project Does

The car repeatedly performs this simple loop:

1. Read a camera frame.
2. Read the ultrasonic distance sensor.
3. Stop if an obstacle is too close.
4. Look for a red STOP sign or blue RIGHT TURN sign using color masks.
5. If no sign action is needed, find the lane center from white lane markings.
6. Steer left or right based on how far the lane center is from the camera center.

This is an MVP intended for classroom experiments and controlled test tracks. It is not a complete autonomous driving system.

## Hardware Required

Exact parts vary by kit, but the script expects hardware equivalent to:

- Raspberry Pi with Raspberry Pi OS
- PiCar or PiCar-X-style robot car chassis
- Motor driver and steering servo controlled by the PiCar library
- Camera supported by OpenCV, such as a Raspberry Pi Camera or USB webcam
- Ultrasonic distance sensor exposed through the PiCar library
- Battery pack suitable for the car motors and Raspberry Pi
- A safe test area with clear lane markings and simple colored signs

## Python Packages Required

The repository includes `requirements.txt` for the generic Python packages:

```text
opencv-python
numpy
```

The script also imports `picarx`, which is hardware-specific and is **not** listed in `requirements.txt` because it usually comes from the PiCar vendor's setup instructions rather than PyPI.

## Check Whether the PiCar Library Is Installed

Run this on the Raspberry Pi:

```bash
python3 -c "from picarx import Picarx; print('picarx is installed')"
```

If it succeeds, the script can import the hardware library. If it fails with `ModuleNotFoundError: No module named 'picarx'`, install the PiCar/PiCar-X software for your specific kit by following the vendor documentation.

You can also search installed packages:

```bash
python3 -m pip list | grep -i picar
```

## Install Dependencies

Recommended setup:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

On some Raspberry Pi systems, camera and OpenCV packages may be easier to install through the OS package manager. If `opencv-python` does not install cleanly, try your distribution's OpenCV package, for example:

```bash
sudo apt update
sudo apt install python3-opencv python3-numpy
```

Then verify imports:

```bash
python3 -c "import cv2, numpy; print('OpenCV and NumPy are ready')"
```

## Run the Script

Before running on the floor, place the car on a stand or blocks so the wheels can spin freely.

```bash
python3 autonomous_picar_mvp.py
```

For a no-movement safety preview, run:

```bash
python3 autonomous_picar_mvp.py --dry-run
```

The script opens two OpenCV windows when a display is available:

- `camera`: the raw camera view
- `lane mask`: the processed white-lane mask used for steering

The terminal prints distance, lane error, steering angle, and sign color-mask areas.

## Stop the Script Safely

Use one of these methods:

- Press **CTRL+C** in the terminal.
- Press **q** while an OpenCV window is focused.
- If the car behaves unexpectedly, lift it carefully from the chassis, not near the wheels.

The script uses a `finally` block to stop the motors, release the camera, and close OpenCV windows when it exits normally or through CTRL+C.

## Safety Notes

- Test with the wheels lifted first.
- Use low speed during testing.
- Keep hands, hair, cables, and loose clothing away from moving wheels.
- Always be ready to stop the program with **CTRL+C**.
- Test in an open area away from stairs, tables, people, and pets.
- Do not leave the car unattended while the script is running.

## Tuning Parameters

Configuration values are near the top of `autonomous_picar_mvp.py`.

| Parameter | What it controls | Tuning advice |
| --- | --- | --- |
| `CAMERA_ID` | Which camera OpenCV opens | Try `0`, then `1` if the wrong camera opens. |
| `BASE_SPEED` | Normal forward speed | Start low, such as `12` to `18`, then increase carefully. |
| `TURN_SPEED` | Speed during the scripted right turn | Keep lower than or close to `BASE_SPEED`. |
| `MAX_STEERING` | Maximum servo steering angle | Reduce if steering is too aggressive. |
| `KP` | Lane-following steering gain | Increase if corrections are too weak; decrease if the car oscillates. |
| `OBSTACLE_LIMIT_CM` | Stop distance for obstacles | Increase for more safety margin. |
| `STOP_COOLDOWN` | Seconds before another STOP sign response | Increase if the car repeatedly stops for the same sign. |
| `RIGHT_COOLDOWN` | Seconds before another RIGHT TURN response | Increase if the same sign triggers multiple turns. |
| `SIGN_AREA_THRESHOLD` | Minimum red/blue mask area for sign detection | Increase to reduce false detections; decrease if signs are missed. |

Tune one value at a time and write down the result. Lighting, camera angle, lane tape color, sign size, and speed all affect behavior.

## Known Limitations

- Uses simple color thresholding, not machine learning.
- Red or blue objects in the background can be mistaken for signs.
- Lane following assumes bright/white lane markings on a darker floor.
- The right turn is a timed maneuver, not a mapped path.
- Camera exposure, shadows, and glare can significantly change results.
- Ultrasonic readings can be noisy or unreliable on soft, angled, or narrow objects.
- The code assumes the `picarx` API provides `Picarx`, `forward`, `backward`,
  `stop`, `set_dir_servo_angle`, and `ultrasonic.read()`. If
  `set_motor_speed` is available, it is used as an additional stop command.

## Troubleshooting

### `ModuleNotFoundError: No module named 'picarx'`

The hardware library is missing. Install the PiCar/PiCar-X package for your exact kit, then rerun:

```bash
python3 -c "from picarx import Picarx; print('picarx is installed')"
```

### Camera does not open

- Check the camera cable or USB connection.
- Confirm the camera is enabled in Raspberry Pi configuration tools if using a Pi camera.
- Try changing `CAMERA_ID` from `0` to `1`.
- Test OpenCV directly:

```bash
python3 -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"
```

### The car drives the wrong direction or steering is reversed

Different kits may mount motors or servos differently. Check the PiCar library documentation and test motor/servo commands separately before running autonomous mode.

### Lane is not detected

- Improve lighting.
- Use brighter or wider lane tape.
- Aim the camera so the lower half of the image sees the lane.
- Adjust the white HSV threshold in `detect_lane()` if your lane marking is not white.

### STOP or RIGHT signs are missed

- Move signs closer to the camera.
- Make signs larger and more saturated.
- Improve lighting.
- Lower `SIGN_AREA_THRESHOLD` gradually.
- Adjust the HSV thresholds in `detect_sign()` for your printed signs.

### False sign detections happen

- Remove red and blue clutter from the background.
- Increase `SIGN_AREA_THRESHOLD`.
- Use a simpler test area with consistent lighting.

### Obstacle stopping is unreliable

- Increase `OBSTACLE_LIMIT_CM`.
- Test the ultrasonic sensor separately.
- Remember that ultrasonic sensors may miss angled or soft surfaces.
- Readings such as `-1` or `-2` are sensor/library error sentinels, not real
  negative distances. The autonomous script tolerates up to two transient
  invalid readings, but stops on the third consecutive invalid reading because
  obstacle detection can no longer be trusted. If failures repeat, check the
  ultrasonic sensor's power, trigger, and echo connections, then run
  `python3 hardware_check.py`.

## Live Web Dashboard

The optional Flask dashboard runs inside the same process as the camera and
robot control loop. **For the first test, put the car on stable blocks with the
wheels clear of the floor.** Start it at the intentionally low default speed:

```bash
python3 autonomous_picar_mvp.py --web
```

Then open this address from a computer or phone on the same network:

```text
http://<raspberry-pi-ip>:5000
```

The camera and processed lane mask are live immediately, but autonomous driving
starts **off**. The page shows distance, detected sign, lane error, steering,
autonomous state, emergency-stop state, and any active sensor/camera fault. Its
HSV sliders update the white lane threshold while the program is running.

- **Start autonomous driving** allows the existing loop to issue motor and
  steering commands.
- **Stop autonomous driving** immediately calls `stop()`, leaves driving
  disabled, and keeps the camera and dashboard running.
- **Emergency stop** immediately calls `stop()` and latches the stopped state.
  The browser cannot clear this latch; restart the program after checking the
  car and test area.

Use dry-run and the dashboard together before enabling real motors:

```bash
python3 autonomous_picar_mvp.py --dry-run --web
```

The dashboard is intentionally local and minimal: it has no authentication or
TLS, threshold changes are not saved after restart, and the sign detector and
lane follower still use simple color thresholds. Do not expose port 5000 to the
public internet. A browser button and GPIO callback are software safety aids,
not replacements for a physical motor-power disconnect. If camera capture
fails, the program disables autonomous driving and latches the emergency state;
if the main loop or web server exits, the cleanup path calls `stop()`.

## Optional Voice Control

Voice control adds two deliberately limited commands:

- `picar start`
- `picar stop`

The word `picar` is required as a wake word. Other phrases are ignored, and
there is no voice command that clears or operates the emergency-stop latch.
Voice start is rejected while an emergency stop or active safety fault exists.
Voice stop is always allowed and immediately calls the motor stop function.
When voice mode starts, autonomous driving remains **off** until a valid start
command is heard.

Install the local speaker command and microphone dependencies on Raspberry Pi
OS:

```bash
sudo apt update
sudo apt install espeak
sudo apt install python3-pyaudio portaudio19-dev
python3 -m pip install SpeechRecognition pyaudio
```

`espeak` provides non-blocking local spoken feedback such as `Starting` and
`Stopping`. If it is unavailable, the same feedback is printed instead. Speech
recognition uses the `SpeechRecognition` package's default Google recognizer,
so recognizing commands requires an internet connection. No cloud
text-to-speech is used.

List microphone names and indexes before selecting an input:

```bash
python3 autonomous_picar_mvp.py --list-mics
```

Run the dashboard and voice listener together:

```bash
python3 autonomous_picar_mvp.py --web --voice
```

Select a microphone by the index printed by `--list-mics`:

```bash
python3 autonomous_picar_mvp.py --web --voice --voice-device-index 1
```

For the first voice test, put the car on stable blocks with all wheels clear of
the floor and use dry-run mode:

```bash
python3 autonomous_picar_mvp.py --dry-run --web --voice
```

In dry-run mode, voice commands, dashboard state changes, and speaker feedback
still work, while motor and steering commands are only printed. Recognition can
be affected by room noise, microphone quality, pronunciation, and network
availability. Voice control is a convenience feature, not a replacement for
the web/GPIO stop controls or a physical motor-power disconnect.
