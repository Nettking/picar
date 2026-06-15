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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -c "from picarx import Picarx; print('picarx is installed')"
python3 autonomous_picar_mvp.py
```

Stop the program at any time with **CTRL+C**. If a display window is open, you can also press **q** while the OpenCV window is focused.

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
python3 -m venv .venv
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
- The code assumes the `picarx` API provides `Picarx`, `forward`, `backward`, `stop`, `set_dir_servo_angle`, and `ultrasonic.read()`.

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
