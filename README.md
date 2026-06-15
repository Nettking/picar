# Autonomous PiCar MVP

A minimum viable autonomous driving system for a Raspberry Pi based PiCar platform.

The vehicle uses computer vision and ultrasonic sensing to:

- Follow lane markings
- Maintain lane position
- Detect and react to traffic signs
- Avoid collisions
- Navigate a simple autonomous driving course

## Features

### Lane Following

The camera continuously detects lane markings and calculates the vehicle's offset from the center of the lane.

A proportional controller adjusts the steering angle to keep the vehicle centered.

### Traffic Sign Recognition

The system currently supports:

| Sign | Action |
|--------|--------|
| STOP | Stop for 2 seconds |
| RIGHT TURN | Perform a predefined right turn |

Traffic signs are detected using color segmentation in HSV color space.

### Obstacle Detection

The ultrasonic sensor monitors the distance in front of the vehicle.

If an obstacle is detected within the safety threshold, the vehicle stops immediately.

## Hardware

- Raspberry Pi
- PiCar chassis
- Camera
- Ultrasonic sensor
- Servo
- Motors

## Running

```bash
python3 autonomous_picar_mvp.py 