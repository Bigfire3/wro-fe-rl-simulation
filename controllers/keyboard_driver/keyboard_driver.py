import sys
import os
import math
import numpy as np

# Setup path to import wro_core
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Potential paths to site-packages depending on OS
site_package_paths = [
    os.path.join(project_root, ".venv", "Lib", "site-packages"),
]
# For Unix systems, find the python subdirectories in .venv/lib
venv_lib_dir = os.path.join(project_root, ".venv", "lib")
if os.path.exists(venv_lib_dir):
    for sub in os.listdir(venv_lib_dir):
        if sub.startswith("python"):
            site_package_paths.append(os.path.join(venv_lib_dir, sub, "site-packages"))

for path in site_package_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# Inject wro_driver folder to sys.path to allow importing wro_core
wro_driver_dir = os.path.join(project_root, "controllers", "wro_driver")
if wro_driver_dir not in sys.path:
    sys.path.append(wro_driver_dir)

from controller import Supervisor, Keyboard
from wro_core import config, control

def main():
    print("=" * 60)
    print("WRO FUTURE ENGINEERS - KEYBOARD CONTROL DRIVER")
    print("=" * 60)
    print("Controls:")
    print("  [Arrow UP]    : Accelerate / Move Forward")
    print("  [Arrow DOWN]  : Decelerate / Move Backward / Reverse")
    print("  [Arrow LEFT]  : Steer Left")
    print("  [Arrow RIGHT] : Steer Right")
    print("  [Space]       : Emergency Stop / Reset to 0")
    print("-" * 60)
    print("Click on the Webots 3D view window to focus keyboard input.")
    
    robot = Supervisor()
    
    # Initialize Keyboard
    keyboard = robot.getKeyboard()
    keyboard.enable(config.TIME_STEP)
    
    # Initialize Devices
    motor_right = robot.getDevice("motor_rear_right")
    motor_left  = robot.getDevice("motor_rear_left")
    motor_right.setPosition(float('inf'))
    motor_left.setPosition(float('inf'))
    
    steer_left  = robot.getDevice("left_steer")
    steer_right = robot.getDevice("right_steer")
    
    # Controller
    car_controller = control.Controller()
    
    # Target speed and steering values
    target_speed = 0.0
    target_steering = 0.0
    
    # Constants for control response
    STEER_STEP = 0.08      # steering angle increment per step (rad)
    SPEED_STEP = 2.0       # motor speed increment per step (rad/s)
    
    DECEL_RATE = 0.85      # deceleration factor when no drive key is pressed
    STEER_CENTER_RATE = 0.75 # centering factor when no steering key is pressed

    while robot.step(config.TIME_STEP) != -1:
        key = keyboard.getKey()
        
        # Flags for current step inputs
        up_pressed = False
        down_pressed = False
        left_pressed = False
        right_pressed = False
        space_pressed = False
        
        # Read all buffered keys in this time step
        while key != -1:
            if key == Keyboard.UP:
                up_pressed = True
            elif key == Keyboard.DOWN:
                down_pressed = True
            elif key == Keyboard.LEFT:
                left_pressed = True
            elif key == Keyboard.RIGHT:
                right_pressed = True
            elif key == ord(' '):
                space_pressed = True
            key = keyboard.getKey()
            
          # Determine steering changes:
          # In control.py: positive target_angle turns right, negative turns left.
          # So left_pressed decreases the angle (steers left),
          # and right_pressed increases the angle (steers right).
        if space_pressed:
            target_speed = 0.0
            target_steering = 0.0
            print("[Keyboard] Emergency Stop & Reset!")
        else:
            # Process Speed
            if up_pressed:
                target_speed = min(target_speed + SPEED_STEP, config.MAX_MOTOR_VELOCITY)
            elif down_pressed:
                target_speed = max(target_speed - SPEED_STEP, -config.MAX_MOTOR_VELOCITY)
            else:
                # Natural deceleration
                if abs(target_speed) < 0.2:
                    target_speed = 0.0
                else:
                    target_speed *= DECEL_RATE
                    
            # Process Steering
            if left_pressed:
                target_steering = max(target_steering - STEER_STEP, -config.MAX_STEERING)
            elif right_pressed:
                target_steering = min(target_steering + STEER_STEP, config.MAX_STEERING)
            else:
                # Return steering to center
                if abs(target_steering) < 0.02:
                    target_steering = 0.0
                else:
                    target_steering *= STEER_CENTER_RATE

        # Log current state periodically
        # Print update every ~500ms
        if int(robot.getTime() * 1000) % 500 < config.TIME_STEP:
            print(f"Time: {robot.getTime():.2f}s | Target Speed: {target_speed:5.1f} rad/s | Target Steering: {target_steering:+5.2f} rad", end='\r')

        # Apply using the standard Controller class (electronic differential & Ackermann steering)
        car_controller.apply(
            target_speed=target_speed,
            target_steering=target_steering,
            motor_left=motor_left,
            motor_right=motor_right,
            steer_left=steer_left,
            steer_right=steer_right,
            use_rl=True # Bypass speed scaling in rules mode so we get exact speed set by keyboard
        )

if __name__ == "__main__":
    main()
