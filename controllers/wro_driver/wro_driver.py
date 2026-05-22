"""
WRO Future Engineers – Webots Controller
=========================================
Wall-follower using Lidar and IMU with manual Ackermann calculation.
Includes real-time SLAM occupancy grid mapping with live browser viewer.
"""

import math
import time
import numpy as np
import cv2
from controller import Robot
from estimation import OpenCVLocalizer

# --- configuration ---
TIME_STEP = 32            # ms
TARGET_SPEED = 5.0       # rad/s (speed of the wheels)
WHEELBASE = 0.14           # m
TRACK_FRONT = 0.12        # m
WHEEL_RADIUS = 0.03       # m (tireRadius from .wbt)

# --- PID configuration ---
P_GAIN = 1.5
D_GAIN = 3.0
MAX_STEERING = 0.9        # rad

# --- Ackermann helper ---
def ackermann_angles(target_angle):
    if abs(target_angle) < 1e-6:
        return 0.0, 0.0
    R = WHEELBASE / math.tan(abs(target_angle))
    inner = math.atan(WHEELBASE / (R - TRACK_FRONT / 2.0))
    outer = math.atan(WHEELBASE / (R + TRACK_FRONT / 2.0))
    
    # In Webots, a positive steering angle turns the wheels to the RIGHT.
    if target_angle > 0: 
        # Right turn: Right wheel is inner (turns sharper), Left wheel is outer
        return outer, inner
    else: 
        # target_angle < 0 means a LEFT turn.
        # Left turn: Left wheel is inner (turns sharper), Right wheel is outer
        return -inner, -outer

# --- initialise robot ---
robot = Robot()

# Camera
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

# Lidar
lidar = robot.getDevice("lidar")
lidar.enable(TIME_STEP)

# IMU
imu = robot.getDevice("imu")
imu.enable(TIME_STEP)

# Motors
motor_right = robot.getDevice("motor_rear_right")
motor_left  = robot.getDevice("motor_rear_left")
motor_right.setPosition(float('inf'))
motor_left.setPosition(float('inf'))

steer_left  = robot.getDevice("left_steer")
steer_right = robot.getDevice("right_steer")

def set_steering_angle(target_angle):
    left, right = ackermann_angles(target_angle)
    steer_left.setPosition(left)
    steer_right.setPosition(right)

# --- initialise OpenCV localizer ---
localizer = OpenCVLocalizer()

# --- pose estimation state ---
# Start exactly in the middle of the South section in strictly positive space (X=1.5, Y=0.5)
robot_x = 1.5
robot_y = 0.5
robot_yaw = 0.0

state = "WAIT_FOR_LIDAR"
ref_img = localizer.create_arena_reference_image()
collected_scans = []

# --- main loop ---
prev_error = 0.0
step_count = 0

while robot.step(TIME_STEP) != -1:
    step_count += 1
    lidar_data = lidar.getRangeImage()
    try:
        cv2.waitKey(1)
    except Exception:
        pass
    
    match state:
        case "WAIT_FOR_LIDAR":
            motor_left.setVelocity(0.0)
            motor_right.setVelocity(0.0)
            steer_left.setPosition(0.0)
            steer_right.setPosition(0.0)
            
            if lidar_data and len(lidar_data) > 0:
                state = "CALCULATE_INITIAL_POSITION"
                
        case "CALCULATE_INITIAL_POSITION":
            motor_left.setVelocity(0.0)
            motor_right.setVelocity(0.0)
            steer_left.setPosition(0.0)
            steer_right.setPosition(0.0)
            
            if lidar_data and len(lidar_data) > 0:
                collected_scans.append(lidar_data)
                if len(collected_scans) >= 10:
                    scans_arr = np.array(collected_scans)
                    # Filter out invalid values by marking them as NaN
                    invalid_mask = (scans_arr <= 0.01) | (scans_arr >= 2.0) | np.isinf(scans_arr) | np.isnan(scans_arr)
                    scans_arr[invalid_mask] = np.nan
                    
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        avg_ranges = np.nanmean(scans_arr, axis=0)
                    avg_ranges = np.nan_to_num(avg_ranges, nan=0.0)

                    tpl_img = localizer.project_lidar_to_template(
                        ranges=avg_ranges,
                        yaw=0.0
                    )
                    try:
                        cv2.imshow("Calibration Reference Map", ref_img)
                        cv2.imshow("Calibration LiDAR Template", tpl_img)
                        print("[Calibration] Displaying Reference and Template. Press any key in the CV window to start driving...")
                        cv2.waitKey(0)
                    except Exception as e:
                        print(f"[Calibration] OpenCV window error: {e}")
                    finally:
                        for win_name in ["Calibration Reference Map", "Calibration LiDAR Template"]:
                            try:
                                cv2.destroyWindow(win_name)
                            except Exception:
                                pass
                    
                    state = "DRIVE"
                    
        case "DRIVE":
            # --- Stage 1: Warmup (0s - 1s) ---
            # Stay still
            if robot.getTime() < 1.0:
                motor_left.setVelocity(0.0)
                motor_right.setVelocity(0.0)
                steer_left.setPosition(0.0)
                steer_right.setPosition(0.0)
            else:
                # --- Stage 2: Estimation Update ---
                if lidar_data:
                    robot_x, robot_y, robot_yaw = localizer.update(
                        lidar_ranges=lidar_data,
                        max_range=2.0,
                        angle_offset=math.pi / 2 # Rotated by 180 degrees
                    )

                # --- Stage 3: Driving ---
                speed = TARGET_SPEED
                steering_angle = 0.0

                if lidar_data:
                    # Helper to ignore values too far away (e.g. open track sections)
                    def cap(dist):
                        return min(dist, 1.5)

                    # Distances: 0=Back, 180=Front
                    # With angle_offset=pi and negative angle_inc:
                    # 90 is +90 deg (Left), 270 is -90 deg (Right)
                    # 135 is +45 deg (Front-Left), 225 is -45 deg (Front-Right)
                    right_dist = cap(lidar_data[270])
                    left_dist = cap(lidar_data[90])
                    front_dist = lidar_data[180]
                    front_right_dist = cap(lidar_data[225])
                    front_left_dist = cap(lidar_data[135])
                    
                    # 1. Centering and Parallel Driving
                    # P_GAIN keeps it centered, D_GAIN keeps it parallel
                    current_diff = left_dist - right_dist
                    derivative = current_diff - prev_error
                    
                    # Invert: More room on Left -> Steer LEFT (negative)
                    raw_steering = -(P_GAIN * current_diff + D_GAIN * derivative)
                    
                    # 2. Side Wall Safety (Minimum distance 10cm)
                    min_side = min(left_dist, right_dist)
                    if min_side < 0.2:
                        safety_force = (0.2 - min_side) * 5.0 
                        if left_dist < right_dist:
                            raw_steering += safety_force  # Push away from LEFT (steer Right)
                        else:
                            raw_steering -= safety_force  # Push away from RIGHT (steer Left)
                    
                    # 3. Cornering (Front Wall Avoidance)
                    if front_dist < 1.0:
                        corner_force = (1.0 - front_dist)**2
                        CORNER_GAIN = 8.0 
                        
                        # Use diagonals to decide steering (adaptive)
                        if front_left_dist > front_right_dist:
                            raw_steering -= (corner_force * CORNER_GAIN)   # Steer LEFT (negative)
                        else:
                            raw_steering += (corner_force * CORNER_GAIN)   # Steer RIGHT (positive)
                    
                    # Clamp raw steering angle
                    raw_steering = max(-MAX_STEERING, min(MAX_STEERING, raw_steering))
                    
                    # 4. Low-Pass Filter (Smooths out sudden spikes)
                    steering_angle = steering_angle + 0.5 * (raw_steering - steering_angle)
                    
                    prev_error = current_diff
                    
                    # 5. Dynamic Speed Control
                    speed_factor = 1.0 - (abs(steering_angle) / MAX_STEERING) * 0.3
                    speed = TARGET_SPEED * max(0.7, speed_factor)

                    # --- Live OpenCV Visualization ---
                    vis_img = localizer.render()
                    try:
                        cv2.imshow("WRO Localization", vis_img)
                        cv2.waitKey(1)
                    except Exception:
                        pass

                # Drive rear wheels
                motor_right.setVelocity(speed)
                motor_left.setVelocity(speed)

                # Apply steering
                set_steering_angle(steering_angle)
