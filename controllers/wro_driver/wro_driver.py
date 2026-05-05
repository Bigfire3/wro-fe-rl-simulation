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
from slam import KnownMapLocalizer

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

# --- SLAM configuration ---
SLAM_GRID_SIZE = 300       # cells (300×300 → 3m×3m at 1cm resolution)
SLAM_RESOLUTION = 0.01    # metres per cell (1cm → 300 cells = 3m = arena size)
SLAM_UPDATE_INTERVAL = 3  # update SLAM every N controller steps
MAP_PUSH_INTERVAL = 10    # push map to browser every N controller steps

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

# --- initialise known-map localizer ---
localizer = KnownMapLocalizer(
    grid_size=SLAM_GRID_SIZE,
    resolution=SLAM_RESOLUTION,
)



# --- pose estimation state ---
# Start exactly in the middle of the North section, driving clockwise (East)
robot_x = 0.0
robot_y = 1.0
robot_yaw = 0.0
imu_offset = None  # Will be calibrated on first step
current_speed_mps = 0.0   # metres per second
prev_imu_yaw = None      # for relative rotation tracking

# --- main loop ---
prev_error = 0.0
step_count = 0

while robot.step(TIME_STEP) != -1:
    step_count += 1
    lidar_data = lidar.getRangeImage()
    
    # --- IMU Yaw Fusion ---
    imu_values = imu.getRollPitchYaw()
    raw_yaw = imu_values[2]
    if imu_offset is None:
        # Calibrate IMU offset so that initial direction is exactly robot_yaw (0.0 = East)
        imu_offset = robot_yaw - raw_yaw
    
    current_imu_yaw = raw_yaw + imu_offset
    
    speed = TARGET_SPEED
    steering_angle = 0.0
    
    # Estimate linear velocity from motor speed command
    # Average of left/right rear wheel velocities × wheel radius
    current_speed_mps = speed * WHEEL_RADIUS  # simplified: both wheels same speed

    dt = TIME_STEP / 1000.0  # seconds
    # --- Dead Reckoning (Fused) ---
    # We use the IMU to track relative rotation, but let the localizer fix absolute drift.
    if prev_imu_yaw is not None:
        yaw_diff = current_imu_yaw - prev_imu_yaw
        # Wrap to [-pi, pi]
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        robot_yaw += yaw_diff
        # Normalize to [-pi, pi]
        robot_yaw = (robot_yaw + math.pi) % (2 * math.pi) - math.pi
    
    prev_imu_yaw = current_imu_yaw

    robot_x += current_speed_mps * dt * math.cos(robot_yaw)
    robot_y += current_speed_mps * dt * math.sin(robot_yaw)

    if lidar_data:
        # Helper to ignore values too far away (e.g. open track sections)
        def cap(dist):
            return min(dist, 1.5)

        # Distances: 90=Right, 270=Left, 180=Front, 135=Front-Right, 225=Front-Left
        right_dist = cap(lidar_data[90])
        left_dist = cap(lidar_data[270])
        front_dist = lidar_data[180]
        front_right_dist = cap(lidar_data[135])
        front_left_dist = cap(lidar_data[225])
        
        # 1. Centering and Parallel Driving
        # P_GAIN keeps it centered, D_GAIN keeps it parallel
        current_diff = left_dist - right_dist
        derivative = current_diff - prev_error
        
        raw_steering = P_GAIN * current_diff + D_GAIN * derivative
        
        # 2. Side Wall Safety (Minimum distance 10cm)
        # Instead of a hard snap, we add a force proportional to the threshold violation.
        min_side = min(left_dist, right_dist)
        if min_side < 0.2:
            safety_force = (0.2 - min_side) * 5.0  # Stronger the closer we get
            if left_dist < right_dist:
                raw_steering -= safety_force  # Smoothly push away from left
            else:
                raw_steering += safety_force  # Smoothly push away from right
        
        # 3. Cornering (Front Wall Avoidance)
        # Calculate force based on (1 - distance)^2 when closer than 1.0m
        if front_dist < 1.0:
            corner_force = (1.0 - front_dist)**2
            
            # Multiply by a factor so the steering angle is strong enough to actually make the turn
            CORNER_GAIN = 4.0 
            
            # Use diagonals to decide direction (steer towards the larger opening)
            if front_left_dist > front_right_dist:
                raw_steering += (corner_force * CORNER_GAIN)   # Steer left
            else:
                raw_steering -= (corner_force * CORNER_GAIN)   # Steer right
        
        # Clamp raw steering angle
        raw_steering = max(-MAX_STEERING, min(MAX_STEERING, raw_steering))
        
        # 4. Low-Pass Filter (Smooths out sudden spikes)
        # 0.5 offers a good balance: smooths out straight-line twitching 
        # but is still fast enough to exit corners cleanly.
        steering_angle = steering_angle + 0.5 * (raw_steering - steering_angle)
        
        prev_error = current_diff
        
        # 5. Dynamic Speed Control
        # Speed smoothly decreases as steering angle increases, but less aggressively now.
        # At 0 steering, speed is 100%. At MAX_STEERING, speed is 70%.
        speed_factor = 1.0 - (abs(steering_angle) / MAX_STEERING) * 0.3
        speed = TARGET_SPEED * max(0.7, speed_factor)

        # --- Localization update (corrects position using known map and fuses IMU) ---
        if step_count % SLAM_UPDATE_INTERVAL == 0:
            robot_x, robot_y, robot_yaw = localizer.update(
                lidar_ranges=lidar_data,
                robot_x=robot_x,
                robot_y=robot_y,
                robot_yaw=robot_yaw,
                imu_yaw=current_imu_yaw,
                max_range=3.0,
                angle_offset=math.pi,  # Rotated by 180 degrees
            )
            # --- Live OpenCV Visualization ---
            vis_img = localizer.render(window_size=600)
            cv2.imshow("WRO Localization", vis_img)
            cv2.waitKey(1)



    # Drive rear wheels
    motor_right.setVelocity(speed)
    motor_left.setVelocity(speed)

    # Apply steering
    set_steering_angle(steering_angle)
