"""
WRO Future Engineers – Webots Controller
=========================================
- Stage 1: Perception
- Stage 2: Estimation
- Stage 3: Planning (RL)
- Stage 4: Control
"""

import sys
import os

# Try to find and inject the virtual environment's site-packages path to allow loading ONNX and other libraries
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

# Inject current folder to sys.path
if script_dir not in sys.path:
    sys.path.append(script_dir)

import cv2
import numpy as np
import warnings
# pyrefly: ignore [missing-import]
from controller import Supervisor

from wro_core import config, perception, estimation, planning, control
from wro_core.obstacle_randomizer import randomize_obstacles
from wro_core.obs_visualizer import draw_observation_window

# --- Load ONNX Model ---
model_path = os.path.join(project_root, "models", "wro_model.onnx")

rl_planner = planning.RLPlanner(model_path)
if not rl_planner.is_ready():
    raise RuntimeError(f"ONNX model could not be loaded from {model_path}. RL Planner is not ready.")

# --- initialise robot ---
robot = Supervisor()
robot_node = robot.getSelf()

# Devices
camera = robot.getDevice("camera")
camera.enable(config.TIME_STEP)

lidar = robot.getDevice("lidar")
lidar.enable(config.TIME_STEP)

imu = robot.getDevice("imu")
imu.enable(config.TIME_STEP)

motor_right = robot.getDevice("motor_rear_right")
motor_left  = robot.getDevice("motor_rear_left")
motor_right.setPosition(float('inf'))
motor_left.setPosition(float('inf'))

# Get actual max velocity from the motor device to dynamically scale speed limit
max_motor_vel = motor_left.getMaxVelocity()
config.MAX_MOTOR_VELOCITY = max_motor_vel
max_linear_velocity = max_motor_vel * config.WHEEL_RADIUS
config.NORM_FACTORS[0] = 1.0 / max_linear_velocity

steer_left  = robot.getDevice("left_steer")
steer_right = robot.getDevice("right_steer")

# --- Initialize estimators and controller ---
estimator = estimation.StateEstimator()
car_controller = control.Controller()

# --- State variables ---
imu_yaw_initial = None


# Reset obstacles
randomize_obstacles(robot, train=False)

# --- Action Repeat configuration ---
frame_skip = int(1000 / (config.CONTROL_FREQ * config.TIME_STEP))
if frame_skip < 1:
    frame_skip = 1
calibration_step_count = None

# --- MAIN LOOP ---
step_count = 0

while robot.step(config.TIME_STEP) != -1:
    step_count += 1
    
    # Avoid OpenCV hang-ups
    try:
        cv2.waitKey(1)
    except Exception:
        pass
        
    # Skip planning, estimation, and control updates between control steps
    # to maintain a consistent control frequency (e.g. 10 Hz / every 100ms)
    if estimator.initial_pose_found:
        if calibration_step_count is None:
            calibration_step_count = step_count
            
        if (step_count - calibration_step_count) % frame_skip != 0:
            continue
        
    # --- 1. STAGE 1: PERCEPTION ---
    sensor_data, imu_yaw_initial = perception.read_sensors(lidar, imu, camera, imu_yaw_initial)
    
    # --- 2. STAGE 2: ESTIMATION ---
    if not estimator.initial_pose_found:
        lidar_ranges = sensor_data["lidar_ranges"]
        try:
            res = estimator.add_calibration_scan(lidar_ranges)
            if res is not None:
                x_init, y_init, yaw_init, direction, debug_img = res
                cv2.imshow("Calibration Result", debug_img)
                print(f"[Calibration] Best candidate found at: x={x_init:.3f}m, y={y_init:.3f}m, yaw={yaw_init:.3f} rad")
                print(f"[Calibration] Resolved driving direction: {direction}")
                cv2.waitKey(1)
                try:
                    cv2.destroyWindow("Calibration Result")
                except Exception:
                    pass
        except Exception as e:
            print(f"[Calibration] Error during calibration: {e}")
            # Fallback
            estimator.set_calibrated_pose(1.5, 0.5, 0.0, "CCW")
            
        if not estimator.initial_pose_found:
            # If not calibrated yet, command zero speed
            motor_right.setVelocity(0.0)
            motor_left.setVelocity(0.0)
            continue

    # Get ego velocity directly from Webots physics
    vel = robot_node.getVelocity()
    if vel is not None:
        v_g = vel[:3]
        R = robot_node.getOrientation()
        if R is not None:
            # Local X axis is [R[0], R[3], R[6]]
            ego_v_x = float(v_g[0] * R[0] + v_g[1] * R[3] + v_g[2] * R[6])
        else:
            ego_v_x = 0.0
    else:
        ego_v_x = 0.0
        
    sensor_data["ego_v_x"] = ego_v_x

    # Calibration is completed, proceed with standard control loop
    robot_pose = estimator.update(sensor_data)
    
    result = planning.compute_observation_vector(
        pose=robot_pose,
        obstacles=estimator.obstacle_mapper.obstacles,
        driving_direction=estimator.driving_direction,
        last_steering=car_controller.last_steering,
        ego_v_x=ego_v_x
    )
    raw_obs, obs_vector = result[0], result[1]
    best_closest = result[2]
    p_inner_40, p_inner_100 = result[5], result[6]
    p_outer_40, p_outer_100 = result[7], result[8]
    p_corner = result[9]
    
    # --- 3. STAGE 3: PLANNING ---
    target_speed, target_steering = rl_planner.plan(obs_vector)
        
    # --- 4. STAGE 4: CONTROL ---
    actual_speed, actual_steering = car_controller.apply(
        target_speed=target_speed,
        target_steering=target_steering,
        motor_left=motor_left,
        motor_right=motor_right,
        steer_left=steer_left,
        steer_right=steer_right
    )
    
    # --- Live OpenCV Visualisierung ---
    if len(sensor_data["lidar_ranges"]) > 0:
        vis_img = estimator.icp_localizer.render()
        vis_img = estimator.obstacle_mapper.render(vis_img, robot_pose, estimator.icp_localizer.scale, estimator.icp_localizer.window_size)
        
        if "camera_image" in sensor_data:
            cam_debug = estimator.obstacle_mapper.render_camera(sensor_data["camera_image"], robot_pose)
            try:
                cv2.imshow("WRO Camera Debug", cam_debug)
            except Exception:
                pass
                
        if raw_obs is not None:
            draw_observation_window(
                pose=robot_pose,
                raw_obs=raw_obs,
                obs_vector=obs_vector,
                driving_direction=estimator.driving_direction,
                best_closest=best_closest,
                p_inner_40=p_inner_40,
                p_inner_100=p_inner_100,
                p_outer_40=p_outer_40,
                p_outer_100=p_outer_100,
                p_corner=p_corner,
                window_name="WRO Observation Debug"
            )
        try:
            cv2.imshow("WRO Localization", vis_img)
        except Exception:
            pass
