"""
WRO Future Engineers – Webots Controller
=========================================
Wall-follower using Lidar and IMU with manual Ackermann calculation.
Includes real-time SLAM occupancy grid mapping with live browser viewer.
Refactored to follow a strict 4-Stage Software Pipeline:
- Stage 1: Perception
- Stage 2: Estimation
- Stage 3: Planning (Hierarchical State Machine)
- Stage 4: Control
"""

import math
import time
import warnings
import numpy as np
import cv2
# pyrefly: ignore [missing-import]
from controller import Robot
from opencv_localizer import OpenCVLocalizer
from trans_icp_localizer import TranslationICPLocalizer
from obstacle_mapper import ObstacleMapper

# --- configuration ---
TIME_STEP = 32            # ms
TARGET_SPEED = 5.0        # rad/s (speed of the wheels)
WHEELBASE = 0.14          # m
TRACK_FRONT = 0.12        # m
WHEEL_RADIUS = 0.03       # m (tireRadius from .wbt)
USE_RL = True             # Enable Reinforcement Learning navigation

# --- PID configuration ---
P_GAIN = 1.5
D_GAIN = 3.0
MAX_STEERING = 0.8        # rad

# --- Load ONNX Model ---
ort_session = None
if USE_RL:
    try:
        import onnxruntime as ort
        import os
        
        # Look for wro_model.onnx in the same folder as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "wro_model.onnx")
        
        if os.path.exists(model_path):
            print(f"[RL] Loading ONNX model from {model_path}...")
            ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            print("[RL] ONNX model loaded successfully!")
        else:
            print(f"[RL Warning] Model file not found at {model_path}. Falling back to Rules-based.")
            USE_RL = False
    except Exception as e:
        print(f"[RL Warning] Failed to load ONNX runtime/model: {e}. Falling back to Rules-based.")
        USE_RL = False

# --- Helper ---
def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))

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
    # Begrenzung auf die physikalischen Grenzen der Webots-Servos mit MAX_STEERING
    left = clamp(left, -MAX_STEERING, MAX_STEERING)
    right = clamp(right, -MAX_STEERING, MAX_STEERING)
    steer_left.setPosition(left)
    steer_right.setPosition(right)

# --- initialise OpenCV localizers ---
localizer = OpenCVLocalizer()
icp_localizer = TranslationICPLocalizer()
obstacle_mapper = ObstacleMapper()

# --- state variables ---
robot_x = 1.5
robot_y = 0.5
robot_yaw = 0.0

state = "STOP"
initial_pose_found = False
driving_direction = None
collected_scans = []
prev_error = 0.0
smoothed_steering = 0.0
imu_yaw_initial = None  # Erster IMU-Wert zum Nullen (wie echte IMU)

# Generate reference image for calibration
ref_img = localizer.create_arena_reference_image()

# --- STAGE 1: PERCEPTION ---
def run_perception(robot, lidar, imu, camera):
    """
    Stage 1: Wahrnehmung.
    Liest Sensordaten aus und bereitet sie in einem standardisierten Format auf.
    Die IMU wird beim ersten Aufruf genullt (wie eine echte IMU).
    Enthält KEINE Zustandslogik.
    """
    global imu_yaw_initial
    
    lidar_data = lidar.getRangeImage()
    if lidar_data is None:
        lidar_data = []
        
    try:
        rpy = imu.getRollPitchYaw()
        imu_yaw_raw = rpy[2] if rpy else 0.0
    except Exception:
        imu_yaw_raw = 0.0
    
    # IMU nullen: Ersten Wert speichern, danach nur noch Änderungen ausgeben
    if imu_yaw_initial is None:
        imu_yaw_initial = imu_yaw_raw
    imu_yaw = imu_yaw_raw - imu_yaw_initial
    
    # Kamera auslesen und konvertieren
    img_bgr = None
    try:
        w = camera.getWidth()
        h = camera.getHeight()
        img_buffer = camera.getImage()
        if img_buffer:
            img_raw = np.frombuffer(img_buffer, dtype=np.uint8).reshape((h, w, 4))
            img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        print(f"[Perception] Error reading camera: {e}")
        
    sensor_dict = {
        "lidar_ranges": lidar_data,
        "imu_yaw": imu_yaw
    }
    if img_bgr is not None:
        sensor_dict["camera_image"] = img_bgr
        
    return sensor_dict

# --- STAGE 2: ESTIMATION ---
def run_estimation(localizer, icp_localizer, obstacle_mapper, sensor_data):
    """
    Stage 2: Zustandsschätzung.
    Aktualisiert die geschätzte Roboterpose (x, y, yaw) und das Belegungsgitter.
    Enthält KEINE Zustandslogik der State Machine, führt aber die einmalige
    Initial-Lokalisierung (Kalibrierung) durch.
    """
    global robot_x, robot_y, robot_yaw, initial_pose_found, driving_direction, collected_scans
    
    lidar_ranges = sensor_data["lidar_ranges"]
    if len(lidar_ranges) > 0:
        if not initial_pose_found:
            collected_scans.append(lidar_ranges)
            if len(collected_scans) >= 10:
                scans_arr = np.array(collected_scans)
                # Ungültige Werte ausfiltern
                invalid_mask = (scans_arr <= 0.01) | (scans_arr >= 2.0) | np.isinf(scans_arr) | np.isnan(scans_arr)
                scans_arr[invalid_mask] = np.nan
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_ranges = np.nanmean(scans_arr, axis=0)
                avg_ranges = np.nan_to_num(avg_ranges, nan=0.0)

                # Kalibrierung über das Template Matching durchführen
                try:
                    # n_rays und Winkelberechnung
                    n_rays = len(avg_ranges)
                    angle_inc = -2.0 * math.pi / n_rays if n_rays > 0 else 0.0
                    angle_offset = math.pi / 2 # 90 Grad Rotation entsprechend update()
                    
                    x_init, y_init, yaw_init, direction, debug_img = localizer.calibrate_initial_pose(
                        avg_ranges=avg_ranges,
                        angle_offset=angle_offset,
                        angle_inc=angle_inc
                    )
                    
                    # Ergebnis anzeigen und blockieren
                    cv2.imshow("Calibration Result", debug_img)
                    print(f"[Calibration] Best candidate found at: x={x_init:.3f}m, y={y_init:.3f}m, yaw={yaw_init:.3f} rad ({math.degrees(yaw_init):.1f}°)")
                    print(f"[Calibration] Resolved driving direction: {direction}")
                    print("[Calibration] Starting in 0.001 seconds...")
                    cv2.waitKey(1)
                    
                    # Initial-Pose setzen
                    localizer.set_initial_pose(x_init, y_init, yaw_init)
                    icp_localizer.set_initial_pose(x_init, y_init, yaw_init)
                    robot_x, robot_y, robot_yaw = x_init, y_init, yaw_init
                    
                    initial_pose_found = True
                    driving_direction = direction
                    
                except Exception as e:
                    print(f"[Calibration] Error during calibration: {e}")
                    # Fallback auf Standardwerte bei Fehler
                    localizer.set_initial_pose(1.5, 0.5, 0.0)
                    icp_localizer.set_initial_pose(1.5, 0.5, 0.0)
                    robot_x, robot_y, robot_yaw = 1.5, 0.5, 0.0
                    initial_pose_found = True
                    driving_direction = "CCW"
                finally:
                    try:
                        cv2.destroyWindow("Calibration Result")
                    except Exception:
                        pass
        else:
            robot_x, robot_y, robot_yaw, outliers = icp_localizer.update(
                lidar_ranges=lidar_ranges,
                imu_yaw=sensor_data["imu_yaw"],
                max_range=2.0
            )
            obstacle_mapper.update([robot_x, robot_y, robot_yaw], outliers)
            if "camera_image" in sensor_data:
                obstacle_mapper.update_obstacle_colors(sensor_data["camera_image"], [robot_x, robot_y, robot_yaw])
            
    return robot_x, robot_y, robot_yaw

# --- STAGE 3: PLANNING (State Machine) ---
def run_planning(pose, sensor_data):
    """
    Stage 3: Pfadplanung & Verhaltenssteuerung.
    Verarbeitet Zustandsübergänge der State Machine und berechnet den Soll-Pfad
    sowie die Soll-Geschwindigkeit und den Soll-Lenkwinkel.
    """
    global state, prev_error, USE_RL, ort_session
    
    target_speed = 0.0
    target_steering = 0.0
    
    lidar_data = sensor_data["lidar_ranges"]
    
    if state == "STOP":
        # Warten bis die initiale Lokalisierung abgeschlossen ist
        if initial_pose_found:
            print(f"[State Machine] STOP -> RUNNING (Start sub: EXPLORE in direction {driving_direction})")
            state = "RUNNING"
            
    elif state == "RUNNING":
        # Fehlerprüfung (Sensorverlust)
        if len(lidar_data) == 0:
            print("[State Machine] Kritischer Fehler! Kein LiDAR-Signal im RUNNING-Zustand.")
            state = "ERROR"
            return 0.0, 0.0
            
        if USE_RL and ort_session is not None:
            # --- Reinforcement Learning Path Planning ---
            rx, ry, ryaw = pose
            
            def cap(dist):
                return min(dist, 1.5)
                
            right_dist = cap(lidar_data[270])
            left_dist = cap(lidar_data[90])
            front_dist = cap(lidar_data[180])
            front_right_dist = cap(lidar_data[225])
            front_left_dist = cap(lidar_data[135])
            
            # Find next obstacle in front
            next_obs_x = 2.0
            next_obs_y = 0.0
            next_obs_color = 0.0 # 0=gray/none, 1=red, -1=green
            
            alpha = ryaw + math.pi / 2.0
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            
            best_dist = float('inf')
            for obs in obstacle_mapper.obstacles:
                ox, oy = obs.position
                dx = ox - rx
                dy = oy - ry
                x_loc = dx * cos_a + dy * sin_a
                y_loc = -dx * sin_a + dy * cos_a
                
                # Check if obstacle is in front of the vehicle
                if x_loc > 0.0:
                    dist = math.hypot(x_loc, y_loc)
                    if dist < best_dist and dist < 2.0:
                        best_dist = dist
                        next_obs_x = x_loc
                        next_obs_y = y_loc
                        if obs.color == "red":
                            next_obs_color = 1.0
                        elif obs.color == "green":
                            next_obs_color = -1.0
                        else:
                            next_obs_color = 0.0
                            
            # Build normalized observation vector (matching wro_gym_env.py)
            obs_vector = np.array([
                left_dist / 1.5,
                right_dist / 1.5,
                front_dist / 1.5,
                front_left_dist / 1.5,
                front_right_dist / 1.5,
                next_obs_x / 2.0,
                next_obs_y / 2.0,
                next_obs_color
            ], dtype=np.float32)
            
            # Clip bounds matching wro_gym_env.py
            low_bounds = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)
            high_bounds = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            obs_vector = np.clip(obs_vector, low_bounds, high_bounds)
            
            # Run ONNX inference
            try:
                obs_input = np.array([obs_vector], dtype=np.float32)
                ort_inputs = {ort_session.get_inputs()[0].name: obs_input}
                ort_outs = ort_session.run(None, ort_inputs)
                
                # ONNX policy outputs (action, value, log_prob)
                action = ort_outs[0][0][0]
                
                target_steering = action * MAX_STEERING
                target_speed = TARGET_SPEED
            except Exception as e:
                print(f"[RL Error] Inference failed: {e}. Falling back to Rules-based.")
                USE_RL = False
                return run_planning(pose, sensor_data)
        else:
            # --- Rules-based Path Planning ---
            target_speed = TARGET_SPEED
            
            def cap(dist):
                return min(dist, 1.5)
    
            # Distanzen ermitteln: 90=Links, 270=Rechts, 180=Vorne, 135=Vorne-Links, 225=Vorne-Rechts
            right_dist = cap(lidar_data[270])
            left_dist = cap(lidar_data[90])
            front_dist = lidar_data[180]
            front_right_dist = cap(lidar_data[225])
            front_left_dist = cap(lidar_data[135])
            
            # 1. Zentrierung & Parallelfahrt (PD-Regler)
            current_diff = left_dist - right_dist
            derivative = current_diff - prev_error
            prev_error = current_diff
            
            raw_steering = -(P_GAIN * current_diff + D_GAIN * derivative)
            
            # 2. Seitenwandschutz (Sicherheitsabstand min. 20cm)
            min_side = min(left_dist, right_dist)
            if min_side < 0.2:
                safety_force = (0.2 - min_side) * 5.0 
                if left_dist < right_dist:
                    raw_steering += safety_force  # Nach rechts lenken
                else:
                    raw_steering -= safety_force  # Nach links lenken
            
            # 3. Kurvenfahrt / Frontwand-Ausweichmanöver
            if front_dist < 1.0:
                corner_force = (1.0 - front_dist)**2
                CORNER_GAIN = 8.0 
                
                if front_left_dist > front_right_dist:
                    raw_steering -= (corner_force * CORNER_GAIN)   # Links lenken
                else:
                    raw_steering += (corner_force * CORNER_GAIN)   # Rechts lenken
            
            # Begrenzung des Lenkwinkels
            target_steering = clamp(raw_steering, -MAX_STEERING, MAX_STEERING)
        
    elif state == "COMPLETED":
        target_speed = 0.0
        target_steering = 0.0
        
    elif state == "ERROR":
        target_speed = 0.0
        target_steering = 0.0
        
    return target_speed, target_steering

# --- STAGE 4: CONTROL ---
def run_control(target_speed, target_steering):
    """
    Stage 4: Regelung.
    Rechnet Soll-Vorgaben in reale Aktuatorsignale um (Ackermann, PID, Filter).
    Enthält KEINE übergeordnete Zustandslogik.
    """
    global smoothed_steering
    
    # 1. Tiefpassfilter auf den Lenkwinkel anwenden
    smoothed_steering = smoothed_steering + 0.5 * (target_steering - smoothed_steering)
    
    # 2. Dynamische Geschwindigkeitsanpassung basierend auf dem Lenkwinkel
    speed_factor = 1.0 - (abs(smoothed_steering) / MAX_STEERING) * 0.3
    speed = target_speed * max(0.7, speed_factor)
    
    # Motoren ansteuern
    motor_right.setVelocity(speed)
    motor_left.setVelocity(speed)
    
    # Lenkung einstellen
    set_steering_angle(smoothed_steering)

# --- MAIN LOOP ---
step_count = 0

while robot.step(TIME_STEP) != -1:
    step_count += 1
    
    # Optionale waitKey Behandlung zur Vermeidung von OpenCV-Hängern
    try:
        cv2.waitKey(1)
    except Exception:
        pass
        
    # --- 1. STAGE 1: PERCEPTION ---
    sensor_data = run_perception(robot, lidar, imu, camera)
    
    # --- 2. STAGE 2: ESTIMATION ---
    robot_pose = run_estimation(localizer, icp_localizer, obstacle_mapper, sensor_data)
    
    # --- 3. STAGE 3: PLANNING ---
    target_speed, target_steering = run_planning(robot_pose, sensor_data)
    
    # --- 4. STAGE 4: CONTROL ---
    run_control(target_speed, target_steering)
    
    # --- Live OpenCV Visualisierung ---
    if len(sensor_data["lidar_ranges"]) > 0:
        vis_img = icp_localizer.render() if initial_pose_found else localizer.render()
        if initial_pose_found:
            vis_img = obstacle_mapper.render(vis_img, robot_pose, icp_localizer.scale, icp_localizer.window_size)
            
            # Overlay mode indicator text
            mode_text = "Planning: RL (ONNX)" if USE_RL else "Planning: Rules-based"
            text_color = (0, 255, 0) if USE_RL else (0, 165, 255) # Green vs Orange BGR
            cv2.putText(
                vis_img,
                mode_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
                lineType=cv2.LINE_AA
            )
            
            if "camera_image" in sensor_data:
                cam_debug = obstacle_mapper.render_camera(sensor_data["camera_image"], robot_pose)
                try:
                    cv2.imshow("WRO Camera Debug", cam_debug)
                except Exception:
                    pass
        try:
            cv2.imshow("WRO Localization", vis_img)
        except Exception:
            pass
