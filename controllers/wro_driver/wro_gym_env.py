import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import cv2

try:
    from controller import Supervisor
except ModuleNotFoundError:
    import sys
    import os
    # Try to find Webots installation to add its Python controller libraries to path
    webots_paths = [
        os.environ.get("WEBOTS_HOME", ""),
        "C:\\Program Files\\Webots",
        os.path.expandvars("%LOCALAPPDATA%\\Programs\\Webots")
    ]
    found = False
    for path in webots_paths:
        if path and os.path.exists(path):
            py_path = os.path.join(path, "lib", "controller", "python")
            if os.path.exists(py_path):
                sys.path.append(py_path)
                found = True
                break
    if not found:
        raise ModuleNotFoundError(
            "Could not find Webots installation. Please set the WEBOTS_HOME environment variable "
            "or install Webots in its default location."
        )
    from controller import Supervisor

class WebotsWroEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(WebotsWroEnv, self).__init__()
        
        # Action space:
        # Output 2 floats:
        # - Dim 0: target_steering_offset in [-1.0, 1.0] (scaled to [-MAX_STEERING, MAX_STEERING])
        # - Dim 1: throttle/speed in [0.0, 1.0] (scaled to [0.0, max_motor_velocity])
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), shape=(2,), dtype=np.float32
        )
        
        # Observation space (12 elements, normalized to [-1.0, 1.0] or [0.0, 1.0]):
        # 0: ego_v_x ([-1.0, 1.0])
        # 1: last_steering ([-1.0, 1.0])
        # 2: lateral_error ([-1.0, 1.0])
        # 3: heading_error ([-1.0, 1.0])
        # 4: lookahead_curv_30 ([-1.0, 1.0])
        # 5: lookahead_curv_60 ([-1.0, 1.0])
        # 6-8: obs1_rel_x, obs1_rel_y, obs1_color ([-1.0, 1.0])
        # 9-11: obs2_rel_x, obs2_rel_y, obs2_color ([-1.0, 1.0])
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 12, dtype=np.float32),
            high=np.array([1.0] * 12, dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize Webots Supervisor
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Devices (get from robot)
        self.camera = self.supervisor.getDevice("camera")
        self.camera.enable(self.timestep)
        
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)
        
        self.imu = self.supervisor.getDevice("imu")
        self.imu.enable(self.timestep)
        
        self.motor_right = self.supervisor.getDevice("motor_rear_right")
        self.motor_left = self.supervisor.getDevice("motor_rear_left")
        self.motor_right.setPosition(float('inf'))
        self.motor_left.setPosition(float('inf'))
        self.motor_right.setVelocity(0.0)
        self.motor_left.setVelocity(0.0)
        
        self.steer_left = self.supervisor.getDevice("left_steer")
        self.steer_right = self.supervisor.getDevice("right_steer")
        
        # Get robot node for teleporting and collision tracking
        self.robot_node = self.supervisor.getSelf()
        self.robot_node.enableContactPointsTracking(self.timestep, includeDescendants=True)
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        
        # Initial pose values for reset (matching wro_driver start coordinates)
        self.initial_translation = [-0.221441, -1.01067, 0.0300003]
        self.initial_rotation = [0, 0, 1, 0] # Face East (+X)
        
        # State estimation helper modules
        from opencv_localizer import OpenCVLocalizer
        from trans_icp_localizer import TranslationICPLocalizer
        from obstacle_mapper import ObstacleMapper
        
        self.localizer = OpenCVLocalizer()
        self.icp_localizer = TranslationICPLocalizer()
        self.obstacle_mapper = ObstacleMapper()
        
        # Get obstacle translation fields from DEF names for reset randomization
        self.red_obstacle_fields = []
        self.green_obstacle_fields = []
        for i in range(3):
            node_red = self.supervisor.getFromDef(f"OBSTACLE_RED_{i}")
            if node_red:
                self.red_obstacle_fields.append(node_red.getField("translation"))
            node_green = self.supervisor.getFromDef(f"OBSTACLE_GREEN_{i}")
            if node_green:
                self.green_obstacle_fields.append(node_green.getField("translation"))
        
        # Constants
        self.MAX_STEERING = 0.8
        self.TARGET_SPEED = 5.0
        self.max_motor_velocity = 10.0
        self.WHEELBASE = 0.14
        self.TRACK_FRONT = 0.12
        
        # Timing
        self.control_freq = 10 # Hz
        self.frame_skip = int(1000 / (self.control_freq * self.timestep))
        if self.frame_skip < 1:
            self.frame_skip = 1
            
        # State tracking variables
        self.imu_yaw_initial = None
        self.smoothed_steering = 0.0  # Low-pass filter matching Stage 4 Control
        self.smoothed_speed = 0.0
        self.stagnation_counter = 0
        self.current_checkpoint_idx = 0
        self.steps_since_last_checkpoint = 0
        self.total_steps = 0
        self.passed_obstacle_ids = set()
        self.driving_direction = "CCW"
        
        # Centralized checkpoints (16 points around 2.0x2.0m loop centered at 1.5,1.5)
        self.checkpoints = [
            (1.5, 0.5), (2.0, 0.5), (2.5, 0.5), (2.5, 1.0),
            (2.5, 1.5), (2.5, 2.0), (2.5, 2.5), (2.0, 2.5), (1.5, 2.5),
            (1.0, 2.5), (0.5, 2.5), (0.5, 2.0), (0.5, 1.5), (0.5, 1.0),
            (0.5, 0.5), (1.0, 0.5)
        ]
        self.checkpoints_cleared_this_lap = 0

    def ackermann_angles(self, target_angle):
        if abs(target_angle) < 1e-6:
            return 0.0, 0.0
        R = self.WHEELBASE / math.tan(abs(target_angle))
        inner = math.atan(self.WHEELBASE / (R - self.TRACK_FRONT / 2.0))
        outer = math.atan(self.WHEELBASE / (R + self.TRACK_FRONT / 2.0))
        if target_angle > 0: 
            return outer, inner
        else: 
            return -inner, -outer

    def set_steering_angle(self, target_angle):
        left, right = self.ackermann_angles(target_angle)
        left = max(-self.MAX_STEERING, min(self.MAX_STEERING, left))
        right = max(-self.MAX_STEERING, min(self.MAX_STEERING, right))
        self.steer_left.setPosition(left)
        self.steer_right.setPosition(right)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Stop robot
        self.motor_right.setVelocity(0.0)
        self.motor_left.setVelocity(0.0)
        self.steer_left.setPosition(0.0)
        self.steer_right.setPosition(0.0)
        
        # Zufällige Pose in Schätzer-Koordinaten wählen (1.2, 0.2) - (1.8, 0.8)
        start_x_est = self.np_random.uniform(1.2, 1.8)
        start_y_est = self.np_random.uniform(0.2, 0.8)
        yaw_est = self.np_random.uniform(-np.pi, np.pi)
        
        # In Webots-Koordinaten umrechnen
        x_webots = start_x_est - 1.5
        y_webots = start_y_est - 1.5
        z_webots = 0.030000  # Höhe konstant halten
        angle_webots = yaw_est + math.pi / 2.0
        
        # Position & Rotation in Webots anwenden
        self.translation_field.setSFVec3f([x_webots, y_webots, z_webots])
        self.rotation_field.setSFRotation([0.0, 0.0, 1.0, angle_webots])
        
        # Hindernisse zurücksetzen und zufällig neu platzieren
        for field in self.red_obstacle_fields + self.green_obstacle_fields:
            field.setSFVec3f([0.0, 0.0, 0.05])
            
        if self.red_obstacle_fields or self.green_obstacle_fields:
            available_red = list(self.red_obstacle_fields)
            available_green = list(self.green_obstacle_fields)
            
            # Hindernisse stehen nur im Westen, Norden, Osten (nicht im Süden)
            sections = ["Westen", "Norden", "Osten"]
            for section in sections:
                # Wähle Szenario: 0 (kein Hindernis), 1 (Mitte), 2 (ein Rand), 3 (beide Ränder)
                scenario = self.np_random.choice([0, 1, 2, 3])
                
                slots = []
                if scenario == 1:
                    slots = [0.0]
                elif scenario == 2:
                    slots = [self.np_random.choice([-0.5, 0.5])]
                elif scenario == 3:
                    slots = [-0.5, 0.5]
                    
                for s in slots:
                    # Querverschiebung (Lateral) bestimmen
                    if section == "Westen":
                        d = self.np_random.choice([-0.9, -1.1])
                    else:
                        d = self.np_random.choice([0.9, 1.1])
                        
                    # Globale Koordinaten berechnen
                    if section == "Westen":
                        x, y = d, s
                    elif section == "Norden":
                        x, y = s, d
                    elif section == "Osten":
                        x, y = d, s
                        
                    # Farbe basierend auf den verfügbaren physischen Boxen bestimmen
                    chosen_color = None
                    if available_red and available_green:
                        chosen_color = self.np_random.choice(["red", "green"])
                    elif available_red:
                        chosen_color = "red"
                    elif available_green:
                        chosen_color = "green"
                        
                    if chosen_color == "red":
                        field = available_red.pop()
                        field.setSFVec3f([x, y, 0.05])
                    elif chosen_color == "green":
                        field = available_green.pop()
                        field.setSFVec3f([x, y, 0.05])
        
        # Reset physics
        self.supervisor.simulationResetPhysics()
        
        # Step simulation to apply physics reset
        for _ in range(5):
            self.supervisor.step(self.timestep)
            
        # Reset localizers and state variables
        self.imu_yaw_initial = None
        
        # Collect 10 scans of Lidar data to calibrate starting position (similar to wro_driver.py)
        collected_scans = []
        import warnings
        
        # Step simulation to accumulate scans while robot remains stationary
        for _ in range(10):
            if self.supervisor.step(self.timestep) == -1:
                break
            
            # Read IMU to set/keep baseline
            rpy = self.imu.getRollPitchYaw()
            imu_yaw_raw = rpy[2] if rpy else 0.0
            if self.imu_yaw_initial is None:
                self.imu_yaw_initial = imu_yaw_raw
                
            lidar_data = self.lidar.getRangeImage()
            if lidar_data is not None and len(lidar_data) > 0:
                collected_scans.append(lidar_data)
                
        # Perform initial pose calibration using OpenCV template matching
        calibrated = False
        if len(collected_scans) >= 10:
            scans_arr = np.array(collected_scans)
            # Filter invalid values
            invalid_mask = (scans_arr <= 0.01) | (scans_arr >= 2.0) | np.isinf(scans_arr) | np.isnan(scans_arr)
            scans_arr[invalid_mask] = np.nan
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_ranges = np.nanmean(scans_arr, axis=0)
            avg_ranges = np.nan_to_num(avg_ranges, nan=0.0)
            
            try:
                n_rays = len(avg_ranges)
                angle_inc = -2.0 * math.pi / n_rays if n_rays > 0 else 0.0
                angle_offset = math.pi / 2
                
                x_init, y_init, yaw_init, direction, debug_img = self.localizer.calibrate_initial_pose(
                    avg_ranges=avg_ranges,
                    angle_offset=angle_offset,
                    angle_inc=angle_inc
                )
                #print(f"[Gym Env Calibration] Best candidate found at: x={x_init:.3f}m, y={y_init:.3f}m, yaw={yaw_init:.3f} rad ({math.degrees(yaw_init):.1f}°)")
                print(f"[Gym Env Calibration] Resolved driving direction: {direction}")
                
                self.localizer.set_initial_pose(x_init, y_init, yaw_init)
                self.icp_localizer.set_initial_pose(x_init, y_init, yaw_init)
                self.driving_direction = direction
                calibrated = True
            except Exception as e:
                print(f"[Gym Env Calibration] Error during calibration: {e}")
                
        if not calibrated:
            # Fallback auf die tatsächlich generierte Pose
            self.localizer.set_initial_pose(start_x_est, start_y_est, yaw_est)
            self.icp_localizer.set_initial_pose(start_x_est, start_y_est, yaw_est)
            # Richtung aus yaw_est ableiten
            sin_yaw = math.sin(yaw_est)
            if sin_yaw < 0 or (abs(sin_yaw) < 1e-5 and yaw_est < 0):
                self.driving_direction = "CCW"
            else:
                self.driving_direction = "CW"
        
        self.obstacle_mapper.obstacles = []
        self.passed_obstacle_ids = set()
        
        self.smoothed_steering = 0.0  # Reset low-pass filter matching Stage 4 Control
        self.smoothed_speed = 0.0
        self.stagnation_counter = 0
        self.checkpoints_cleared_this_lap = 0
        
        # Nächstgelegenen Checkpoint ermitteln
        dists = [math.hypot(start_x_est - cp[0], start_y_est - cp[1]) for cp in self.checkpoints]
        self.current_checkpoint_idx = int(np.argmin(dists))
        
        self.steps_since_last_checkpoint = 0
        self.total_steps = 0
        
        # Get observation
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_closest_point_on_path(self, rx, ry):
        if self.driving_direction == "CCW":
            vertices = [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]
        else:
            vertices = [(2.5, 0.5), (0.5, 0.5), (0.5, 2.5), (2.5, 2.5)]
            
        p = np.array([rx, ry])
        best_dist = float('inf')
        best_closest = None
        best_tangent = None
        best_segment_idx = 0
        best_t = 0.0
        
        for i in range(4):
            a = np.array(vertices[i])
            b = np.array(vertices[(i + 1) % 4])
            ap = p - a
            ab = b - a
            ab_len_sq = np.dot(ab, ab)
            if ab_len_sq < 1e-9:
                t = 0.0
            else:
                t = np.dot(ap, ab) / ab_len_sq
                t = np.clip(t, 0.0, 1.0)
            closest = a + t * ab
            dist = np.linalg.norm(p - closest)
            if dist < best_dist:
                best_dist = dist
                best_closest = closest
                best_segment_idx = i
                best_t = t
                ab_norm = np.linalg.norm(ab)
                best_tangent = ab / ab_norm if ab_norm > 1e-9 else np.array([1.0, 0.0])
                
        return best_closest, best_dist, best_tangent, best_segment_idx, best_t

    def _get_point_at_s(self, s):
        if self.driving_direction == "CCW":
            vertices = [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]
        else:
            vertices = [(2.5, 0.5), (0.5, 0.5), (0.5, 2.5), (2.5, 2.5)]
            
        s = s % 8.0
        k = int(s // 2.0) % 4
        t = (s % 2.0) / 2.0
        p_start = np.array(vertices[k])
        p_end = np.array(vertices[(k + 1) % 4])
        return p_start + t * (p_end - p_start)

    def _get_obs(self):
        # Perception
        lidar_data = self.lidar.getRangeImage()
        if lidar_data is None or len(lidar_data) == 0:
            lidar_data = [1.5] * 360
            
        rpy = self.imu.getRollPitchYaw()
        imu_yaw = rpy[2] - self.imu_yaw_initial if (rpy and self.imu_yaw_initial is not None) else 0.0
        
        # Estimation
        rx, ry, ryaw, outliers = self.icp_localizer.update(lidar_data, imu_yaw)
        self.obstacle_mapper.update([rx, ry, ryaw], outliers)
        
        img_buffer = self.camera.getImage()
        if img_buffer:
            w = self.camera.getWidth()
            h = self.camera.getHeight()
            img_raw = np.frombuffer(img_buffer, dtype=np.uint8).reshape((h, w, 4))
            img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)
            self.obstacle_mapper.update_obstacle_colors(img_bgr, [rx, ry, ryaw])
            
        # Get ego velocity using robot node orientation and velocity
        vel = self.robot_node.getVelocity()
        if vel is not None:
            v_g = vel[:3]
            R = self.robot_node.getOrientation()
            if R is not None:
                # Local X axis is [R[0], R[3], R[6]]
                ego_v_x = v_g[0] * R[0] + v_g[1] * R[3] + v_g[2] * R[6]
            else:
                ego_v_x = 0.0
        else:
            ego_v_x = 0.0
            
        # 1. lateral_error
        best_closest, best_dist, best_tangent, best_segment_idx, best_t = self._get_closest_point_on_path(rx, ry)
        left_normal = np.array([-best_tangent[1], best_tangent[0]])
        p = np.array([rx, ry])
        lateral_error = np.dot(p - best_closest, left_normal)
        lateral_error_normalized = np.clip(lateral_error / 0.5, -1.0, 1.0)
        
        # 2. heading_error
        theta_tangent = math.atan2(best_tangent[1], best_tangent[0])
        diff_yaw = ryaw - theta_tangent
        diff_yaw = (diff_yaw + math.pi) % (2.0 * math.pi) - math.pi
        heading_error_normalized = np.clip(diff_yaw / (math.pi / 2.0), -1.0, 1.0)
        
        # 3. lookahead points (30cm & 60cm)
        s_current = best_segment_idx * 2.0 + best_t * 2.0
        
        s_30 = (s_current + 0.3) % 8.0
        p_30 = self._get_point_at_s(s_30)
        
        s_60 = (s_current + 0.6) % 8.0
        p_60 = self._get_point_at_s(s_60)
        
        # Transform lookahead points to local coordinates
        alpha = ryaw + math.pi / 2.0
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        
        dx_30 = p_30[0] - rx
        dy_30 = p_30[1] - ry
        y_loc_30 = -dx_30 * sin_a + dy_30 * cos_a
        
        dx_60 = p_60[0] - rx
        dy_60 = p_60[1] - ry
        y_loc_60 = -dx_60 * sin_a + dy_60 * cos_a
        
        # 4. Obstacles
        valid_obstacles = []
        for obs in self.obstacle_mapper.obstacles:
            ox, oy = obs.position
            dx = ox - rx
            dy = oy - ry
            x_loc = dx * cos_a + dy * sin_a
            y_loc = -dx * sin_a + dy * cos_a
            
            if x_loc > 0.0:
                color_val = 1.0 if obs.color == "green" else -1.0 if obs.color == "red" else 0.0
                valid_obstacles.append({
                    "x_loc": x_loc,
                    "y_loc": y_loc,
                    "color": color_val
                })
                
        # Sort by relative x_loc ascending
        valid_obstacles.sort(key=lambda item: item["x_loc"])
        
        obs_features = []
        for i in range(2):
            if i < len(valid_obstacles):
                o = valid_obstacles[i]
                rel_x = np.clip(o["x_loc"] / 2.0, -1.0, 1.0)
                rel_y = np.clip(o["y_loc"] / 2.0, -1.0, 1.0)
                color = o["color"]
            else:
                rel_x = 1.0  # 2.0 / 2.0
                rel_y = 0.0
                color = 0.0
            obs_features.extend([rel_x, rel_y, color])
            
        # Build observation vector
        obs_vector = np.array([
            np.clip(ego_v_x, -1.0, 1.0),
            np.clip(self.smoothed_steering / self.MAX_STEERING, -1.0, 1.0),
            lateral_error_normalized,
            heading_error_normalized,
            np.clip(y_loc_30 / 0.3, -1.0, 1.0),
            np.clip(y_loc_60 / 0.6, -1.0, 1.0),
            obs_features[0],
            obs_features[1],
            obs_features[2],
            obs_features[3],
            obs_features[4],
            obs_features[5]
        ], dtype=np.float32)
        
        # Clip to ensure valid observation space boundaries
        obs_vector = np.clip(obs_vector, self.observation_space.low, self.observation_space.high)
        return obs_vector

    def step(self, action):
        # 1. Apply action (target steering offset)
        target_steering = float(action[0]) * self.MAX_STEERING
        
        # Apply low-pass filter on steering
        self.smoothed_steering = self.smoothed_steering + 0.2 * (target_steering - self.smoothed_steering)
        
        # Throttle/Speed is action[1] scaled by self.max_motor_velocity
        target_speed = float(action[1]) * self.max_motor_velocity
        
        # Apply low-pass filter on speed
        self.smoothed_speed = self.smoothed_speed + 0.2 * (target_speed - self.smoothed_speed)
        
        # Set actuator commands
        self.set_steering_angle(self.smoothed_steering)
        self.motor_right.setVelocity(self.smoothed_speed)
        self.motor_left.setVelocity(self.smoothed_speed)
        
        # 2. Step simulation physics
        for _ in range(self.frame_skip):
            if self.supervisor.step(self.timestep) == -1:
                break
                
        # 3. Get new observation
        obs = self._get_obs()
        
        # 4. Compute reward & check termination
        rx = self.icp_localizer.X_real
        ry = self.icp_localizer.Y_real
        ryaw = self.icp_localizer.yaw_real
        
        # Calculate lateral error for reward
        best_closest, best_dist, best_tangent, best_segment_idx, best_t = self._get_closest_point_on_path(rx, ry)
        left_normal = np.array([-best_tangent[1], best_tangent[0]])
        p = np.array([rx, ry])
        lateral_error = np.dot(p - best_closest, left_normal)
        lateral_error_normalized = np.clip(lateral_error / 0.5, -1.0, 1.0)
        
        # Get ego velocity
        vel = self.robot_node.getVelocity()
        if vel is not None:
            v_g = vel[:3]
            R = self.robot_node.getOrientation()
            if R is not None:
                # Local X axis is [R[0], R[3], R[6]]
                ego_v_x = v_g[0] * R[0] + v_g[1] * R[3] + v_g[2] * R[6]
            else:
                ego_v_x = 0.0
        else:
            ego_v_x = 0.0
            
        # Reward function: ego_v_x * 1.0 - abs(lateral_error_normalized) * 0.1
        reward = ego_v_x * 1.0 - abs(lateral_error_normalized) * 0.1
        
        terminated = False
        truncated = False
        
        # Check collision via Webots physics contact points (chassis or wheels touching wall/obstacle)
        collision = False
        try:
            contact_points = self.robot_node.getContactPoints(includeDescendants=True)
            for cp in contact_points:
                # cp.getPoint() returns [x, y, z] in global coordinates.
                # Floor contacts are at z ≈ 0.0m.
                # Wall or obstacle contacts are at z > 0.01m (1cm) since the chassis and wheels contact them higher.
                if cp.getPoint()[2] > 0.01:
                    collision = True
                    break
        except Exception as e:
            print(f"[Gym Env] Error checking contact points: {e}")
                
        if collision:
            reward = -20.0
            terminated = True
            print("[Gym Env] COLLISION detected! Resetting.")
            
        # Checkpoint Progress tracking
        checkpoint_reached = False
        step_dir = 1 if self.driving_direction == "CCW" else -1
        for lookahead in [1, 2]:
            idx = (self.current_checkpoint_idx + lookahead * step_dir) % len(self.checkpoints)
            target_cp = self.checkpoints[idx]
            dist_to_cp = math.hypot(rx - target_cp[0], ry - target_cp[1])
            if dist_to_cp < 0.55:  # Increased from 0.35 to 0.55 to be robust against wall-hugging/swerving
                # Check if we crossed/landed on index 0 (lap completion)
                crossed_zero = False
                for i in range(1, lookahead + 1):
                    check_idx = (self.current_checkpoint_idx + i * step_dir) % len(self.checkpoints)
                    if check_idx == 0:
                        crossed_zero = True
                
                self.current_checkpoint_idx = idx
                self.checkpoints_cleared_this_lap += lookahead
                checkpoint_reached = True
                
                if crossed_zero:
                    if self.checkpoints_cleared_this_lap >= len(self.checkpoints) - 2:
                        terminated = True
                        print(f"[Gym Env] LAP completed ({self.driving_direction})! Terminating episode.")
                    self.checkpoints_cleared_this_lap = 0
                break
        
        # Stagnation check: truncate if ego_v_x < 0.01 m/s for 50 steps
        if ego_v_x < 0.01:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            
        if self.stagnation_counter >= 50:
            truncated = True
            print("[Gym Env] TIMEOUT (Stillstand)! Resetting.")
            
        # Obstacle passing side validation
        alpha = ryaw + math.pi / 2.0
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        
        for obstacle in self.obstacle_mapper.obstacles:
            if obstacle.color not in ["red", "green"]:
                continue
                
            # Create unique ID for obstacle based on position
            obs_key = (round(obstacle.position[0], 1), round(obstacle.position[1], 1))
            if obs_key in self.passed_obstacle_ids:
                continue
                
            ox, oy = obstacle.position
            dx = ox - rx
            dy = oy - ry
            x_loc = dx * cos_a + dy * sin_a
            y_loc = -dx * sin_a + dy * cos_a
            
            # Check if robot has just passed it (x_loc goes behind, within lateral range)
            if -0.3 < x_loc <= 0.0 and math.hypot(x_loc, y_loc) < 0.8:
                self.passed_obstacle_ids.add(obs_key)
                
                # Check correct side:
                if obstacle.color == "red":
                    # Red obstacle: must pass on the right (obstacle on left side, y_loc > 0)
                    if y_loc > 0:
                        reward += 15.0
                        print(f"[Gym Env] RED Obstacle passed CORRECTLY! Reward +15.0")
                    else:
                        reward = -20.0
                        terminated = True
                        print(f"[Gym Env] RED Obstacle passed INCORRECTLY! Resetting.")
                elif obstacle.color == "green":
                    # Green obstacle: must pass on the left (obstacle on right side, y_loc < 0)
                    if y_loc < 0:
                        reward += 15.0
                        print(f"[Gym Env] GREEN Obstacle passed CORRECTLY! Reward +15.0")
                    else:
                        reward = -20.0
                        terminated = True
                        print(f"[Gym Env] GREEN Obstacle passed INCORRECTLY! Resetting.")
                        
        self.total_steps += 1
        # Hard timeout after 1000 steps (100 seconds)
        if self.total_steps > 1000:
            truncated = True
            
        info = {
            "x": rx,
            "y": ry,
            "yaw": ryaw,
            "collision": collision,
            "checkpoint": self.current_checkpoint_idx
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        rx = self.icp_localizer.X_real
        ry = self.icp_localizer.Y_real
        ryaw = self.icp_localizer.yaw_real
        
        # Get background ICP render
        vis_img = self.icp_localizer.render()
        
        # Render obstacles
        vis_img = self.obstacle_mapper.render(vis_img, [rx, ry, ryaw], self.icp_localizer.scale, self.icp_localizer.window_size)
        
        # Draw next checkpoint
        step_dir = 1 if self.driving_direction == "CCW" else -1
        next_idx = (self.current_checkpoint_idx + step_dir) % len(self.checkpoints)
        cx, cy = self.checkpoints[next_idx]
        cx_px = int(cx * self.icp_localizer.scale)
        cy_px = int(self.icp_localizer.window_size - cy * self.icp_localizer.scale)
        cv2.circle(vis_img, (cx_px, cy_px), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA) # Yellow checkpoint indicator
        
        cv2.imshow("RL Training Environment", vis_img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
