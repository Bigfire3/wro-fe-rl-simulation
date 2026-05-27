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
        # Output 1 float: target_steering_offset in [-1.0, 1.0]
        # (We scale it to [-MAX_STEERING, MAX_STEERING] in step())
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space (normalized to [0.0, 1.0] for distances and [-1.0, 1.0] for coordinates/color):
        # 1. Left wall distance (0 to 1.0)
        # 2. Right wall distance (0 to 1.0)
        # 3. Front wall distance (0 to 1.0)
        # 4. Front-left wall distance (0 to 1.0)
        # 5. Front-right wall distance (0 to 1.0)
        # 6. Next obstacle local X (relative forward, -1.0 to 1.0)
        # 7. Next obstacle local Y (relative side, -1.0 to 1.0)
        # 8. Next obstacle color / action requirement (-1.0 for green = pass left, 1.0 for red = pass right, 0.0 for none)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
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
        
        # Constants
        self.MAX_STEERING = 0.8
        self.TARGET_SPEED = 5.0
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
        self.current_checkpoint_idx = 0
        self.steps_since_last_checkpoint = 0
        self.total_steps = 0
        self.passed_obstacle_ids = set()

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
        
        # Reset position in Webots
        self.translation_field.setSFVec3f(self.initial_translation)
        self.rotation_field.setSFRotation(self.initial_rotation)
        
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
                #print(f"[Gym Env Calibration] Resolved driving direction: {direction}")
                
                self.localizer.set_initial_pose(x_init, y_init, yaw_init)
                self.icp_localizer.set_initial_pose(x_init, y_init, yaw_init)
                calibrated = True
            except Exception as e:
                print(f"[Gym Env Calibration] Error during calibration: {e}")
                
        if not calibrated:
            # Fallback
            start_x_est = 1.5 + self.initial_translation[0]
            start_y_est = 1.5 + self.initial_translation[1]
            self.localizer.set_initial_pose(start_x_est, start_y_est, 0.0)
            self.icp_localizer.set_initial_pose(start_x_est, start_y_est, 0.0)
        
        self.obstacle_mapper.obstacles = []
        self.passed_obstacle_ids = set()
        
        self.smoothed_steering = 0.0  # Reset low-pass filter matching Stage 4 Control
        self.current_checkpoint_idx = 0
        self.steps_since_last_checkpoint = 0
        self.total_steps = 0
        
        # Get observation
        obs = self._get_obs()
        info = {}
        return obs, info

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
            
        # Extract features
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
        
        # alpha coordinate rotation aligned with camera projection
        alpha = ryaw + math.pi / 2.0
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)
        
        best_dist = float('inf')
        for obs in self.obstacle_mapper.obstacles:
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
                        
        # Scale to normalize the observations:
        # Lidar distances (indices 0 to 4) are currently capped at 1.5m -> divide by 1.5 to get [0, 1.0]
        # Obstacle X and Y (indices 5 and 6) are currently in [-2.0, 2.0] -> divide by 2.0 to get [-1.0, 1.0]
        # Obstacle color (index 7) is already in [-1.0, 1.0]
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
        
        # Clip to ensure valid observation space boundaries
        obs_vector = np.clip(obs_vector, self.observation_space.low, self.observation_space.high)
        return obs_vector

    def step(self, action):
        # 1. Apply action (target steering offset)
        target_steering = float(action[0]) * self.MAX_STEERING
        
        # Apply low-pass filter on steering (matching Stage 4 Control of wro_driver)
        self.smoothed_steering = self.smoothed_steering + 0.5 * (target_steering - self.smoothed_steering)
        
        # Dynamic speed adjustment based on steering angle (matching Stage 4 Control of wro_driver)
        speed_factor = 1.0 - (abs(self.smoothed_steering) / self.MAX_STEERING) * 0.3
        speed = self.TARGET_SPEED * max(0.7, speed_factor)
        
        # Set actuator commands
        self.set_steering_angle(self.smoothed_steering)
        self.motor_right.setVelocity(speed)
        self.motor_left.setVelocity(speed)
        
        # 2. Step simulation physics
        for _ in range(self.frame_skip):
            if self.supervisor.step(self.timestep) == -1:
                break
                
        # 3. Get new observation
        obs = self._get_obs()
        
        # 4. Compute reward & check termination
        reward = 0.0
        terminated = False
        truncated = False
        
        rx = self.icp_localizer.X_real
        ry = self.icp_localizer.Y_real
        ryaw = self.icp_localizer.yaw_real
        
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
            reward -= 15.0 # High collision penalty
            terminated = True
            print("[Gym Env] COLLISION detected! Resetting.")
            
        # Checkpoint Progress tracking
        # Arena midline checkpoints (17 points around 2.0x2.0m loop centered at 1.5,1.5)
        checkpoints = [
            (1.28, 0.49), # Start pos
            (1.6, 0.5),
            (2.0, 0.5),
            (2.5, 0.5),   # Corner 1
            (2.5, 1.0),
            (2.5, 1.5),   # East center
            (2.5, 2.0),
            (2.5, 2.5),   # Corner 2
            (2.0, 2.5),
            (1.5, 2.5),   # North center
            (1.0, 2.5),
            (0.5, 2.5),   # Corner 3
            (0.5, 2.0),
            (0.5, 1.5),   # West center
            (0.5, 1.0),
            (0.5, 0.5),   # Corner 4
            (1.0, 0.5),
        ]
        
        # Check if we reached the next checkpoint or skipped to the one after it
        checkpoint_reached = False
        for lookahead in [1, 2]:
            idx = (self.current_checkpoint_idx + lookahead) % len(checkpoints)
            target_cp = checkpoints[idx]
            dist_to_cp = math.hypot(rx - target_cp[0], ry - target_cp[1])
            if dist_to_cp < 0.55:  # Increased from 0.35 to 0.55 to be robust against wall-hugging/swerving
                # Check if we crossed/landed on index 0 (lap completion)
                crossed_zero = False
                for i in range(1, lookahead + 1):
                    check_idx = (self.current_checkpoint_idx + i) % len(checkpoints)
                    if check_idx == 0:
                        crossed_zero = True
                
                self.current_checkpoint_idx = idx
                reward += 3.0 * lookahead  # Reward proportional to checkpoints cleared
                self.steps_since_last_checkpoint = 0
                checkpoint_reached = True
                
                if crossed_zero:
                    reward += 10.0
                    print("[Gym Env] LAP completed! Bonus reward +10.0")
                break
        
        if not checkpoint_reached:
            self.steps_since_last_checkpoint += 1
            
        # Stagnation check (Option C): truncate if checkpoint not reached in 12 seconds (120 steps)
        if self.steps_since_last_checkpoint > 120:
            reward -= 5.0
            truncated = True
            print("[Gym Env] TIMEOUT (No progress)! Resetting.")
            
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
                        reward += 5.0
                        self.steps_since_last_checkpoint = 0  # Reset stagnation steps on correct obstacle pass
                        print(f"[Gym Env] RED Obstacle passed CORRECTLY! Reward +5.0")
                    else:
                        reward -= 5.0
                        terminated = True
                        print(f"[Gym Env] RED Obstacle passed INCORRECTLY! Resetting.")
                elif obstacle.color == "green":
                    # Green obstacle: must pass on the left (obstacle on right side, y_loc < 0)
                    if y_loc < 0:
                        reward += 5.0
                        self.steps_since_last_checkpoint = 0  # Reset stagnation steps on correct obstacle pass
                        print(f"[Gym Env] GREEN Obstacle passed CORRECTLY! Reward +5.0")
                    else:
                        reward -= 5.0
                        terminated = True
                        print(f"[Gym Env] GREEN Obstacle passed INCORRECTLY! Resetting.")
                        
        # Small survival reward to encourage keeping speed up
        reward += 0.05
        
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
        checkpoints = [
            (1.28, 0.49), (1.6, 0.5), (2.0, 0.5), (2.5, 0.5), (2.5, 1.0),
            (2.5, 1.5), (2.5, 2.0), (2.5, 2.5), (2.0, 2.5), (1.5, 2.5),
            (1.0, 2.5), (0.5, 2.5), (0.5, 2.0), (0.5, 1.5), (0.5, 1.0),
            (0.5, 0.5), (1.0, 0.5)
        ]
        next_idx = (self.current_checkpoint_idx + 1) % len(checkpoints)
        cx, cy = checkpoints[next_idx]
        cx_px = int(cx * self.icp_localizer.scale)
        cy_px = int(self.icp_localizer.window_size - cy * self.icp_localizer.scale)
        cv2.circle(vis_img, (cx_px, cy_px), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA) # Yellow checkpoint indicator
        
        cv2.imshow("RL Training Environment", vis_img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
