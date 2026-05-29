import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import cv2
import os
import sys

# Ensure wro_core is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from wro_core import config, perception, estimation, planning, control
from wro_core.obs_visualizer import draw_observation_window
from wro_core.obstacle_randomizer import randomize_obstacles
from wro_core import geometry

try:
    from controller import Supervisor
except ModuleNotFoundError:
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
        
        # Observation space (12 elements, normalized to [-1.0, 1.0]):
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
        
        # State estimation and control helper modules
        self.estimator = estimation.StateEstimator()
        self.car_controller = control.Controller()
        
        # Timing
        self.frame_skip = int(1000 / (config.CONTROL_FREQ * self.timestep))
        if self.frame_skip < 1:
            self.frame_skip = 1
            
        # State tracking variables
        self.imu_yaw_initial = None
        self.stagnation_counter = 0
        self.current_checkpoint_idx = 0
        self.total_steps = 0
        self.passed_obstacle_ids = set()
        self.checkpoints_cleared_this_lap = 0
        
        # Variables stored for visualization/rendering
        self.raw_obs = np.zeros(12, dtype=np.float32)
        self.obs_vector = np.zeros(12, dtype=np.float32)
        self.best_closest = None
        self.p_30 = None
        self.p_60 = None
        self.rx = 1.5
        self.ry = 0.5
        self.ryaw = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Stop robot
        self.motor_right.setVelocity(0.0)
        self.motor_left.setVelocity(0.0)
        self.steer_left.setPosition(0.0)
        self.steer_right.setPosition(0.0)
        
        # Random start pose
        start_x_est = self.np_random.uniform(1.2, 1.8)
        start_y_est = self.np_random.uniform(0.2, 0.8)
        yaw_est = self.np_random.uniform(-np.pi, np.pi)
        
        # Convert to Webots coordinates
        x_webots = start_x_est - 1.5
        y_webots = start_y_est - 1.5
        z_webots = 0.030000
        angle_webots = yaw_est + math.pi / 2.0
        
        # Teleport robot
        self.translation_field.setSFVec3f([x_webots, y_webots, z_webots])
        self.rotation_field.setSFRotation([0.0, 0.0, 1.0, angle_webots])
        
        # Reset obstacles
        randomize_obstacles(self.supervisor, train=True, seed=self.np_random)
        
        # Reset physics
        self.supervisor.simulationResetPhysics()
        
        # Step simulation to apply reset
        for _ in range(5):
            self.supervisor.step(self.timestep)
            
        # Reset state estimators & controllers
        self.imu_yaw_initial = None
        self.car_controller.reset()
        self.estimator.reset()
        
        # Calibrate starting position
        collected_scans = []
        import warnings
        
        for _ in range(10):
            if self.supervisor.step(self.timestep) == -1:
                break
            sensor_data, self.imu_yaw_initial = perception.read_sensors(
                self.lidar, self.imu, self.camera, self.imu_yaw_initial
            )
            lidar_ranges = sensor_data["lidar_ranges"]
            if len(lidar_ranges) > 0:
                collected_scans.append(lidar_ranges)
                
        calibrated = False
        if len(collected_scans) >= 10:
            scans_arr = np.array(collected_scans)
            invalid_mask = (scans_arr <= 0.01) | (scans_arr >= 2.0) | np.isinf(scans_arr) | np.isnan(scans_arr)
            scans_arr[invalid_mask] = np.nan
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_ranges = np.nanmean(scans_arr, axis=0)
            avg_ranges = np.nan_to_num(avg_ranges, nan=0.0)
            
            try:
                x_init, y_init, yaw_init, direction, debug_img = self.estimator.calibrate_from_scans(avg_ranges)
                print(f"[Gym Env Calibration] Resolved driving direction: {direction}")
                self.estimator.set_calibrated_pose(x_init, y_init, yaw_init, direction)
                calibrated = True
            except Exception as e:
                print(f"[Gym Env Calibration] Error during calibration: {e}")
                
        if not calibrated:
            # Fallback based on generated yaw
            sin_yaw = math.sin(yaw_est)
            direction = "CCW" if (sin_yaw < 0 or (abs(sin_yaw) < 1e-5 and yaw_est < 0)) else "CW"
            self.estimator.set_calibrated_pose(start_x_est, start_y_est, yaw_est, direction)
            
        self.passed_obstacle_ids = set()
        self.stagnation_counter = 0
        self.checkpoints_cleared_this_lap = 0
        
        # Nearest checkpoint index
        dists = [math.hypot(start_x_est - cp[0], start_y_est - cp[1]) for cp in config.CHECKPOINTS]
        self.current_checkpoint_idx = int(np.argmin(dists))
        
        self.total_steps = 0
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # 1. Perception
        sensor_data, self.imu_yaw_initial = perception.read_sensors(
            self.lidar, self.imu, self.camera, self.imu_yaw_initial
        )
        
        # 2. Estimation
        rx, ry, ryaw = self.estimator.update(sensor_data)
        
        # Get velocity
        vel = self.robot_node.getVelocity()
        if vel is not None:
            v_g = vel[:3]
            R = self.robot_node.getOrientation()
            ego_v_x = float(v_g[0] * R[0] + v_g[1] * R[3] + v_g[2] * R[6]) if R is not None else 0.0
        else:
            ego_v_x = 0.0
            
        # 3. Planning (observation computation)
        self.raw_obs, self.obs_vector, self.best_closest, self.p_30, self.p_60 = planning.compute_observation_vector(
            pose=(rx, ry, ryaw),
            obstacles=self.estimator.obstacle_mapper.obstacles,
            driving_direction=self.estimator.driving_direction,
            smoothed_steering=self.car_controller.smoothed_steering,
            ego_v_x=ego_v_x
        )
        self.rx, self.ry, self.ryaw = rx, ry, ryaw
        
        return self.obs_vector

    def step(self, action):
        # 1. Apply action
        target_steering = float(action[0]) * config.MAX_STEERING
        target_speed = float(action[1]) * config.MAX_MOTOR_VELOCITY
        
        # 4. Control
        self.car_controller.apply(
            target_speed=target_speed,
            target_steering=target_steering,
            motor_left=self.motor_left,
            motor_right=self.motor_right,
            steer_left=self.steer_left,
            steer_right=self.steer_right,
            use_rl=True
        )
        
        # Step simulation physics
        for _ in range(self.frame_skip):
            if self.supervisor.step(self.timestep) == -1:
                break
                
        # 3. Get next observation
        obs = self._get_obs()
        
        # 4. Compute reward & check termination
        rx, ry, ryaw = self.rx, self.ry, self.ryaw
        
        # Calculate lateral error for reward
        best_closest, best_dist, best_tangent, best_segment_idx, best_t = geometry.get_closest_point_on_path(
            rx, ry, self.estimator.driving_direction
        )
        left_normal = np.array([-best_tangent[1], best_tangent[0]])
        p = np.array([rx, ry])
        lateral_error = np.dot(p - best_closest, left_normal)
        lateral_error_normalized = np.clip(lateral_error / 0.5, -1.0, 1.0)
        
        # Velocity
        vel = self.robot_node.getVelocity()
        if vel is not None:
            v_g = vel[:3]
            R = self.robot_node.getOrientation()
            ego_v_x = float(v_g[0] * R[0] + v_g[1] * R[3] + v_g[2] * R[6]) if R is not None else 0.0
        else:
            ego_v_x = 0.0
            
        reward = ego_v_x * 1.0 - abs(lateral_error_normalized) * 0.1
        
        terminated = False
        truncated = False
        
        # Collision detection via contact points
        collision = False
        try:
            contact_points = self.robot_node.getContactPoints(includeDescendants=True)
            for cp in contact_points:
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
        step_dir = 1 if self.estimator.driving_direction == "CCW" else -1
        for lookahead in [1, 2]:
            idx = (self.current_checkpoint_idx + lookahead * step_dir) % len(config.CHECKPOINTS)
            target_cp = config.CHECKPOINTS[idx]
            dist_to_cp = math.hypot(rx - target_cp[0], ry - target_cp[1])
            if dist_to_cp < 0.55:
                crossed_zero = False
                for i in range(1, lookahead + 1):
                    check_idx = (self.current_checkpoint_idx + i * step_dir) % len(config.CHECKPOINTS)
                    if check_idx == 0:
                        crossed_zero = True
                        
                self.current_checkpoint_idx = idx
                self.checkpoints_cleared_this_lap += lookahead
                
                if crossed_zero:
                    if self.checkpoints_cleared_this_lap >= len(config.CHECKPOINTS) - 2:
                        terminated = True
                        print(f"[Gym Env] LAP completed ({self.estimator.driving_direction})! Terminating episode.")
                    self.checkpoints_cleared_this_lap = 0
                break
                
        # Stagnation check
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
        
        for obstacle in self.estimator.obstacle_mapper.obstacles:
            if obstacle.color not in ["red", "green"]:
                continue
                
            obs_key = (round(obstacle.position[0], 1), round(obstacle.position[1], 1))
            if obs_key in self.passed_obstacle_ids:
                continue
                
            ox, oy = obstacle.position
            dx = ox - rx
            dy = oy - ry
            x_loc = dx * cos_a + dy * sin_a
            y_loc = -dx * sin_a + dy * cos_a
            
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
        vis_img = self.estimator.icp_localizer.render()
        vis_img = self.estimator.obstacle_mapper.render(
            vis_img, [self.rx, self.ry, self.ryaw], self.estimator.icp_localizer.scale, self.estimator.icp_localizer.window_size
        )
        
        # Draw next checkpoint
        step_dir = 1 if self.estimator.driving_direction == "CCW" else -1
        next_idx = (self.current_checkpoint_idx + step_dir) % len(config.CHECKPOINTS)
        cx, cy = config.CHECKPOINTS[next_idx]
        cx_px = int(cx * self.estimator.icp_localizer.scale)
        cy_px = int(self.estimator.icp_localizer.window_size - cy * self.estimator.icp_localizer.scale)
        cv2.circle(vis_img, (cx_px, cy_px), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        
        draw_observation_window(
            pose=(self.rx, self.ry, self.ryaw),
            raw_obs=self.raw_obs,
            obs_vector=self.obs_vector,
            driving_direction=self.estimator.driving_direction,
            best_closest=self.best_closest,
            p_30=self.p_30,
            p_60=self.p_60,
            window_name="WRO Observation Debug"
        )
            
        cv2.imshow("RL Training Environment", vis_img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
