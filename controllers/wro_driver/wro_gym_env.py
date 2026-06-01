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
        
        # Observation space (15 elements, normalized to [-1.0, 1.0]):
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 15, dtype=np.float32),
            high=np.array([1.0] * 15, dtype=np.float32),
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
        
        # Get actual max velocity from the motor device to dynamically scale speed limit
        self.max_motor_vel = self.motor_left.getMaxVelocity()
        config.MAX_MOTOR_VELOCITY = self.max_motor_vel
        self.max_linear_velocity = self.max_motor_vel * config.WHEEL_RADIUS
        config.NORM_FACTORS[0] = 1.0 / self.max_linear_velocity
        
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
        self.total_steps = 0
        self.passed_obstacle_ids = set()
        self.curriculum_stage = 1
        self.prev_path_s = 0.0
        self.current_path_s = 0.0
        self.prev_steer_action = 0.0
        self.cumulative_progress = 0.0
        
        # Variables stored for visualization/rendering
        self.raw_obs = np.zeros(15, dtype=np.float32)
        self.obs_vector = np.zeros(15, dtype=np.float32)
        self.best_closest = None
        self.p_40 = None
        self.p_80 = None
        self.p_150 = None
        self.p_corner = None
        self.rx = 1.5
        self.ry = 0.5
        self.ryaw = 0.0
        self.camera_image = None

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
        if self.curriculum_stage == 1:
            base_yaw = self.np_random.choice([-np.pi / 2.0, np.pi / 2.0])
            yaw_est = base_yaw + self.np_random.uniform(-0.1, 0.1)
        else:
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
        calibrated = False
        for _ in range(10):
            if self.supervisor.step(self.timestep) == -1:
                break
            sensor_data, self.imu_yaw_initial = perception.read_sensors(
                self.lidar, self.imu, self.camera, self.imu_yaw_initial
            )
            lidar_ranges = sensor_data["lidar_ranges"]
            try:
                res = self.estimator.add_calibration_scan(lidar_ranges)
                if res is not None:
                    _, _, _, direction, _ = res
                    print(f"[Gym Env Calibration] Resolved driving direction: {direction}")
                    calibrated = True
                    break
            except Exception as e:
                print(f"[Gym Env Calibration] Error during calibration: {e}")
                break
                
        if not calibrated:
            # Fallback based on generated yaw
            sin_yaw = math.sin(yaw_est)
            direction = "CCW" if (sin_yaw < 0 or (abs(sin_yaw) < 1e-5 and yaw_est < 0)) else "CW"
            self.estimator.set_calibrated_pose(start_x_est, start_y_est, yaw_est, direction)
            
        self.passed_obstacle_ids = set()
        self.stagnation_counter = 0
        self.prev_steer_action = 0.0
        self.cumulative_progress = 0.0
        self.total_steps = 0
        
        obs = self._get_obs()
        self.prev_path_s = self.current_path_s
        return obs, {}

    def _get_obs(self):
        # 1. Perception
        sensor_data, self.imu_yaw_initial = perception.read_sensors(
            self.lidar, self.imu, self.camera, self.imu_yaw_initial
        )
        self.camera_image = sensor_data.get("camera_image")
        
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
        self.raw_obs, self.obs_vector, self.best_closest, self.p_40, self.p_80, self.p_150, self.p_corner = planning.compute_observation_vector(
            pose=(rx, ry, ryaw),
            obstacles=self.estimator.obstacle_mapper.obstacles,
            driving_direction=self.estimator.driving_direction,
            smoothed_steering=self.car_controller.smoothed_steering,
            ego_v_x=ego_v_x
        )
        self.rx, self.ry, self.ryaw = rx, ry, ryaw
        
        # Track path progress s
        from wro_core import geometry
        best_closest, best_dist, best_tangent, best_segment_idx, best_t = geometry.get_closest_point_on_path(
            rx, ry, self.estimator.driving_direction
        )
        self.current_path_s = best_segment_idx * 2.0 + best_t * 2.0
        
        return self.obs_vector

    def step(self, action):
        # 1. Apply action
        target_steering = float(action[0]) * config.MAX_STEERING
        if self.curriculum_stage == 1:
            # Stage 1: fixed target speed (30% of max motor velocity, i.e., 3.0 rad/s)
            target_speed = 0.3 * config.MAX_MOTOR_VELOCITY
        else:
            # Stage 2: variable speed control
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
        
        # Extract normalized values directly from computed observation vector
        speed_ratio = float(obs[0])
        ego_v_x = speed_ratio * self.max_linear_velocity
        lateral_error_normalized = float(obs[2])
        heading_error_normalized = float(obs[3])
        
        # Step penalty
        step_penalty = -0.3
        
        # Smoothness penalty
        steer_diff = float(action[0]) - self.prev_steer_action
        self.prev_steer_action = float(action[0])
        smoothness_penalty = -0.15 * (steer_diff ** 2)
        
        # Compute progress s diff
        total_len = 4.0 + math.pi
        diff_s = self.current_path_s - self.prev_path_s
        if diff_s < -total_len / 2.0:
            diff_s += total_len
        elif diff_s > total_len / 2.0:
            diff_s -= total_len
        self.prev_path_s = self.current_path_s
        
        # Accumulate progress
        self.cumulative_progress += diff_s
        
        if self.curriculum_stage == 1:
            # Stage 1: Safety focus (high lateral penalty, standard speed reward)
            lateral_penalty = -abs(lateral_error_normalized) * 1.5
            heading_penalty = -abs(heading_error_normalized) * 0.5
            speed_reward = (max(0.0, speed_ratio) ** 2) * 5.0
            
            reward = speed_reward + step_penalty + lateral_penalty + heading_penalty + smoothness_penalty
        else:
            # Stage 2: Performance focus (progress-based reward, relaxed lateral penalty)
            progress_reward = diff_s * 15.0
            
            lat_err_abs = abs(lateral_error_normalized)
            if lat_err_abs <= 0.5: # within +/- 25cm
                lateral_penalty = 0.0
            else:
                lateral_penalty = -3.0 * ((lat_err_abs - 0.5) ** 2)
                
            heading_penalty = 0.0
            action_steer_penalty = -0.5 * abs(float(action[0])) * max(0.0, speed_ratio)
            
            reward = progress_reward + step_penalty + lateral_penalty + heading_penalty + smoothness_penalty + action_steer_penalty
        
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
            reward = -50.0
            terminated = True
            print("[Gym Env] COLLISION detected! Resetting.")
            
        # Lap completion check based on cumulative progress
        if self.cumulative_progress >= (total_len - 0.2):
            terminated = True
            print(f"[Gym Env] LAP completed ({self.estimator.driving_direction})! Terminating episode.")
                
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
                        reward = -50.0
                        terminated = True
                        print(f"[Gym Env] RED Obstacle passed INCORRECTLY! Resetting.")
                elif obstacle.color == "green":
                    # Green obstacle: must pass on the left (obstacle on right side, y_loc < 0)
                    if y_loc < 0:
                        reward += 15.0
                        print(f"[Gym Env] GREEN Obstacle passed CORRECTLY! Reward +15.0")
                    else:
                        reward = -50.0
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
            "progress": self.cumulative_progress
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        vis_img = self.estimator.icp_localizer.render()
        vis_img = self.estimator.obstacle_mapper.render(
            vis_img, [self.rx, self.ry, self.ryaw], self.estimator.icp_localizer.scale, self.estimator.icp_localizer.window_size
        )
        
        draw_observation_window(
            pose=(self.rx, self.ry, self.ryaw),
            raw_obs=self.raw_obs,
            obs_vector=self.obs_vector,
            driving_direction=self.estimator.driving_direction,
            best_closest=self.best_closest,
            p_40=self.p_40,
            p_80=self.p_80,
            p_150=self.p_150,
            p_corner=self.p_corner,
            window_name="WRO Observation Debug"
        )
        
        if self.camera_image is not None:
            cam_debug = self.estimator.obstacle_mapper.render_camera(
                self.camera_image, [self.rx, self.ry, self.ryaw]
            )
            try:
                cv2.imshow("WRO Camera Debug", cam_debug)
            except Exception:
                pass
            
        cv2.imshow("RL Training Environment", vis_img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
