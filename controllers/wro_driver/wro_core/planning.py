import os
import math
import numpy as np

from . import config
from . import geometry

def compute_observation_vector(pose, obstacles, driving_direction, smoothed_steering, ego_v_x):
    """
    Computes the standard 12-dimensional observation vector for the WRO agent.
    
    Returns:
        raw_obs (np.ndarray): Unnormalized features.
        obs_vector (np.ndarray): Normalized and clipped features ([-1.0, 1.0]).
        best_closest (np.ndarray): Nearest point on path coordinates (for visualization/rewards).
        p_40 (np.ndarray): Coordinates of the 40cm lookahead point.
        p_80 (np.ndarray): Coordinates of the 80cm lookahead point.
    """
    rx, ry, ryaw = pose
    
    # 1. lateral_error
    best_closest, best_dist, best_tangent, best_segment_idx, best_t = geometry.get_closest_point_on_path(rx, ry, driving_direction)
    left_normal = np.array([-best_tangent[1], best_tangent[0]])
    p = np.array([rx, ry])
    lateral_error = np.dot(p - best_closest, left_normal)
    
    # 2. heading_error (repaired)
    theta_robot = math.atan2(math.cos(ryaw), -math.sin(ryaw))
    theta_tangent = math.atan2(best_tangent[1], best_tangent[0])
    diff_yaw = theta_robot - theta_tangent
    diff_yaw = (diff_yaw + math.pi) % (2.0 * math.pi) - math.pi
    
    # 3. lookahead points (40cm & 80cm)
    s_current = best_segment_idx * 2.0 + best_t * 2.0
    
    s_40 = (s_current + 0.4) % (2.0 * math.pi)
    p_40 = geometry.get_point_at_s(s_40, driving_direction)
    
    s_80 = (s_current + 0.8) % (2.0 * math.pi)
    p_80 = geometry.get_point_at_s(s_80, driving_direction)
    
    # Transform lookahead points to local coordinates
    alpha = ryaw + math.pi / 2.0
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    
    dx_40 = p_40[0] - rx
    dy_40 = p_40[1] - ry
    y_loc_40 = -dx_40 * sin_a + dy_40 * cos_a
    
    dx_80 = p_80[0] - rx
    dy_80 = p_80[1] - ry
    y_loc_80 = -dx_80 * sin_a + dy_80 * cos_a
    
    # 4. Obstacles
    valid_obstacles = []
    for obs in obstacles:
        ox, oy = obs.position
        dx = ox - rx
        dy = oy - ry
        x_loc = dx * cos_a + dy * sin_a
        y_loc = -dx * sin_a + dy * cos_a
        
        if x_loc > 0.0 and abs(y_loc) <= 1.0:
            color_val = 1.0 if obs.color == "green" else -1.0 if obs.color == "red" else 0.0
            valid_obstacles.append({
                "x_loc": x_loc,
                "y_loc": y_loc,
                "color": color_val,
                "dist": math.hypot(x_loc, y_loc)
            })
            
    # Sort by total Euclidean distance ascending
    valid_obstacles.sort(key=lambda item: item["dist"])
    
    # Build raw observation vector
    raw_obs = np.array([
        ego_v_x,
        smoothed_steering,
        lateral_error,
        diff_yaw,
        y_loc_40,
        y_loc_80,
        # Obstacle 1
        valid_obstacles[0]["x_loc"] if len(valid_obstacles) > 0 else 2.0,
        valid_obstacles[0]["y_loc"] if len(valid_obstacles) > 0 else 0.0,
        valid_obstacles[0]["color"] if len(valid_obstacles) > 0 else 0.0,
        # Obstacle 2
        valid_obstacles[1]["x_loc"] if len(valid_obstacles) > 1 else 2.0,
        valid_obstacles[1]["y_loc"] if len(valid_obstacles) > 1 else 0.0,
        valid_obstacles[1]["color"] if len(valid_obstacles) > 1 else 0.0
    ], dtype=np.float32)
    
    obs_vector = np.clip(raw_obs * config.NORM_FACTORS, -1.0, 1.0)
    
    return raw_obs, obs_vector, best_closest, p_40, p_80

class RuleBasedPlanner:
    def __init__(self):
        self.prev_error = 0.0
        
    def reset(self):
        self.prev_error = 0.0
        
    def plan(self, lidar_data):
        """
        Executes rules-based path planning.
        Returns target_speed, target_steering.
        """
        if len(lidar_data) == 0:
            return 0.0, 0.0
            
        target_speed = config.TARGET_SPEED_RULES
        
        def cap(dist):
            return min(dist, 1.5)
            
        # Distances: 90=Left, 270=Right, 180=Front, 135=Front-Left, 225=Front-Right
        right_dist = cap(lidar_data[270])
        left_dist = cap(lidar_data[90])
        front_dist = lidar_data[180]
        front_right_dist = cap(lidar_data[225])
        front_left_dist = cap(lidar_data[135])
        
        # 1. centering & parallel driving (PD controller)
        current_diff = left_dist - right_dist
        derivative = current_diff - self.prev_error
        self.prev_error = current_diff
        
        raw_steering = -(config.P_GAIN * current_diff + config.D_GAIN * derivative)
        
        # 2. sidewall protection (safety margin 20cm)
        min_side = min(left_dist, right_dist)
        if min_side < 0.2:
            safety_force = (0.2 - min_side) * 5.0
            if left_dist < right_dist:
                raw_steering += safety_force  # steer right
            else:
                raw_steering -= safety_force  # steer left
                
        # 3. curves / front wall avoidance
        if front_dist < 1.0:
            corner_force = (1.0 - front_dist)**2
            CORNER_GAIN = 8.0
            
            if front_left_dist > front_right_dist:
                raw_steering -= (corner_force * CORNER_GAIN)   # steer left
            else:
                raw_steering += (corner_force * CORNER_GAIN)   # steer right
                
        target_steering = np.clip(raw_steering, -config.MAX_STEERING, config.MAX_STEERING)
        return target_speed, target_steering

class RLPlanner:
    def __init__(self, model_path):
        self.ort_session = None
        try:
            import onnxruntime as ort
            if os.path.exists(model_path):
                print(f"[RLPlanner] Loading ONNX model from {model_path}...")
                self.ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                print("[RLPlanner] ONNX model loaded successfully!")
            else:
                print(f"[RLPlanner Warning] Model file not found at {model_path}.")
        except Exception as e:
            print(f"[RLPlanner Warning] Failed to load ONNX runtime/model: {e}")
            
    def is_ready(self):
        return self.ort_session is not None
        
    def plan(self, obs_vector):
        """
        Runs ONNX inference and returns (target_speed, target_steering).
        """
        if self.ort_session is None:
            raise RuntimeError("RLPlanner ONNX session is not loaded.")
            
        obs_input = np.array([obs_vector], dtype=np.float32)
        ort_inputs = {self.ort_session.get_inputs()[0].name: obs_input}
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        # ONNX policy outputs (action, value, log_prob)
        action = ort_outs[0][0]
        act_steer = np.clip(action[0], -1.0, 1.0)
        act_speed = np.clip(action[1], 0.0, 1.0)
        
        target_steering = act_steer * config.MAX_STEERING
        target_speed = act_speed * config.MAX_MOTOR_VELOCITY
        
        return target_speed, target_steering
