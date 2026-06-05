import os
import math
import numpy as np

from . import config
from . import geometry

def compute_observation_vector(pose, obstacles, driving_direction, smoothed_steering, ego_v_x):
    """
    Computes the 14-dimensional observation vector for the WRO agent.
    
    Uses inner/outer track boundary lookahead points instead of centerline
    deviation, giving the agent direct knowledge of track boundaries.
    
    Returns:
        raw_obs (np.ndarray): Unnormalized features (14 elements).
        obs_vector (np.ndarray): Normalized and clipped features ([-1.0, 1.0]).
        best_closest (np.ndarray): Nearest point on centerline (for visualization).
        lateral_error (float): Signed lateral distance to centerline (for reward only).
        s_current (float): Current arc-length position on centerline.
        p_inner_40 (np.ndarray): Inner boundary point at 40cm lookahead.
        p_inner_100 (np.ndarray): Inner boundary point at 100cm lookahead.
        p_outer_40 (np.ndarray): Outer boundary point at 40cm lookahead.
        p_outer_100 (np.ndarray): Outer boundary point at 100cm lookahead.
        corner_global (np.ndarray): Next inner corner position.
    """
    rx, ry, ryaw = pose
    
    # 1. Get current position on centerline (needed for obstacle s-distance, progress, and reward)
    best_closest, best_dist, best_tangent, best_segment_idx, best_t = geometry.get_closest_point_on_path(rx, ry, driving_direction)
    s_current = best_segment_idx * 2.0 + best_t * 2.0
    total_len = 4.0 + math.pi
    
    # Compute signed lateral error (for reward only, not in obs vector)
    left_normal = np.array([-best_tangent[1], best_tangent[0]])
    p = np.array([rx, ry])
    lateral_error = np.dot(p - best_closest, left_normal)
    
    # Compute heading error (diff_yaw) relative to the centerline tangent
    theta_robot = math.atan2(math.cos(ryaw), -math.sin(ryaw))
    theta_tangent = math.atan2(best_tangent[1], best_tangent[0])
    diff_yaw = theta_robot - theta_tangent
    diff_yaw = (diff_yaw + math.pi) % (2.0 * math.pi) - math.pi
    
    # 2. Boundary lookahead points (40cm and 100cm ahead on centerline)
    s_40 = (s_current + 0.4) % total_len
    s_100 = (s_current + 1.0) % total_len
    
    p_inner_40, p_outer_40 = geometry.get_boundary_points_at_s(s_40, driving_direction)
    p_inner_100, p_outer_100 = geometry.get_boundary_points_at_s(s_100, driving_direction)
    
    # Transform boundary points to local coordinates (extract y component)
    alpha = ryaw + math.pi / 2.0
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    
    def to_local_y(gx, gy):
        dx = gx - rx
        dy = gy - ry
        return -dx * sin_a + dy * cos_a
    
    inner_y_40 = to_local_y(p_inner_40[0], p_inner_40[1])
    inner_y_100 = to_local_y(p_inner_100[0], p_inner_100[1])
    outer_y_40 = to_local_y(p_outer_40[0], p_outer_40[1])
    outer_y_100 = to_local_y(p_outer_100[0], p_outer_100[1])
    
    # 3. Next Corner Position (Eck-Beacon)
    corner_global = geometry.get_next_inner_corner(s_current, driving_direction)
    dx_c = corner_global[0] - rx
    dy_c = corner_global[1] - ry
    corner_x_loc = dx_c * cos_a + dy_c * sin_a
    corner_y_loc = -dx_c * sin_a + dy_c * cos_a
    
    # 4. Obstacles
    valid_obstacles = []
    for obs in obstacles:
        ox, oy = obs.position
        
        # Project obstacle position onto the centerline path
        _, obs_path_dist, _, _, obs_t = geometry.get_closest_point_on_path(ox, oy, driving_direction)
        s_obs = obs_t * 2.0
        
        # Calculate signed path distance from robot to obstacle
        s_dist = s_obs - s_current
        if s_dist > total_len / 2.0:
            s_dist -= total_len
        elif s_dist < -total_len / 2.0:
            s_dist += total_len
            
        dx = ox - rx
        dy = oy - ry
        x_loc = dx * cos_a + dy * sin_a
        y_loc = -dx * sin_a + dy * cos_a
        
        # Filter:
        # - Obstacle is on our track corridor (within 0.8m of centerline)
        # - Obstacle is ahead of the robot along the path (0.0 < s_dist <= 2.2m)
        # - Obstacle is in front of the robot locally (x_loc > 0.0)
        if obs_path_dist <= 0.8 and 0.0 < s_dist <= 2.2 and x_loc > 0.0:
            color_val = 1.0 if obs.color == "green" else -1.0 if obs.color == "red" else 0.0
            valid_obstacles.append({
                "x_loc": x_loc,
                "y_loc": y_loc,
                "color": color_val,
                "s_dist": s_dist
            })
            
    # Sort by path distance (s_dist) ascending
    valid_obstacles.sort(key=lambda item: item["s_dist"])
    
    # Build raw observation vector (15 elements)
    raw_obs = np.array([
        ego_v_x,
        smoothed_steering,
        diff_yaw,
        inner_y_40,
        inner_y_100,
        outer_y_40,
        outer_y_100,
        corner_x_loc,
        corner_y_loc,
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
    
    return raw_obs, obs_vector, best_closest, lateral_error, s_current, p_inner_40, p_inner_100, p_outer_40, p_outer_100, corner_global


class RLPlanner:
    def __init__(self, model_path):
        self.ort_session = None
        self.current_steering = 0.0
        self.current_speed = 0.0
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
        
    def reset(self):
        self.current_steering = 0.0
        self.current_speed = 0.0
        
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
        act_steer_delta = np.clip(action[0], -1.0, 1.0)
        act_speed_delta = np.clip(action[1], -1.0, 1.0)
        
        # Accumulate steering: max change 0.2 rad per step
        self.current_steering += act_steer_delta * 0.2
        self.current_steering = np.clip(self.current_steering, -config.MAX_STEERING, config.MAX_STEERING)
        
        # Accumulate speed: max change 10.0 rad/s per step
        self.current_speed += act_speed_delta * 10.0
        self.current_speed = np.clip(self.current_speed, 0.0, config.MAX_MOTOR_VELOCITY)
        
        return self.current_speed, self.current_steering
