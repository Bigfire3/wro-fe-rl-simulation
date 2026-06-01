import numpy as np
import math

def get_track_vertices(direction):
    if direction == "CCW":
        return [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]
    else:
        return [(2.5, 0.5), (0.5, 0.5), (0.5, 2.5), (2.5, 2.5)]

def get_closest_point_on_path(rx, ry, direction):
    # Center of circle
    cx, cy = 1.5, 1.5
    R = 1.0
    
    dx = rx - cx
    dy = ry - cy
    dist_to_center = math.hypot(dx, dy)
    
    if dist_to_center < 1e-9:
        ux, uy = 1.0, 0.0
        dist_to_center = 0.0
    else:
        ux, uy = dx / dist_to_center, dy / dist_to_center
        
    best_closest = np.array([cx + R * ux, cy + R * uy])
    best_dist = abs(dist_to_center - R)
    
    theta = math.atan2(uy, ux)
    start_theta = -math.pi / 2.0
    
    if direction == "CCW":
        best_tangent = np.array([-math.sin(theta), math.cos(theta)])
        angle_diff = (theta - start_theta) % (2.0 * math.pi)
    else:
        best_tangent = np.array([math.sin(theta), -math.cos(theta)])
        angle_diff = (start_theta - theta) % (2.0 * math.pi)
        
    s_current = angle_diff * R
    # Return best_segment_idx = 0 and best_t = s_current / 2.0 so that
    # s_current = best_segment_idx * 2.0 + best_t * 2.0 holds.
    best_segment_idx = 0
    best_t = s_current / 2.0
    
    return best_closest, best_dist, best_tangent, best_segment_idx, best_t

def get_point_at_s(s, direction):
    cx, cy = 1.5, 1.5
    R = 1.0
    
    s = s % (2.0 * math.pi * R)
    angle_diff = s / R
    
    start_theta = -math.pi / 2.0
    if direction == "CCW":
        theta = start_theta + angle_diff
    else:
        theta = start_theta - angle_diff
        
    p_s = np.array([cx + R * math.cos(theta), cy + R * math.sin(theta)])
    return p_s
