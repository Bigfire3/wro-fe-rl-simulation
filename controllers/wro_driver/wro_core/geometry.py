import numpy as np
import math

def get_track_vertices(direction):
    if direction == "CCW":
        return [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]
    else:
        return [(2.5, 0.5), (0.5, 0.5), (0.5, 2.5), (2.5, 2.5)]

def get_path_segments(direction):
    if direction == "CCW":
        return [
            # Segment 0: Line bottom
            {"type": "line", "p1": np.array([1.0, 0.5]), "p2": np.array([2.0, 0.5]), "length": 1.0},
            # Segment 1: Arc bottom-right
            {"type": "arc", "center": np.array([2.0, 1.0]), "radius": 0.5, "start_angle": -math.pi/2.0, "span": math.pi/2.0, "length": math.pi * 0.25},
            # Segment 2: Line right
            {"type": "line", "p1": np.array([2.5, 1.0]), "p2": np.array([2.5, 2.0]), "length": 1.0},
            # Segment 3: Arc top-right
            {"type": "arc", "center": np.array([2.0, 2.0]), "radius": 0.5, "start_angle": 0.0, "span": math.pi/2.0, "length": math.pi * 0.25},
            # Segment 4: Line top
            {"type": "line", "p1": np.array([2.0, 2.5]), "p2": np.array([1.0, 2.5]), "length": 1.0},
            # Segment 5: Arc top-left
            {"type": "arc", "center": np.array([1.0, 2.0]), "radius": 0.5, "start_angle": math.pi/2.0, "span": math.pi/2.0, "length": math.pi * 0.25},
            # Segment 6: Line left
            {"type": "line", "p1": np.array([0.5, 2.0]), "p2": np.array([0.5, 1.0]), "length": 1.0},
            # Segment 7: Arc bottom-left
            {"type": "arc", "center": np.array([1.0, 1.0]), "radius": 0.5, "start_angle": math.pi, "span": math.pi/2.0, "length": math.pi * 0.25},
        ]
    else: # CW
        return [
            # Segment 0: Line bottom
            {"type": "line", "p1": np.array([2.0, 0.5]), "p2": np.array([1.0, 0.5]), "length": 1.0},
            # Segment 1: Arc bottom-left
            {"type": "arc", "center": np.array([1.0, 1.0]), "radius": 0.5, "start_angle": -math.pi/2.0, "span": -math.pi/2.0, "length": math.pi * 0.25},
            # Segment 2: Line left
            {"type": "line", "p1": np.array([0.5, 1.0]), "p2": np.array([0.5, 2.0]), "length": 1.0},
            # Segment 3: Arc top-left
            {"type": "arc", "center": np.array([1.0, 2.0]), "radius": 0.5, "start_angle": math.pi, "span": -math.pi/2.0, "length": math.pi * 0.25},
            # Segment 4: Line top
            {"type": "line", "p1": np.array([1.0, 2.5]), "p2": np.array([2.0, 2.5]), "length": 1.0},
            # Segment 5: Arc top-right
            {"type": "arc", "center": np.array([2.0, 2.0]), "radius": 0.5, "start_angle": math.pi/2.0, "span": -math.pi/2.0, "length": math.pi * 0.25},
            # Segment 6: Line right
            {"type": "line", "p1": np.array([2.5, 2.0]), "p2": np.array([2.5, 1.0]), "length": 1.0},
            # Segment 7: Arc bottom-right
            {"type": "arc", "center": np.array([2.0, 1.0]), "radius": 0.5, "start_angle": 0.0, "span": -math.pi/2.0, "length": math.pi * 0.25},
        ]

def project_on_line(p, p1, p2):
    d_seg = p2 - p1
    seg_len = np.linalg.norm(d_seg)
    if seg_len < 1e-9:
        return p1, np.linalg.norm(p - p1), np.array([1.0, 0.0]), 0.0
    tangent = d_seg / seg_len
    t = np.dot(p - p1, tangent) / seg_len
    t_clamped = np.clip(t, 0.0, 1.0)
    closest = p1 + t_clamped * d_seg
    dist = np.linalg.norm(p - closest)
    return closest, dist, tangent, t_clamped

def project_on_arc(p, center, radius, start_angle, span):
    D = p - center
    dist_c = np.linalg.norm(D)
    if dist_c < 1e-9:
        theta = start_angle
    else:
        theta = math.atan2(D[1], D[0])
        
    phi = theta - start_angle
    # Map phi to [-pi, pi]
    phi = (phi + math.pi) % (2.0 * math.pi) - math.pi
    
    if span > 0: # CCW
        phi_clamped = np.clip(phi, 0.0, span)
        theta_clamped = start_angle + phi_clamped
        tangent = np.array([-math.sin(theta_clamped), math.cos(theta_clamped)])
        t = phi_clamped / span
    else: # CW
        phi_clamped = np.clip(phi, span, 0.0)
        theta_clamped = start_angle + phi_clamped
        tangent = np.array([math.sin(theta_clamped), -math.cos(theta_clamped)])
        t = phi_clamped / span
        
    closest = center + radius * np.array([math.cos(theta_clamped), math.sin(theta_clamped)])
    dist = np.linalg.norm(p - closest)
    return closest, dist, tangent, t

def get_closest_point_on_path(rx, ry, direction):
    segments = get_path_segments(direction)
    p = np.array([rx, ry])
    
    best_closest = None
    best_dist = float('inf')
    best_tangent = None
    best_segment_idx = 0
    best_t = 0.0
    
    s_start = 0.0
    best_s = 0.0
    
    for idx, seg in enumerate(segments):
        seg_len = seg["length"]
        if seg["type"] == "line":
            closest, dist, tangent, t = project_on_line(p, seg["p1"], seg["p2"])
        else: # arc
            closest, dist, tangent, t = project_on_arc(p, seg["center"], seg["radius"], seg["start_angle"], seg["span"])
            
        if dist < best_dist:
            best_dist = dist
            best_closest = closest
            best_tangent = tangent
            best_segment_idx = idx
            best_t = t
            best_s = s_start + t * seg_len
            
        s_start += seg_len
        
    # Return best_segment_idx = 0 and best_t = best_s / 2.0 so that
    # s_current = best_segment_idx * 2.0 + best_t * 2.0 = best_s holds.
    return best_closest, best_dist, best_tangent, 0, best_s / 2.0

def get_point_at_s(s, direction):
    segments = get_path_segments(direction)
    total_len = 4.0 + math.pi
    s = s % total_len
    
    s_start = 0.0
    for seg in segments:
        seg_len = seg["length"]
        if s_start <= s <= s_start + seg_len + 1e-9:
            t = (s - s_start) / seg_len
            t = np.clip(t, 0.0, 1.0)
            if seg["type"] == "line":
                return seg["p1"] + t * (seg["p2"] - seg["p1"])
            else: # arc
                theta = seg["start_angle"] + t * seg["span"]
                return seg["center"] + seg["radius"] * np.array([math.cos(theta), math.sin(theta)])
        s_start += seg_len
    return segments[-1]["p2"] if segments[-1]["type"] == "line" else segments[-1]["center"] + segments[-1]["radius"] * np.array([math.cos(segments[-1]["start_angle"] + segments[-1]["span"]), math.sin(segments[-1]["start_angle"] + segments[-1]["span"])])

def get_next_inner_corner(s, direction):
    total_len = 4.0 + math.pi
    s = s % total_len
    pi = math.pi
    
    if direction == "CCW":
        if s < 1.0 + pi * 0.25:
            return np.array([2.0, 1.0])
        elif s < 2.0 + pi * 0.5:
            return np.array([2.0, 2.0])
        elif s < 3.0 + pi * 0.75:
            return np.array([1.0, 2.0])
        else:
            return np.array([1.0, 1.0])
    else: # CW
        if s < 1.0 + pi * 0.25:
            return np.array([1.0, 1.0])
        elif s < 2.0 + pi * 0.5:
            return np.array([1.0, 2.0])
        elif s < 3.0 + pi * 0.75:
            return np.array([2.0, 2.0])
        else:
            return np.array([2.0, 1.0])

def get_point_and_tangent_at_s(s, direction):
    """Returns (point, tangent) at arc-length position s along the centerline."""
    segments = get_path_segments(direction)
    total_len = 4.0 + math.pi
    s = s % total_len
    
    s_start = 0.0
    for seg in segments:
        seg_len = seg["length"]
        if s_start <= s <= s_start + seg_len + 1e-9:
            t = (s - s_start) / seg_len
            t = np.clip(t, 0.0, 1.0)
            if seg["type"] == "line":
                point = seg["p1"] + t * (seg["p2"] - seg["p1"])
                d_seg = seg["p2"] - seg["p1"]
                tangent = d_seg / np.linalg.norm(d_seg)
            else:  # arc
                theta = seg["start_angle"] + t * seg["span"]
                point = seg["center"] + seg["radius"] * np.array([math.cos(theta), math.sin(theta)])
                if seg["span"] > 0:  # CCW arc
                    tangent = np.array([-math.sin(theta), math.cos(theta)])
                else:  # CW arc
                    tangent = np.array([math.sin(theta), -math.cos(theta)])
            return point, tangent
        s_start += seg_len
    
    # Fallback: last segment end
    seg = segments[-1]
    if seg["type"] == "line":
        point = seg["p2"].copy()
        d_seg = seg["p2"] - seg["p1"]
        tangent = d_seg / np.linalg.norm(d_seg)
    else:
        theta = seg["start_angle"] + seg["span"]
        point = seg["center"] + seg["radius"] * np.array([math.cos(theta), math.sin(theta)])
        if seg["span"] > 0:
            tangent = np.array([-math.sin(theta), math.cos(theta)])
        else:
            tangent = np.array([math.sin(theta), -math.cos(theta)])
    return point, tangent

def _ray_rect_intersection(px, py, dx, dy, x_min, y_min, x_max, y_max):
    """Find the nearest positive-t intersection of ray (px,py)+t*(dx,dy) with rectangle walls.
    
    Returns the distance t, or float('inf') if no intersection found.
    """
    t_min = float('inf')
    
    # Check vertical walls (x = x_min, x = x_max)
    for wall_x in [x_min, x_max]:
        if abs(dx) > 1e-9:
            t = (wall_x - px) / dx
            if t > 1e-6:
                hit_y = py + t * dy
                if y_min - 1e-9 <= hit_y <= y_max + 1e-9:
                    t_min = min(t_min, t)
    
    # Check horizontal walls (y = y_min, y = y_max)
    for wall_y in [y_min, y_max]:
        if abs(dy) > 1e-9:
            t = (wall_y - py) / dy
            if t > 1e-6:
                hit_x = px + t * dx
                if x_min - 1e-9 <= hit_x <= x_max + 1e-9:
                    t_min = min(t_min, t)
    
    return t_min

def get_boundary_points_at_s(s, direction):
    """
    Compute inner and outer track boundary points at centerline position s.
    
    Projects perpendicular from the centerline to the inner wall rectangle
    (1.0, 1.0)-(2.0, 2.0) and the outer wall rectangle (0.0, 0.0)-(3.0, 3.0).
    
    Returns (inner_point, outer_point) as numpy arrays.
    """
    point, tangent = get_point_and_tangent_at_s(s, direction)
    left_normal = np.array([-tangent[1], tangent[0]])
    
    # For CCW: left normal points toward inner wall (toward field center)
    # For CW: left normal points toward outer wall
    if direction == "CCW":
        inner_dir = left_normal
        outer_dir = -left_normal
    else:
        inner_dir = -left_normal
        outer_dir = left_normal
    
    # Ray-cast to inner wall (rectangle 1.0-2.0)
    d_inner = _ray_rect_intersection(
        point[0], point[1], inner_dir[0], inner_dir[1],
        1.0, 1.0, 2.0, 2.0
    )
    if d_inner == float('inf'):
        d_inner = 0.5  # fallback
    inner_point = point + inner_dir * d_inner
    
    # Ray-cast to outer wall (rectangle 0.0-3.0)
    d_outer = _ray_rect_intersection(
        point[0], point[1], outer_dir[0], outer_dir[1],
        0.0, 0.0, 3.0, 3.0
    )
    if d_outer == float('inf'):
        d_outer = 0.5  # fallback
    outer_point = point + outer_dir * d_outer
    
    return inner_point, outer_point
