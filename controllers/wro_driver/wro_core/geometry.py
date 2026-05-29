import numpy as np

def get_track_vertices(direction):
    if direction == "CCW":
        return [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]
    else:
        return [(2.5, 0.5), (0.5, 0.5), (0.5, 2.5), (2.5, 2.5)]

def get_closest_point_on_path(rx, ry, direction):
    vertices = get_track_vertices(direction)
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

def get_point_at_s(s, direction):
    vertices = get_track_vertices(direction)
    s = s % 8.0
    k = int(s // 2.0) % 4
    t = (s % 2.0) / 2.0
    p_start = np.array(vertices[k])
    p_end = np.array(vertices[(k + 1) % 4])
    return p_start + t * (p_end - p_start)
