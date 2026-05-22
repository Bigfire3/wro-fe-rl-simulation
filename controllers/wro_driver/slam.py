"""
Known-Map Localization + Occupancy Grid
========================================
Uses a predefined track layout (3×3m arena with inner obstacle)
for accurate localization instead of pure SLAM.
LiDAR scans are matched against the known map to correct position drift.
Includes section tracking and lap counting.
"""

import math
import numpy as np
import cv2
import os


# ======================================================================
# Track geometry constants (metres, world coordinates)
# ======================================================================
# Outer arena: 3×3m centred at origin → walls at ±1.5
OUTER_MIN_X = -1.5
OUTER_MAX_X = 1.5
OUTER_MIN_Y = -1.5
OUTER_MAX_Y = 1.5

# Inner obstacle walls (from track.wbt – centred 1×1m square)
INNER_MIN_X = -0.5
INNER_MAX_X = 0.5
INNER_MIN_Y = -0.5
INNER_MAX_Y = 0.5

# Wall segments: list of (x1, y1, x2, y2) line segments
WALL_SEGMENTS = [
    # Outer walls
    (OUTER_MIN_X, OUTER_MIN_Y, OUTER_MAX_X, OUTER_MIN_Y),  # bottom
    (OUTER_MAX_X, OUTER_MIN_Y, OUTER_MAX_X, OUTER_MAX_Y),  # right
    (OUTER_MAX_X, OUTER_MAX_Y, OUTER_MIN_X, OUTER_MAX_Y),  # top
    (OUTER_MIN_X, OUTER_MAX_Y, OUTER_MIN_X, OUTER_MIN_Y),  # left
    # Inner walls
    (INNER_MIN_X, INNER_MIN_Y, INNER_MAX_X, INNER_MIN_Y),  # bottom
    (INNER_MAX_X, INNER_MIN_Y, INNER_MAX_X, INNER_MAX_Y),  # right
    (INNER_MAX_X, INNER_MAX_Y, INNER_MIN_X, INNER_MAX_Y),  # top
    (INNER_MIN_X, INNER_MAX_Y, INNER_MIN_X, INNER_MIN_Y),  # left
]

# Section definitions: (name, type, x_range, y_range)
# The corridor is split into 8 sections around the inner obstacle.
SECTIONS = [
    ("N",  "straight", (INNER_MIN_X, INNER_MAX_X), (INNER_MAX_Y, OUTER_MAX_Y)),
    ("NE", "corner",   (INNER_MAX_X, OUTER_MAX_X), (INNER_MAX_Y, OUTER_MAX_Y)),
    ("E",  "straight", (INNER_MAX_X, OUTER_MAX_X), (INNER_MIN_Y, INNER_MAX_Y)),
    ("SE", "corner",   (INNER_MAX_X, OUTER_MAX_X), (OUTER_MIN_Y, INNER_MIN_Y)),
    ("S",  "straight", (INNER_MIN_X, INNER_MAX_X), (OUTER_MIN_Y, INNER_MIN_Y)),
    ("SW", "corner",   (OUTER_MIN_X, INNER_MIN_X), (OUTER_MIN_Y, INNER_MIN_Y)),
    ("W",  "straight", (OUTER_MIN_X, INNER_MIN_X), (INNER_MIN_Y, INNER_MAX_Y)),
    ("NW", "corner",   (OUTER_MIN_X, INNER_MIN_X), (INNER_MAX_Y, OUTER_MAX_Y)),
]


class KnownMapLocalizer:
    """
    Localization on a known track map.

    - Pre-fills the occupancy grid with known walls
    - Corrects robot pose using LiDAR vs. expected wall distances
    - Tracks which section the robot is in and counts laps
    """

    L_OCC = 0.9
    L_FREE = -0.4
    L_MIN = -5.0
    L_MAX = 5.0

    def __init__(self, grid_size=300, resolution=0.02):
        self.grid_size = grid_size
        self.resolution = resolution
        self.origin = (0.0, 0.0)

        # Log-odds grid
        self.log_odds = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Pre-fill known walls
        self._draw_known_walls()

        # Tracking
        self.trajectory = []
        self.last_lidar_points = []
        self.robot_cell = (grid_size // 2, grid_size // 2)
        self.robot_yaw = 0.0

        # Section & lap tracking
        self.current_section_idx = -1
        self.current_section_name = "?"
        self.current_section_type = "?"
        self.lap_count = 0
        self._start_section_idx = -1   # determined on first update
        self._section_history = []     # track section transitions
        self._sections_visited = set()

        # Position correction
        self._correction_alpha = 0.3   # blend factor for correction

        # Dynamic Obstacle Memory
        # Dictionary mapping grid cells (gx, gy) to confidence value [0.0, 1.0]
        self.obstacles = {}
        self.obstacle_resolution = 0.05 # 5cm grid for obstacles

    # ------------------------------------------------------------------
    # Pre-fill the known map
    # ------------------------------------------------------------------
    def _draw_known_walls(self):
        """Draw all known wall segments onto the occupancy grid."""
        wall_thickness = 2  # cells

        for x1, y1, x2, y2 in WALL_SEGMENTS:
            gx1, gy1 = self._world_to_grid(x1, y1)
            gx2, gy2 = self._world_to_grid(x2, y2)
            cells = self._bresenham(gx1, gy1, gx2, gy2)
            for cx, cy in cells:
                for dx in range(-wall_thickness, wall_thickness + 1):
                    for dy in range(-wall_thickness, wall_thickness + 1):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            self.log_odds[ny, nx] = self.L_MAX

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, lidar_ranges, robot_x, robot_y, robot_yaw, imu_yaw,
               max_range=3.0, angle_offset=-math.pi):
        """
        Update localization with one LiDAR scan.
        """
        n_rays = len(lidar_ranges)
        # Reversed sign for horizontal mirroring
        angle_inc = -2.0 * math.pi / n_rays

        # --- 1. South-Fixed Start Alignment (Only on first call) ---
        if not hasattr(self, '_initialized') or not self._initialized:
            n_rays = len(lidar_ranges)
            angle_inc = -2.0 * math.pi / n_rays
            
            # 1. Parallel alignment (Tilt detection)
            tilt = self._detect_initial_yaw_at(lidar_ranges, 0.0, -1.0, 0.0, n_rays, angle_inc, angle_offset)
            
            # 2. Geometric Intersection Logic (User's Circle Idea)
            R_TEST = 1.0
                 # 3. Decision & Tentative Mode
            self._direction_locked = False
            # During init, we assume start center (0, -1) for visualization
            intersections_left = self._count_intersections(lidar_ranges, angle_inc, angle_offset, True, 0.0, -1.0, 0.0 - tilt)
            intersections_right = self._count_intersections(lidar_ranges, angle_inc, angle_offset, False, 0.0, -1.0, 0.0 - tilt)
            
            print(f"[Localizer] Circle Intersections (R={R_TEST}m): Left={intersections_left}, Right={intersections_right}")

            if intersections_left < intersections_right:
                # Left side is shorter (Inner) -> Facing EAST (CCW)
                print("[Localizer] Init: CCW (East) confirmed by intersection count.")
                base_yaw = 0.0
                ty = -1.5 + lidar_ranges[270]
                self._direction_locked = True
            elif intersections_right < intersections_left:
                # Right side is shorter (Inner) -> Facing WEST (CW)
                print("[Localizer] Init: Eindeutig CW (West).")
                base_yaw = math.pi
                ty = -1.5 + lidar_ranges[90]
                self._direction_locked = True
            else:
                # AMBIGUOUS (4 Points) -> Start tentative CCW
                print("[Localizer] Init: AMBIGUOUS (4 Points). Starting TENTATIVE CCW...")
                base_yaw = 0.0
                ty = -1.5 + lidar_ranges[270]
                self._direction_locked = False

            corrected_x = 0.0 
            corrected_y = ty
            corrected_yaw = base_yaw - tilt
            
            self._initialized = True
            self.robot_x = corrected_x
            self.robot_y = corrected_y
            self.robot_yaw = corrected_yaw
            self.trajectory = [(corrected_x, corrected_y)]
            return corrected_x, corrected_y, corrected_yaw

        # Reset debug visualizations
        self.debug_intersections = []

        # Reset debug visualizations
        self.debug_intersections = []

        # --- 2. Live Re-Validation (if tentative) ---
        if not self._direction_locked:
            intersections_left = self._count_intersections(lidar_ranges, angle_inc, angle_offset, True, robot_x, robot_y, robot_yaw)
            intersections_right = self._count_intersections(lidar_ranges, angle_inc, angle_offset, False, robot_x, robot_y, robot_yaw)
            
            if intersections_left != intersections_right:
                # DECISION TIME!
                print(f"[Localizer] Tentative phase ended. Left={intersections_left}, Right={intersections_right}")
                
                # Check if our tentative CCW was wrong
                should_be_cw = (intersections_right < intersections_left)
                if should_be_cw:
                    print("[Localizer] CRITICAL FLIP: Tentative direction was WRONG. Flipping to CW.")
                    # Flip current pose
                    robot_x = -robot_x # Mirror X relative to start (0,0)
                    robot_yaw = (robot_yaw + math.pi + np.pi) % (2 * np.pi) - np.pi
                    # Transform whole trajectory
                    self.trajectory = [(-tx, ty) for tx, ty in self.trajectory]
                else:
                    print("[Localizer] Tentative direction CONFIRMED.")
                
                self._direction_locked = True

        # Save latest position for rendering
        self.robot_x = robot_x
        self.robot_y = robot_y
        self.robot_yaw = robot_yaw

        # --- 3. Normal Position/Yaw correction ---
        corrected_x, corrected_y, corrected_yaw = self._correct_position(
            lidar_ranges, robot_x, robot_y, robot_yaw, imu_yaw,
            n_rays, angle_inc, angle_offset
        )

        self.robot_yaw = corrected_yaw
        rcx, rcy = self._world_to_grid(corrected_x, corrected_y)
        self.robot_cell = (rcx, rcy)

        # Record trajectory
        self.trajectory.append((corrected_x, corrected_y))
        if len(self.trajectory) > 5000:
            self.trajectory = self.trajectory[-5000:]

        # Compute LiDAR endpoints and classify them (Obstacle vs. Wall)
        ranges = np.array(lidar_ranges)
        valid = (ranges > 0.01) & (ranges < max_range) & ~np.isinf(ranges) & ~np.isnan(ranges)
        if np.any(valid):
            ray_angles = corrected_yaw + angle_offset + np.arange(n_rays)[valid] * angle_inc
            pts = np.stack([corrected_x + ranges[valid] * np.cos(ray_angles),
                            corrected_y + ranges[valid] * np.sin(ray_angles)], axis=1)

            # Expected walls (Manhattan grid)
            segments = [
                ([-1.5, -1.5], [ 1.5, -1.5]), ([-1.5,  1.5], [ 1.5,  1.5]),
                ([-1.5, -1.5], [-1.5,  1.5]), ([ 1.5, -1.5], [ 1.5,  1.5]),
                ([-0.5, -0.5], [ 0.5, -0.5]), ([-0.5,  0.5], [ 0.5,  0.5]),
                ([-0.5, -0.5], [-0.5,  0.5]), ([ 0.5, -0.5], [ 0.5,  0.5]),
            ]
            
            min_dists = np.full(len(pts), 1e6)
            for A_raw, B_raw in segments:
                A, B = np.array(A_raw), np.array(B_raw)
                v = B - A
                w = pts - A
                v_len_sq = np.sum(v * v)
                t = np.sum(w * v, axis=1) / v_len_sq
                
                closest_on_line = A + t[:, None] * v
                dist_to_line_sq = np.sum((closest_on_line - pts)**2, axis=1)
                
                out_of_bounds = np.maximum(0, -t) + np.maximum(0, t - 1)
                penalty = out_of_bounds * 5.0
                
                dists = np.sqrt(dist_to_line_sq) + penalty
                min_dists = np.minimum(min_dists, dists)

            # Filter threshold: >5cm from expected wall = Obstacle
            is_obstacle = min_dists > 0.05
            obstacle_pts = pts[is_obstacle]
            wall_pts = pts[~is_obstacle]
        else:
            obstacle_pts = np.empty((0, 2))
            wall_pts = np.empty((0, 2))

        # Update Obstacle Memory (Persistence focus)
        decay_rate = 0.005 # Much slower decay
        keys_to_delete = []
        for key in self.obstacles:
            # Only decay if the robot is VERY close to the spot (within 1.0m)
            # This ensures obstacles stay on the map when the robot is far away.
            owx = key[0] * self.obstacle_resolution
            owy = key[1] * self.obstacle_resolution
            dist_to_robot = math.hypot(owx - corrected_x, owy - corrected_y)
            
            if dist_to_robot < 1.0: # Only decay if nearby
                self.obstacles[key] -= decay_rate
                if self.obstacles[key] <= 0:
                    keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.obstacles[key]

        for ox, oy in obstacle_pts:
            gx = int(round(ox / self.obstacle_resolution))
            gy = int(round(oy / self.obstacle_resolution))
            key = (gx, gy)
            self.obstacles[key] = min(1.0, self.obstacles.get(key, 0.0) + 0.2)

        self.last_lidar_points = wall_pts
        # Store ALL valid lidar points (wall + obstacle) for raw overlay
        if np.any(valid):
            self.last_all_lidar_points = pts.tolist()
        else:
            self.last_all_lidar_points = []

        self._update_section(corrected_x, corrected_y)

        return corrected_x, corrected_y, corrected_yaw

    def _count_intersections(self, lidar_ranges, angle_inc, angle_offset, is_left_side, rx, ry, ryaw):
        """Counts intersections with a 1.0m radius and stores points for debug."""
        R_TEST = 1.0
        base_idx = 90 if is_left_side else 270
        base_dist = lidar_ranges[base_idx]
        if base_dist >= R_TEST: return 0
        
        has_front = False
        has_back = False
        n_rays = len(lidar_ranges)
        
        if not hasattr(self, 'debug_intersections'): self.debug_intersections = []
        
        for i in range(n_rays):
            d = lidar_ranges[i]
            if d < 0.01: continue
            angle_rad = (i * angle_inc) + angle_offset
            
            # Check for points at the 1.0m boundary
            if abs(d - R_TEST) < 0.05:
                side_y = d * math.sin(angle_rad)
                if (is_left_side and side_y > 0) or (not is_left_side and side_y < 0):
                    long_pos = d * math.cos(angle_rad)
                    
                    # Store point for visualization using the provided pose
                    wx = rx + d * math.cos(ryaw + angle_rad)
                    wy = ry + d * math.sin(ryaw + angle_rad)
                    
                    if long_pos > 0.3: 
                        if not has_front: self.debug_intersections.append((wx, wy, (255, 0, 0))) # Blue cross
                        has_front = True
                    if long_pos < -0.3: 
                        if not has_back: self.debug_intersections.append((wx, wy, (0, 0, 255))) # Red cross
                        has_back = True
                        
        return (1 if has_front else 0) + (1 if has_back else 0)

    def _detect_initial_yaw_at(self, lidar_ranges, tx, ty, tyaw, n_rays, angle_inc, angle_offset):
        """Finds tilt relative to grid if we assume we are at (tx, ty) facing tyaw."""
        ranges = np.array(lidar_ranges)
        valid = (ranges > 0.01) & (ranges < 2.5)
        if not np.any(valid): return 0.0
        
        # Project points using the hypothetical yaw
        ray_angles = tyaw + angle_offset + np.arange(n_rays)[valid] * angle_inc
        wx = tx + ranges[valid] * np.cos(ray_angles)
        wy = ty + ranges[valid] * np.sin(ray_angles)
        
        dx = wx[1:] - wx[:-1]
        dy = wy[1:] - wy[:-1]
        dist_sq = dx**2 + dy**2
        seg_mask = (dist_sq > 1e-8) & (dist_sq < 0.1**2)
        if not np.any(seg_mask): return 0.0
        
        seg_angles = np.arctan2(dy[seg_mask], dx[seg_mask])
        residuals = seg_angles % (np.pi / 2)
        median_residual = np.median(residuals)
        return (median_residual + np.pi/4) % (np.pi/2) - np.pi/4

    def _get_alignment_error(self, lidar_ranges, tx, ty, tyaw, n_rays, angle_inc, angle_offset):
        """Calculates total alignment error for a given pose."""
        ranges = np.array(lidar_ranges)
        valid = (ranges > 0.01) & (ranges < 2.5)
        if not np.any(valid): return 1e6
        
        ray_angles = tyaw + angle_offset + np.arange(n_rays)[valid] * angle_inc
        pts = np.stack([tx + ranges[valid] * np.cos(ray_angles),
                        ty + ranges[valid] * np.sin(ray_angles)], axis=1)
        
        segments = [
            ([-1.5, -1.5], [ 1.5, -1.5]), ([-1.5,  1.5], [ 1.5,  1.5]),
            ([-1.5, -1.5], [-1.5,  1.5]), ([ 1.5, -1.5], [ 1.5,  1.5]),
            ([-0.5, -0.5], [ 0.5, -0.5]), ([-0.5,  0.5], [ 0.5,  0.5]),
            ([-0.5, -0.5], [-0.5,  0.5]), ([ 0.5, -0.5], [ 0.5,  0.5]),
        ]
        
        min_dists = np.full(len(pts), 1e6)
        for A_raw, B_raw in segments:
            A, B = np.array(A_raw), np.array(B_raw)
            v = B - A
            w = pts - A
            v_len_sq = np.sum(v * v)
            t = np.sum(w * v, axis=1) / v_len_sq
            
            # Distance to the infinite line
            closest_on_line = A + t[:, None] * v
            dist_to_line_sq = np.sum((closest_on_line - pts)**2, axis=1)
            
            # Penalty for being outside the [0, 1] segment range
            # If t < 0 or t > 1, we add a large penalty based on the distance to the endpoint
            out_of_bounds = np.maximum(0, -t) + np.maximum(0, t - 1)
            penalty = out_of_bounds * 5.0 # High penalty factor for overhanging
            
            dists = np.sqrt(dist_to_line_sq) + penalty
            min_dists = np.minimum(min_dists, dists)
            
        return np.mean(min_dists)

    def render(self, window_size=600, robot_x=None, robot_y=None, robot_yaw=None, lidar_pts=None):
        """
        Renders state onto a full window canvas. Points can extend beyond the arena into the padding.
        """
        # Fallback to stored values
        rx = robot_x if robot_x is not None else getattr(self, 'robot_x', 0.0)
        ry = robot_y if robot_y is not None else getattr(self, 'robot_y', 0.0)
        ryaw = robot_yaw if robot_yaw is not None else self.robot_yaw
        
        padding = 50
        arena_size = window_size - 2 * padding
        scale = arena_size / 3.0
        
        # 1. Prepare Background
        if not hasattr(self, 'arena_bg_gray') or self.arena_bg_gray is None or self.arena_bg_gray.shape[0] != arena_size:
            try:
                path = os.path.join(os.path.dirname(__file__), "..", "..", "worlds", "textures", "Spielfeld.png")
                raw_bg = cv2.imread(path)
                if raw_bg is not None:
                    gray = cv2.cvtColor(raw_bg, cv2.COLOR_BGR2GRAY)
                    self.arena_bg_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    self.arena_bg_gray = cv2.resize(self.arena_bg_gray, (arena_size, arena_size))
                else:
                    self.arena_bg_gray = np.zeros((arena_size, arena_size, 3), dtype=np.uint8)
            except:
                self.arena_bg_gray = np.zeros((arena_size, arena_size, 3), dtype=np.uint8)

        # Create full canvas
        img = np.full((window_size, window_size, 3), 40, dtype=np.uint8) # Dark grey frame
        # Place arena background in center
        img[padding:padding+arena_size, padding:padding+arena_size] = self.arena_bg_gray
        
        # Global Mapping: Welt [-1.5, 1.5] -> Fenster [padding, window_size - padding]
        def to_win(wx, wy):
            px = int(padding + (wx + 1.5) * scale)
            py = int(padding + (1.5 - wy) * scale)
            return (px, py)

        # 2. Draw Components (No clipping, can draw into padding)
        segments = [
            ([-1.5, -1.5], [ 1.5, -1.5]), ([-1.5,  1.5], [ 1.5,  1.5]),
            ([-1.5, -1.5], [-1.5,  1.5]), ([ 1.5, -1.5], [ 1.5,  1.5]),
            ([-0.5, -0.5], [ 0.5, -0.5]), ([-0.5,  0.5], [ 0.5,  0.5]),
            ([-0.5, -0.5], [-0.5,  0.5]), ([ 0.5, -0.5], [ 0.5,  0.5]),
        ]
        for A, B in segments:
            cv2.line(img, to_win(A[0], A[1]), to_win(B[0], B[1]), (120, 120, 120), 2, cv2.LINE_AA)

        if len(self.trajectory) > 1:
            pts = [to_win(x, y) for x, y in self.trajectory]
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i+1], (255, 150, 0), 1, cv2.LINE_AA)

        # Draw Dynamic Obstacles (Subtle, Grouped/Zusammengefasst)
        if hasattr(self, 'obstacles') and len(self.obstacles) > 0:
            # Simple clustering: Group adjacent cells
            processed = set()
            clusters = []
            keys = list(self.obstacles.keys())
            for k in keys:
                if k in processed or self.obstacles[k] < 0.2: continue
                # Start a new cluster
                cluster = [k]
                q = [k]
                processed.add(k)
                while q:
                    curr = q.pop(0)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            neighbor = (curr[0]+dx, curr[1]+dy)
                            if neighbor in self.obstacles and neighbor not in processed and self.obstacles[neighbor] >= 0.2:
                                processed.add(neighbor)
                                cluster.append(neighbor)
                                q.append(neighbor)
                clusters.append(cluster)

            for cluster in clusters:
                # Calculate centroid and average confidence
                sum_x = sum(c[0] for c in cluster)
                sum_y = sum(c[1] for c in cluster)
                avg_conf = sum(self.obstacles[c] for c in cluster) / len(cluster)
                
                wx = (sum_x / len(cluster)) * self.obstacle_resolution
                wy = (sum_y / len(cluster)) * self.obstacle_resolution
                p = to_win(wx, wy)
                
                # Small subtle green rectangle
                brightness = int(100 + avg_conf * 155)
                color = (0, brightness, 0)
                
                # Draw a small 10x10 pixel rectangle (centered)
                size = 5 
                cv2.rectangle(img, (p[0]-size, p[1]-size), (p[0]+size, p[1]+size), color, -1)
                cv2.rectangle(img, (p[0]-size, p[1]-size), (p[0]+size, p[1]+size), (0, 60, 0), 1) # Dark outline

        # Draw Extracted Wall Segments (color-coded, each segment unique shade)
        wall_pts_arr = getattr(self, 'last_lidar_points', np.empty((0, 2)))
        if isinstance(wall_pts_arr, list):
            wall_pts_arr = np.array(wall_pts_arr) if len(wall_pts_arr) > 0 else np.empty((0, 2))
        if len(wall_pts_arr) > 0 and wall_pts_arr.ndim == 2:
            # Inner = light blue tones (BGR), Outer = dark blue tones (BGR)
            inner_colors = [
                (255, 230, 180), (255, 210, 160),
                (255, 200, 140), (240, 230, 180),
            ]
            outer_colors = [
                (180, 80, 0), (160, 60, 0),
                (140, 50, 0), (120, 70, 20),
            ]
            nominal_lines = [
                (False, -1.5, False, 0), (False,  1.5, False, 1),
                (True,  -1.5, False, 2), (True,   1.5, False, 3),
                (False, -0.5, True,  0), (False,  0.5, True,  1),
                (True,  -0.5, True,  2), (True,   0.5, True,  3),
            ]
            tol = 0.08
            for is_vertical, coord, is_inner, ci in nominal_lines:
                color = inner_colors[ci] if is_inner else outer_colors[ci]
                if is_vertical:
                    mask = np.abs(wall_pts_arr[:, 0] - coord) < tol
                    valid_pts = wall_pts_arr[mask]
                    if len(valid_pts) > 1:
                        min_v = np.min(valid_pts[:, 1])
                        max_v = np.max(valid_pts[:, 1])
                        if is_inner and (max_v - min_v) > 1.0:
                            center = (min_v + max_v) / 2.0
                            min_v, max_v = center - 0.5, center + 0.5
                        cv2.line(img, to_win(coord, min_v), to_win(coord, max_v), color, 3, cv2.LINE_AA)
                else:
                    mask = np.abs(wall_pts_arr[:, 1] - coord) < tol
                    valid_pts = wall_pts_arr[mask]
                    if len(valid_pts) > 1:
                        min_v = np.min(valid_pts[:, 0])
                        max_v = np.max(valid_pts[:, 0])
                        if is_inner and (max_v - min_v) > 1.0:
                            center = (min_v + max_v) / 2.0
                            min_v, max_v = center - 0.5, center + 0.5
                        cv2.line(img, to_win(min_v, coord), to_win(max_v, coord), color, 3, cv2.LINE_AA)

        # Draw raw LiDAR points ON TOP in ORANGE (Larger for visibility)
        all_pts = getattr(self, 'last_all_lidar_points', [])
        for px, py in all_pts:
            cv2.circle(img, to_win(px, py), 3, (0, 140, 255), -1, cv2.LINE_AA)

        rp = to_win(rx, ry)
        cv2.circle(img, rp, 10, (0, 255, 0), 2, cv2.LINE_AA)
        dir_len = 0.2
        tp = to_win(rx + dir_len * math.cos(ryaw), ry + dir_len * math.sin(ryaw))
        cv2.line(img, rp, tp, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw Debug Circle (1.0m Radius)
        radius_px = int(1.0 * scale)
        cv2.circle(img, rp, radius_px, (255, 255, 0), 1, cv2.LINE_AA)

        # Draw Debug Intersections (Crosses)
        if hasattr(self, 'debug_intersections'):
            for wx, wy, color in self.debug_intersections:
                cv2.drawMarker(img, to_win(wx, wy), color, cv2.MARKER_CROSS, 15, 2)

        # Draw Arena Boundary Line
        cv2.rectangle(img, (padding, padding), (window_size-padding, window_size-padding), (200, 200, 200), 1)

        return img

    def _correct_position(self, lidar_ranges, robot_x, robot_y, robot_yaw, imu_yaw,
                          n_rays, angle_inc, angle_offset):
        """
        Simplified Scan Matching: Focus only on Rotation Fusion.
        1. Convert LiDAR points to world angles using IMU yaw.
        2. Extract wall segment angles.
        3. Correct IMU yaw using the Manhattan grid alignment of walls.
        4. Skip X/Y correction for now.
        """
        ranges = np.array(lidar_ranges)
        
        valid = (ranges > 0.01) & (ranges < 3.0) & ~np.isinf(ranges) & ~np.isnan(ranges)
        if not np.any(valid):
            return robot_x, robot_y, imu_yaw
            
        ranges_v = ranges[valid]
        # Ray angles in world frame
        ray_angles_world = imu_yaw + angle_offset + np.arange(n_rays)[valid] * angle_inc
        
        # We need world coordinates just to find the angles of segments
        wx = ranges_v * np.cos(ray_angles_world)
        wy = ranges_v * np.sin(ray_angles_world)
        
        if len(wx) < 5:
            return robot_x, robot_y, imu_yaw

        # --- 1. Yaw correction from wall segment angles ---
        dx_pts = wx[1:] - wx[:-1]
        dy_pts = wy[1:] - wy[:-1]
        
        dist_sq = dx_pts**2 + dy_pts**2
        seg_mask = (dist_sq > 1e-8) & (dist_sq < 0.05**2)
        
        fused_yaw = imu_yaw
        
        if np.any(seg_mask):
            seg_angles = np.arctan2(dy_pts[seg_mask], dx_pts[seg_mask])
            
            # Residual relative to Manhattan grid (0, 90, 180, 270)
            # This residual is the rotation error of the LiDAR scan in the world frame.
            residuals = seg_angles % (np.pi / 2)
            median_residual = np.median(residuals)
            
            # Map residual [0, pi/2] to error [-pi/4, pi/4]
            yaw_error = (median_residual + np.pi/4) % (np.pi/2) - np.pi/4
            
            # Correct the yaw: if segments are rotated by 'yaw_error', we subtract it.
            wall_yaw = imu_yaw - yaw_error
            
            # PURE IMU: ignore wall angles for heading to avoid noise
            fused_yaw = imu_yaw
            
        # Smoothly update robot_yaw using shortest-path angular difference
        alpha = self._correction_alpha
        yaw_diff = fused_yaw - robot_yaw
        # Wrap diff to [-pi, pi] to handle 180/-180 boundary correctly
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        new_yaw = robot_yaw + alpha * yaw_diff
        
        # Keep new_yaw within [-pi, pi]
        new_yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # --- 2. Position correction (Corner-Aware Segment Matching) ---
        # Re-project points using the refined yaw
        ray_angles_corrected = new_yaw + angle_offset + np.arange(n_rays)[valid] * angle_inc
        wx_c = robot_x + ranges_v * np.cos(ray_angles_corrected)
        wy_c = robot_y + ranges_v * np.sin(ray_angles_corrected)
        
        pts = np.stack([wx_c, wy_c], axis=1) # (N, 2)
        
        segments = [
            # Outer
            ([-1.5, -1.5], [ 1.5, -1.5]), ([-1.5,  1.5], [ 1.5,  1.5]),
            ([-1.5, -1.5], [-1.5,  1.5]), ([ 1.5, -1.5], [ 1.5,  1.5]),
            # Inner
            ([-0.5, -0.5], [ 0.5, -0.5]), ([-0.5,  0.5], [ 0.5,  0.5]),
            ([-0.5, -0.5], [-0.5,  0.5]), ([ 0.5, -0.5], [ 0.5,  0.5]),
        ]
        
        all_errors = []
        all_ts = []
        for A_raw, B_raw in segments:
            A, B = np.array(A_raw), np.array(B_raw)
            v = B - A
            w = pts - A
            t = np.sum(w * v, axis=1) / np.sum(v * v)
            t_clamped = np.clip(t, 0, 1)
            closest = A + t_clamped[:, None] * v
            all_errors.append(closest - pts)
            all_ts.append(t_clamped)
            
        all_errors = np.stack(all_errors, axis=0) # (8, N, 2)
        all_ts = np.stack(all_ts, axis=0)         # (8, N)
        dists_sq = np.sum(all_errors**2, axis=2) # (8, N)
        s_idx = np.argmin(dists_sq, axis=0)      # (N,)
        
        final_errors = all_errors[s_idx, np.arange(len(pts))] # (N, 2)
        final_ts = all_ts[s_idx, np.arange(len(pts))]         # (N,)
        
        # Determine if each point is at a corner (t is 0 or 1)
        is_at_corner = (final_ts < 1e-3) | (final_ts > 0.999)
        
        # Segment orientations: segments 0,1,4,5 are horizontal, 2,3,6,7 are vertical
        is_horiz_seg = np.isin(s_idx, [0, 1, 4, 5])
        
        x_corrections = []
        y_corrections = []
        
        for i in range(len(pts)):
            err = final_errors[i]
            if is_at_corner[i]:
                # At a corner, both X and Y errors are reliable signals
                x_corrections.append(err[0])
                y_corrections.append(err[1])
            elif is_horiz_seg[i]:
                # Middle of horizontal wall: only Y error is signal
                y_corrections.append(err[1])
            else:
                # Middle of vertical wall: only X error is signal
                x_corrections.append(err[0])
        
        dx_shift = np.median(x_corrections) if x_corrections else 0.0
        dy_shift = np.median(y_corrections) if y_corrections else 0.0
        
        # Clamp correction to 20cm to allow catching larger corner offsets
        MAX_C = 0.2
        dx_shift = np.clip(dx_shift, -MAX_C, MAX_C)
        dy_shift = np.clip(dy_shift, -MAX_C, MAX_C)
        
        new_x = robot_x + alpha * dx_shift
        new_y = robot_y + alpha * dy_shift
        
        return new_x, new_y, new_yaw

    # ------------------------------------------------------------------
    # Section & Lap tracking
    # ------------------------------------------------------------------
    def _update_section(self, x, y):
        """Determine which section the robot is in and count laps."""
        new_idx = -1
        for i, (name, stype, (xmin, xmax), (ymin, ymax)) in enumerate(SECTIONS):
            if xmin <= x <= xmax and ymin <= y <= ymax:
                new_idx = i
                break

        if new_idx < 0:
            return  # robot outside known sections

        if new_idx != self.current_section_idx:
            prev_idx = self.current_section_idx
            self.current_section_idx = new_idx
            self.current_section_name = SECTIONS[new_idx][0]
            self.current_section_type = SECTIONS[new_idx][1]

            # First section detection → set as start
            if self._start_section_idx < 0:
                self._start_section_idx = new_idx
                self._sections_visited = {new_idx}
                self._section_history = [new_idx]
            else:
                self._section_history.append(new_idx)
                self._sections_visited.add(new_idx)

                # Lap complete: returned to start after visiting ≥7 sections
                if (new_idx == self._start_section_idx and
                        len(self._sections_visited) >= 7):
                    self.lap_count += 1
                    self._sections_visited = {new_idx}
                    self._section_history = [new_idx]

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def get_map_image(self):
        """Return the occupancy grid as an RGBA uint8 array (H×W×4)."""
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

        h, w = prob.shape
        img = np.zeros((h, w, 4), dtype=np.uint8)

        occ_mask = prob > 0.6
        free_mask = prob < 0.4
        unk_mask = ~occ_mask & ~free_mask

        img[occ_mask] = [30, 30, 40, 255]
        img[free_mask] = [220, 225, 230, 200]
        img[unk_mask] = [100, 100, 110, 40]

        # Draw section boundaries (thin lines)
        for _, _, (xmin, xmax), (ymin, ymax) in SECTIONS:
            gx1, gy1 = self._world_to_grid(xmin, ymin)
            gx2, gy2 = self._world_to_grid(xmax, ymax)
            # Clamp to grid
            gx1 = max(0, min(self.grid_size - 1, gx1))
            gx2 = max(0, min(self.grid_size - 1, gx2))
            gy1 = max(0, min(self.grid_size - 1, gy1))
            gy2 = max(0, min(self.grid_size - 1, gy2))
            # Draw rectangle border
            for gx in range(gx1, gx2 + 1):
                if 0 <= gy1 < h:
                    img[gy1, gx] = [60, 60, 80, 100]
                if 0 <= gy2 < h:
                    img[gy2, gx] = [60, 60, 80, 100]
            for gy in range(gy1, gy2 + 1):
                if 0 <= gx1 < w:
                    img[gy, gx1] = [60, 60, 80, 100]
                if 0 <= gx2 < w:
                    img[gy, gx2] = [60, 60, 80, 100]

        return img

    def get_map_png_bytes(self):
        """Return the map as PNG bytes."""
        img = self.get_map_image()
        return self._encode_png(img)

    def get_state(self):
        """Return state dict for the browser viewer."""
        rcx, rcy = self.robot_cell

        traj = self.trajectory
        if len(traj) > 500:
            step = len(traj) // 500
            traj = traj[::step]

        lpts = self.last_lidar_points
        if len(lpts) > 180:
            step = len(lpts) // 180
            lpts = lpts[::step]

        return {
            "robot_gx": rcx,
            "robot_gy": rcy,
            "robot_yaw": self.robot_yaw,
            "trajectory": [self._world_to_grid(x, y) for x, y in traj],
            "lidar_points": [self._world_to_grid(x, y) for x, y in lpts],
            "grid_size": self.grid_size,
            "resolution": self.resolution,
            "occupied_cells": int(np.sum(self.log_odds > 0.5)),
            # Section & lap info
            "section_name": self.current_section_name,
            "section_type": self.current_section_type,
            "section_idx": self.current_section_idx,
            "lap_count": self.lap_count,
            # Section geometries for viewer overlay
            "sections": [
                {
                    "name": name,
                    "type": stype,
                    "gx1": self._world_to_grid(xr[0], yr[0])[0],
                    "gy1": self._world_to_grid(xr[0], yr[0])[1],
                    "gx2": self._world_to_grid(xr[1], yr[1])[0],
                    "gy2": self._world_to_grid(xr[1], yr[1])[1],
                }
                for name, stype, xr, yr in SECTIONS
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _world_to_grid(self, wx, wy):
        """Convert world coordinates (metres) to grid indices, matching Webots +Y = North to Canvas -Y = Up."""
        gx = int((wx - self.origin[0]) / self.resolution) + self.grid_size // 2
        gy = int((-wy - self.origin[1]) / self.resolution) + self.grid_size // 2
        return gx, gy

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    @staticmethod
    def _encode_png(img_rgba):
        import struct
        import zlib

        height, width, _ = img_rgba.shape

        def _chunk(chunk_type, data):
            c = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc

        sig = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
        ihdr = _chunk(b'IHDR', ihdr_data)

        raw = b''
        for y in range(height):
            raw += b'\x00'
            raw += img_rgba[y].tobytes()

        compressed = zlib.compress(raw, 6)
        idat = _chunk(b'IDAT', compressed)
        iend = _chunk(b'IEND', b'')

        return sig + ihdr + idat + iend
