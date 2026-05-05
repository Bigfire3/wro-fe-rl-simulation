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

        Returns corrected (robot_x, robot_y).
        """
        n_rays = len(lidar_ranges)
        # Reversed sign for horizontal mirroring
        angle_inc = -2.0 * math.pi / n_rays

        self.robot_yaw = robot_yaw

        # --- Position correction using known wall distances (ICP) ---
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

        # Compute LiDAR endpoints for visualization (use corrected yaw)
        lidar_pts = []
        for i, r in enumerate(lidar_ranges):
            if r <= 0.01 or r > max_range or math.isinf(r) or math.isnan(r):
                continue
            angle = corrected_yaw + angle_offset + i * angle_inc
            ex = corrected_x + r * math.cos(angle)
            ey = corrected_y + r * math.sin(angle)
            lidar_pts.append((ex, ey))

        self.last_lidar_points = lidar_pts

        self._update_section(corrected_x, corrected_y)

        return corrected_x, corrected_y, corrected_yaw

    def render(self, window_size=600):
        """
        Renders the current state (background, trajectory, lidar, robot) to an OpenCV image.
        """
        if not hasattr(self, 'background_img') or self.background_img is None:
            # Try to load the arena texture
            try:
                # Path relative to controllers/wro_driver/
                path = os.path.join(os.path.dirname(__file__), "..", "..", "worlds", "textures", "Spielfeld.png")
                raw_bg = cv2.imread(path)
                if raw_bg is not None:
                    # Convert to grayscale then back to BGR for drawing colors on top
                    gray = cv2.cvtColor(raw_bg, cv2.COLOR_BGR2GRAY)
                    bg_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    self.background_img = cv2.resize(bg_bgr, (window_size, window_size))
                else:
                    print(f"[Localizer] Warning: Could not load background from {path}")
                    self.background_img = np.zeros((window_size, window_size, 3), dtype=np.uint8)
            except Exception as e:
                print(f"[Localizer] Error loading background: {e}")
                self.background_img = np.zeros((window_size, window_size, 3), dtype=np.uint8)

        # Start with the background image
        img = self.background_img.copy()
        
        # Scale factor from grid/world to window
        scale = window_size / self.grid_size
        
        def to_win(world_x, world_y):
            gx, gy = self._world_to_grid(world_x, world_y)
            return (int(gx * scale), int(gy * scale))

        # 1. Draw Walls (Grey)
        segments = [
            # Outer
            ([-1.5, -1.5], [ 1.5, -1.5]), ([-1.5,  1.5], [ 1.5,  1.5]),
            ([-1.5, -1.5], [-1.5,  1.5]), ([ 1.5, -1.5], [ 1.5,  1.5]),
            # Inner
            ([-0.5, -0.5], [ 0.5, -0.5]), ([-0.5,  0.5], [ 0.5,  0.5]),
            ([-0.5, -0.5], [-0.5,  0.5]), ([ 0.5, -0.5], [ 0.5,  0.5]),
        ]
        for A, B in segments:
            p1 = to_win(A[0], A[1])
            p2 = to_win(B[0], B[1])
            cv2.line(img, p1, p2, (80, 80, 80), 2, cv2.LINE_AA)

        # 2. Draw Trajectory (Blue)
        if len(self.trajectory) > 1:
            pts = [to_win(x, y) for x, y in self.trajectory]
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i+1], (255, 100, 0), 1, cv2.LINE_AA)

        # 3. Draw LiDAR points (Red)
        for px, py in self.last_lidar_points:
            p = to_win(px, py)
            cv2.circle(img, p, 2, (0, 0, 255), -1, cv2.LINE_AA)

        # 4. Draw Robot (Green Circle + Direction Line)
        if self.trajectory:
            rx, ry = self.trajectory[-1]
            rp = to_win(rx, ry)
            cv2.circle(img, rp, 8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Direction arrow
            dir_len = 0.15 # metres
            dx = dir_len * math.cos(self.robot_yaw)
            dy = dir_len * math.sin(self.robot_yaw)
            tp = to_win(rx + dx, ry + dy)
            cv2.line(img, rp, tp, (0, 255, 0), 2, cv2.LINE_AA)

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
