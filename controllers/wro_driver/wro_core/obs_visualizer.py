import cv2
import numpy as np
import os
import math

class ObsVisualizer:
    def __init__(self):
        self.window_size = 600
        self.scale = 200.0
        
        # Load background Spielfeld.png
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "worlds", "textures", "Spielfeld.png"))
        
        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                # Convert to grayscale
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                self.bg_img = cv2.resize(img_gray, (self.window_size, self.window_size))
            else:
                print(f"[ObsVisualizer] Warning: Could not load image at '{img_path}'. Using blank background.")
                self.bg_img = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
        else:
            print(f"[ObsVisualizer] Warning: File '{img_path}' does not exist. Using blank background.")
            self.bg_img = np.zeros((self.window_size, self.window_size), dtype=np.uint8)

    def draw(self, pose, raw_obs, obs_vector, driving_direction, best_closest,
             p_inner_40=None, p_inner_100=None, p_outer_40=None, p_outer_100=None,
             p_corner=None):
        """
        Draws the observation debug window with boundary lookahead points.
        
        Parameters:
            pose: tuple of (rx, ry, ryaw)
            raw_obs: numpy array of 14 elements (unnormalized)
            obs_vector: numpy array of 14 elements (normalized and clipped)
            driving_direction: "CCW" or "CW"
            best_closest: coordinates of closest point on centerline (x, y) or None
            p_inner_40: inner boundary point at 40cm lookahead or None
            p_inner_100: inner boundary point at 100cm lookahead or None
            p_outer_40: outer boundary point at 40cm lookahead or None
            p_outer_100: outer boundary point at 100cm lookahead or None
            p_corner: next inner corner position or None
        """
        # 1. Convert grayscale background to BGR
        img = cv2.cvtColor(self.bg_img, cv2.COLOR_GRAY2BGR)
        
        # 2. Draw Ideal Line (rounded square track centerline)
        from . import geometry
        path_points = []
        total_len = 4.0 + math.pi
        n_points = 100
        for i in range(n_points):
            s = (i / n_points) * total_len
            pt = geometry.get_point_at_s(s, driving_direction)
            pt_x = int(pt[0] * self.scale)
            pt_y = int(self.window_size - pt[1] * self.scale)
            path_points.append((pt_x, pt_y))
            
        for i in range(n_points):
            p1 = path_points[i]
            p2 = path_points[(i + 1) % n_points]
            cv2.line(img, p1, p2, color=(255, 180, 50), thickness=1, lineType=cv2.LINE_AA)
        
        # 2.5 Draw inner wall rectangle (1,1)-(2,2) and outer wall rectangle (0,0)-(3,3)
        def to_px(x, y):
            return (int(x * self.scale), int(self.window_size - y * self.scale))
        
        # Inner wall
        inner_corners = [to_px(1, 1), to_px(2, 1), to_px(2, 2), to_px(1, 2)]
        for i in range(4):
            cv2.line(img, inner_corners[i], inner_corners[(i+1) % 4], (100, 100, 255), 1, cv2.LINE_AA)
        
        # Outer wall
        outer_corners = [to_px(0, 0), to_px(3, 0), to_px(3, 3), to_px(0, 3)]
        for i in range(4):
            cv2.line(img, outer_corners[i], outer_corners[(i+1) % 4], (255, 100, 100), 1, cv2.LINE_AA)
        
        rx, ry, ryaw = pose
        rx_px = int(rx * self.scale)
        ry_px = int(self.window_size - ry * self.scale)
        
        # 3. (Centerline distance line removed to focus on boundaries)
            
        # 4. Draw boundary lookahead points
        # Inner boundary: orange dots
        if p_inner_40 is not None:
            px_i40 = int(p_inner_40[0] * self.scale)
            py_i40 = int(self.window_size - p_inner_40[1] * self.scale)
            cv2.circle(img, (px_i40, py_i40), 6, (0, 140, 255), -1, lineType=cv2.LINE_AA)  # Orange
            cv2.line(img, (rx_px, ry_px), (px_i40, py_i40), (0, 100, 200), 1, lineType=cv2.LINE_AA)
            
        if p_inner_100 is not None:
            px_i100 = int(p_inner_100[0] * self.scale)
            py_i100 = int(self.window_size - p_inner_100[1] * self.scale)
            cv2.circle(img, (px_i100, py_i100), 6, (0, 100, 200), -1, lineType=cv2.LINE_AA)  # Dark orange
            cv2.line(img, (rx_px, ry_px), (px_i100, py_i100), (0, 80, 160), 1, lineType=cv2.LINE_AA)
            
        # Outer boundary: cyan dots
        if p_outer_40 is not None:
            px_o40 = int(p_outer_40[0] * self.scale)
            py_o40 = int(self.window_size - p_outer_40[1] * self.scale)
            cv2.circle(img, (px_o40, py_o40), 6, (255, 255, 0), -1, lineType=cv2.LINE_AA)  # Cyan
            cv2.line(img, (rx_px, ry_px), (px_o40, py_o40), (200, 200, 0), 1, lineType=cv2.LINE_AA)
            
        if p_outer_100 is not None:
            px_o100 = int(p_outer_100[0] * self.scale)
            py_o100 = int(self.window_size - p_outer_100[1] * self.scale)
            cv2.circle(img, (px_o100, py_o100), 6, (200, 200, 0), -1, lineType=cv2.LINE_AA)  # Dark cyan
            cv2.line(img, (rx_px, ry_px), (px_o100, py_o100), (160, 160, 0), 1, lineType=cv2.LINE_AA)
            
        # (Connecting lines between inner and outer boundary points removed)
            
        if p_corner is not None:
            pcr_px = int(p_corner[0] * self.scale)
            pcr_py = int(self.window_size - p_corner[1] * self.scale)
            # Draw next inner corner as a solid cyan dot
            cv2.circle(img, (pcr_px, pcr_py), 5, (255, 255, 0), -1, lineType=cv2.LINE_AA)
            
        # 4.5 Draw Obstacles from observation vector (reconstruct global position)
        if len(raw_obs) >= 15:
            alpha = ryaw + math.pi / 2.0
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            
            # Obstacle indices in new 15-element obs vector
            o1_idx, o2_idx = 9, 12
            
            # Obstacle 1
            o1_x_loc, o1_y_loc, o1_col = raw_obs[o1_idx], raw_obs[o1_idx+1], raw_obs[o1_idx+2]
            if not (abs(o1_x_loc - 2.0) < 1e-4 and abs(o1_y_loc) < 1e-4 and abs(o1_col) < 1e-4):
                ox = rx + (o1_x_loc * cos_a - o1_y_loc * sin_a)
                oy = ry + (o1_x_loc * sin_a + o1_y_loc * cos_a)
                pt1_x = int((ox - 0.025) * self.scale)
                pt1_y = int(self.window_size - (oy + 0.025) * self.scale)
                pt2_x = int((ox + 0.025) * self.scale)
                pt2_y = int(self.window_size - (oy - 0.025) * self.scale)
                color = (0, 255, 0) if o1_col > 0.1 else (0, 0, 255) if o1_col < -0.1 else (128, 128, 128)
                cv2.rectangle(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, -1)
                
                # Draw label "1" above the obstacle square
                ox_px = int(ox * self.scale)
                cv2.putText(img, "1", (ox_px - 4, pt1_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(img, "1", (ox_px - 4, pt1_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                
            # Obstacle 2
            o2_x_loc, o2_y_loc, o2_col = raw_obs[o2_idx], raw_obs[o2_idx+1], raw_obs[o2_idx+2]
            if not (abs(o2_x_loc - 2.0) < 1e-4 and abs(o2_y_loc) < 1e-4 and abs(o2_col) < 1e-4):
                ox = rx + (o2_x_loc * cos_a - o2_y_loc * sin_a)
                oy = ry + (o2_x_loc * sin_a + o2_y_loc * cos_a)
                pt1_x = int((ox - 0.025) * self.scale)
                pt1_y = int(self.window_size - (oy + 0.025) * self.scale)
                pt2_x = int((ox + 0.025) * self.scale)
                pt2_y = int(self.window_size - (oy - 0.025) * self.scale)
                color = (0, 255, 0) if o2_col > 0.1 else (0, 0, 255) if o2_col < -0.1 else (128, 128, 128)
                cv2.rectangle(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, -1)
                
                # Draw label "2" above the obstacle square
                ox_px = int(ox * self.scale)
                cv2.putText(img, "2", (ox_px - 4, pt1_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, lineType=cv2.LINE_AA)
                cv2.putText(img, "2", (ox_px - 4, pt1_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            
        # 5. Draw Robot position & heading arrow
        cv2.circle(img, (rx_px, ry_px), 8, (255, 50, 50), -1, lineType=cv2.LINE_AA) # Blue robot dot
        
        # Heading arrow vector (0.25 meters)
        L = 0.25
        ex_px = int((rx - L * math.sin(ryaw)) * self.scale)
        ey_px = int(self.window_size - (ry + L * math.cos(ryaw)) * self.scale)
        cv2.arrowedLine(img, (rx_px, ry_px), (ex_px, ey_px), (0, 255, 0), 2, cv2.LINE_AA, 0, 0.3)
        
        # 6. Create Sidebar (300px width, 600px height)
        sidebar_w = 300
        sidebar = np.zeros((self.window_size, sidebar_w, 3), dtype=np.uint8)
        # Background: very dark grey
        sidebar[:] = (25, 25, 25)
        
        # Title
        cv2.putText(sidebar, "OBSERVATION DEBUG", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(sidebar, f"Dir: {driving_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        
        # Divider line
        cv2.line(sidebar, (15, 75), (sidebar_w - 15, 75), (80, 80, 80), 1)
        
        # Table Header
        cv2.putText(sidebar, "Idx  Feature  Raw  Norm", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, lineType=cv2.LINE_AA)
        cv2.line(sidebar, (15, 105), (sidebar_w - 15, 105), (50, 50, 50), 1)
        
        labels = [
            "ego_v_x", "steer", "diff_yaw", "inn_y40", "inn_y100",
            "out_y40", "out_y100", "corner_x",
            "corner_y", "obs1_x", "obs1_y", "obs1_col",
            "obs2_x", "obs2_y", "obs2_col"
        ]
        num_feats = 15
        y_spacing = 30
        
        y_offset = 125
        for i in range(num_feats):
            label = labels[i]
            r_val = float(raw_obs[i]) if i < len(raw_obs) else 0.0
            n_val = float(obs_vector[i]) if i < len(obs_vector) else 0.0
            
            # Format lines: Idx Feature Raw Norm
            # Color logic based on normalization value
            if abs(n_val) > 0.8:
                val_color = (0, 0, 255) # Red warning for extreme values
            elif abs(n_val) > 0.01:
                val_color = (100, 255, 100) # Soft green
            else:
                val_color = (200, 200, 200) # Grey for zero/near-zero
                
            idx_str = f"{i:2d}"
            feat_str = f"{label:<10}"
            raw_str = f"{r_val: >6.2f}"
            norm_str = f"{n_val: >6.2f}"
            
            # Render index
            cv2.putText(sidebar, idx_str, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, lineType=cv2.LINE_AA)
            # Render feature name
            cv2.putText(sidebar, feat_str, (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, lineType=cv2.LINE_AA)
            # Render raw
            cv2.putText(sidebar, raw_str, (155, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, lineType=cv2.LINE_AA)
            # Render norm
            cv2.putText(sidebar, norm_str, (230, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, val_color, 1, lineType=cv2.LINE_AA)
            
            y_offset += y_spacing
            
        # Legend
        y_offset += 10
        cv2.line(sidebar, (15, y_offset), (sidebar_w - 15, y_offset), (50, 50, 50), 1)
        y_offset += 20
        cv2.circle(sidebar, (25, y_offset), 5, (0, 140, 255), -1)
        cv2.putText(sidebar, "Inner boundary", (40, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, lineType=cv2.LINE_AA)
        y_offset += 20
        cv2.circle(sidebar, (25, y_offset), 5, (255, 255, 0), -1)
        cv2.putText(sidebar, "Outer boundary", (40, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, lineType=cv2.LINE_AA)
            
        # Combine map and sidebar
        combined = np.hstack((img, sidebar))
        return combined

# Global helper function for ease of use
_global_visualizer = None

def draw_observation_window(pose, raw_obs, obs_vector, driving_direction, best_closest,
                            p_inner_40=None, p_inner_100=None, p_outer_40=None, p_outer_100=None,
                            p_corner=None, window_name="WRO Observation Debug"):
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = ObsVisualizer()
    
    combined_img = _global_visualizer.draw(
        pose, raw_obs, obs_vector, driving_direction, best_closest,
        p_inner_40, p_inner_100, p_outer_40, p_outer_100, p_corner
    )
    cv2.imshow(window_name, combined_img)
    cv2.waitKey(1)
