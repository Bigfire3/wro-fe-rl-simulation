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
        img_path = os.path.abspath(os.path.join(script_dir, "..", "..", "worlds", "textures", "Spielfeld.png"))
        
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

    def draw(self, pose, raw_obs, obs_vector, driving_direction, best_closest, p_30, p_60):
        """
        Draws the observation debug window.
        
        Parameters:
            pose: tuple of (rx, ry, ryaw)
            raw_obs: numpy array of 12 elements (unnormalized)
            obs_vector: numpy array of 12 elements (normalized and clipped)
            driving_direction: "CCW" or "CW"
            best_closest: coordinates of closest point on path (x, y) or None
            p_30: coordinates of lookahead point 30cm (x, y) or None
            p_60: coordinates of lookahead point 60cm (x, y) or None
        """
        # 1. Convert grayscale background to BGR
        img = cv2.cvtColor(self.bg_img, cv2.COLOR_GRAY2BGR)
        
        # 2. Draw Ideal Line (the 2x2 meter square track)
        # Vertices in real-world coordinates: (0.5, 0.5) to (2.5, 2.5)
        pts_ideal = np.array([
            [int(0.5 * self.scale), int(self.window_size - 0.5 * self.scale)],
            [int(2.5 * self.scale), int(self.window_size - 0.5 * self.scale)],
            [int(2.5 * self.scale), int(self.window_size - 2.5 * self.scale)],
            [int(0.5 * self.scale), int(self.window_size - 2.5 * self.scale)]
        ], dtype=np.int32)
        cv2.polylines(img, [pts_ideal], isClosed=True, color=(255, 180, 50), thickness=2, lineType=cv2.LINE_AA)
        
        rx, ry, ryaw = pose
        rx_px = int(rx * self.scale)
        ry_px = int(self.window_size - ry * self.scale)
        
        # 3. Draw distance line from robot to closest point
        if best_closest is not None:
            cx_px = int(best_closest[0] * self.scale)
            cy_px = int(self.window_size - best_closest[1] * self.scale)
            cv2.line(img, (rx_px, ry_px), (cx_px, cy_px), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(img, (cx_px, cy_px), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            
        # 4. Draw lookahead points (30cm & 60cm)
        if p_30 is not None:
            p30_px = int(p_30[0] * self.scale)
            p30_py = int(self.window_size - p_30[1] * self.scale)
            cv2.circle(img, (p30_px, p30_py), 6, (0, 255, 255), -1, lineType=cv2.LINE_AA) # Yellow dot
            cv2.line(img, (rx_px, ry_px), (p30_px, p30_py), (0, 200, 200), 1, lineType=cv2.LINE_AA)
            
        if p_60 is not None:
            p60_px = int(p_60[0] * self.scale)
            p60_py = int(self.window_size - p_60[1] * self.scale)
            cv2.circle(img, (p60_px, p60_py), 6, (0, 165, 255), -1, lineType=cv2.LINE_AA) # Orange dot
            cv2.line(img, (rx_px, ry_px), (p60_px, p60_py), (0, 120, 220), 1, lineType=cv2.LINE_AA)
            
        # 4.5 Draw Obstacles from observation vector (reconstruct global position)
        if len(raw_obs) >= 12:
            alpha = ryaw + math.pi / 2.0
            cos_a = math.cos(alpha)
            sin_a = math.sin(alpha)
            
            # Obstacle 1
            o1_x_loc, o1_y_loc, o1_col = raw_obs[6], raw_obs[7], raw_obs[8]
            if not (abs(o1_x_loc - 2.0) < 1e-4 and abs(o1_y_loc) < 1e-4 and abs(o1_col) < 1e-4):
                ox = rx + (o1_x_loc * cos_a - o1_y_loc * sin_a)
                oy = ry + (o1_x_loc * sin_a + o1_y_loc * cos_a)
                pt1_x = int((ox - 0.025) * self.scale)
                pt1_y = int(self.window_size - (oy + 0.025) * self.scale)
                pt2_x = int((ox + 0.025) * self.scale)
                pt2_y = int(self.window_size - (oy - 0.025) * self.scale)
                color = (0, 255, 0) if o1_col > 0.1 else (0, 0, 255) if o1_col < -0.1 else (128, 128, 128)
                cv2.rectangle(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, -1)
                
            # Obstacle 2
            o2_x_loc, o2_y_loc, o2_col = raw_obs[9], raw_obs[10], raw_obs[11]
            if not (abs(o2_x_loc - 2.0) < 1e-4 and abs(o2_y_loc) < 1e-4 and abs(o2_col) < 1e-4):
                ox = rx + (o2_x_loc * cos_a - o2_y_loc * sin_a)
                oy = ry + (o2_x_loc * sin_a + o2_y_loc * cos_a)
                pt1_x = int((ox - 0.025) * self.scale)
                pt1_y = int(self.window_size - (oy + 0.025) * self.scale)
                pt2_x = int((ox + 0.025) * self.scale)
                pt2_y = int(self.window_size - (oy - 0.025) * self.scale)
                color = (0, 255, 0) if o2_col > 0.1 else (0, 0, 255) if o2_col < -0.1 else (128, 128, 128)
                cv2.rectangle(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, -1)
            
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
            "ego_v_x", "steer", "lat_err", "hdg_err",
            "y_loc_30", "y_loc_60", "obs1_x", "obs1_y",
            "obs1_col", "obs2_x", "obs2_y", "obs2_col"
        ]
        
        y_offset = 130
        for i in range(12):
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
            cv2.putText(sidebar, idx_str, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, lineType=cv2.LINE_AA)
            # Render feature name
            cv2.putText(sidebar, feat_str, (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, lineType=cv2.LINE_AA)
            # Render raw
            cv2.putText(sidebar, raw_str, (155, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, lineType=cv2.LINE_AA)
            # Render norm
            cv2.putText(sidebar, norm_str, (230, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, val_color, 1, lineType=cv2.LINE_AA)
            
            y_offset += 35
            
        # Combine map and sidebar
        combined = np.hstack((img, sidebar))
        return combined

# Global helper function for ease of use
_global_visualizer = None

def draw_observation_window(pose, raw_obs, obs_vector, driving_direction, best_closest, p_30, p_60, window_name="WRO Observation Debug"):
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = ObsVisualizer()
    
    combined_img = _global_visualizer.draw(pose, raw_obs, obs_vector, driving_direction, best_closest, p_30, p_60)
    cv2.imshow(window_name, combined_img)
    cv2.waitKey(1)
