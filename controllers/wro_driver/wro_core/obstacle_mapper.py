import math
import cv2
import numpy as np

class Obstacle:
    def __init__(self, position):
        """
        Repräsentiert ein erkanntes Hindernis.
        
        Parameter:
            position: Globale Position des Hindernismittelpunkts als [x, y] in Metern.
        """
        self.position = position  # [x, y] in Metern (Mittelpunkt des Quadrats)
        self.size = 0.05         # Konstante Größe von 50mm
        self.confidence = 0.3     # Startwert der Confidence
        self.color = "gray"       # Initiale Farbe
        self.votes_red = 0
        self.votes_green = 0

class ObstacleMapper:
    def __init__(self):
        """
        Verwaltet die Liste der erkannten Hindernisse und führt Sichtbarkeitsprüfungen durch.
        """
        self.obstacles = []
        
        # Parameter für das Positions-Update der Hindernisse (Exponential Moving Average)
        # Höherer Wert = schnelleres Anpassen (sensibler), niedrigerer Wert = stabiler/rauschärmer
        self.position_alpha = 0.1
        
        # Statische Segmente der Arena (Wände) definieren für die Sichtbarkeitsprüfung
        self.segments = [
            # Außenwände (3.0m x 3.0m)
            ("H", 0.0, 3.0, 0.0), # Süd
            ("H", 0.0, 3.0, 3.0), # Nord
            ("V", 0.0, 3.0, 0.0), # West
            ("V", 0.0, 3.0, 3.0), # Ost
            # Innenwände (1.0m x 1.0m Box in der Mitte: [1.0, 2.0] x [1.0, 2.0])
            ("H", 1.0, 2.0, 1.0), # Süd
            ("H", 1.0, 2.0, 2.0), # Nord
            ("V", 1.0, 2.0, 1.0), # West
            ("V", 1.0, 2.0, 2.0)  # Ost
        ]

    def is_visible(self, rx, ry, ox, oy, target_obs=None):
        """
        Prüft, ob die Sichtlinie zwischen Roboter (rx, ry) und Ziel (ox, oy)
        durch statische Wandsegmente oder andere Hindernisse verdeckt wird.
        """
        # 1. Wände prüfen
        for seg_type, lim1, lim2, val in self.segments:
            if seg_type == "H":
                # Horizontale Wand: y = val, x in [lim1, lim2]
                if min(ry, oy) <= val <= max(ry, oy):
                    if abs(oy - ry) < 1e-7:
                        # Falls Sichtlinie exakt horizontal
                        if abs(ry - val) < 1e-7:
                            if max(min(rx, ox), lim1) <= min(max(rx, ox), lim2):
                                return False
                    else:
                        # Schnittpunkt der Sichtlinie mit der Wandgeraden berechnen
                        x_inter = rx + (val - ry) * (ox - rx) / (oy - ry)
                        if lim1 <= x_inter <= lim2 and min(rx, ox) <= x_inter <= max(rx, ox):
                            return False
            else: # "V"
                # Vertikale Wand: x = val, y in [lim1, lim2]
                if min(rx, ox) <= val <= max(rx, ox):
                    if abs(ox - rx) < 1e-7:
                        # Falls Sichtlinie exakt vertikal
                        if abs(rx - val) < 1e-7:
                            if max(min(ry, oy), lim1) <= min(max(ry, oy), lim2):
                                return False
                    else:
                        # Schnittpunkt der Sichtlinie mit der Wandgeraden berechnen
                        y_inter = ry + (val - rx) * (oy - ry) / (ox - rx)
                        if lim1 <= y_inter <= lim2 and min(ry, oy) <= y_inter <= max(ry, oy):
                            return False

        # 2. Andere Hindernisse prüfen
        dist_rt_sq = (ox - rx)**2 + (oy - ry)**2
        if dist_rt_sq < 1e-9:
            return True

        for obs in self.obstacles:
            if target_obs is not None and obs is target_obs:
                continue
            # Falls target_obs nicht übergeben wurde, aber das Hindernis obs extrem nah an (ox, oy) ist,
            # schließen wir es aus, um Selbst-Okklusion zu vermeiden.
            obs_x, obs_y = obs.position
            if math.hypot(obs_x - ox, obs_y - oy) < 0.01:
                continue

            dist_ro_sq = (obs_x - rx)**2 + (obs_y - ry)**2

            # Ist das Hindernis näher am Roboter als das Ziel?
            if dist_ro_sq >= dist_rt_sq:
                continue

            # Skalarprojektion t berechnen, um zu prüfen, ob der Fußpunkt auf der Strecke RT liegt
            dx_rt = ox - rx
            dy_rt = oy - ry
            dx_ro = obs_x - rx
            dy_ro = obs_y - ry

            t = (dx_ro * dx_rt + dy_ro * dy_rt) / dist_rt_sq

            if 0.0 <= t <= 1.0:
                # Orthogonaler Abstand d zur Sichtlinie RT berechnen
                numerator = abs(dy_rt * obs_x - dx_rt * obs_y + ox * ry - oy * rx)
                denominator = math.sqrt(dist_rt_sq)
                d = numerator / denominator

                # Verdeckungskriterium: Radius des Hindernisses (0.025m)
                if d < 0.025:
                    return False

        return True

    def get_edge_visibilities(self, rx, ry, target_obs):
        """
        Berechnet die Sichtbarkeit und Positionen der Kanten des Hindernisses.
        """
        ox, oy = target_obs.position
        theta = math.atan2(oy - ry, ox - rx)
        
        # Versatz senkrecht zur Sichtlinie (Radius des Hindernisses = 0.025m)
        dx_perp = -math.sin(theta) * 0.025
        dy_perp = math.cos(theta) * 0.025
        
        pt_left_x, pt_left_y = ox + dx_perp, oy + dy_perp
        pt_right_x, pt_right_y = ox - dx_perp, oy - dy_perp
        
        visible_left = self.is_visible(rx, ry, pt_left_x, pt_left_y, target_obs)
        visible_right = self.is_visible(rx, ry, pt_right_x, pt_right_y, target_obs)
        
        return visible_left, visible_right, (pt_left_x, pt_left_y), (pt_right_x, pt_right_y)

    def is_fully_visible(self, rx, ry, target_obs):
        """
        Prüft, ob das Ziel-Hindernis zu 100% sichtbar ist (keine Teilverdeckungen).
        Beide Kanten (links und rechts) müssen unverdeckt sein.
        """
        visible_left, visible_right, _, _ = self.get_edge_visibilities(rx, ry, target_obs)
        return visible_left and visible_right

    def update(self, robot_pose, outlier_points):
        """
        Aktualisiert das Hindernis-Mapping basierend auf den neuen Outlier-Punkten
        und der geschätzten Pose des Roboters.
        """
        robot_x, robot_y, robot_yaw = robot_pose

        # Schritt 1: Clustering (abstandsbasierte Gruppierung der Outlier-Punkte)
        clusters = []
        visited = set()
        for i, p1 in enumerate(outlier_points):
            if i in visited:
                continue
            cluster = [p1]
            visited.add(i)
            queue = [p1]
            while queue:
                curr = queue.pop(0)
                for j, p2 in enumerate(outlier_points):
                    if j not in visited:
                        dist = math.hypot(curr[0] - p2[0], curr[1] - p2[1])
                        if dist < 0.1:  # Schwellenwert 0.1m für Nachbarschaft
                            visited.add(j)
                            cluster.append(p2)
                            queue.append(p2)
            clusters.append(cluster)

        # Filtern: Nur Cluster mit >= 2 Punkten weiterverarbeiten (Rauschfilter)
        valid_clusters = [c for c in clusters if len(c) >= 2]

        # Schritt 2: Datenassoziation & Positions-Update
        confirmed_obstacles = set()
        for cluster in valid_clusters:
            # Schwerpunkt (Centroid) bestimmen
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)

            best_obs = None
            min_dist = 0.15
            for obs in self.obstacles:
                dist = math.hypot(obs.position[0] - cx, obs.position[1] - cy)
                if dist < min_dist:
                    min_dist = dist
                    best_obs = obs

            if best_obs is not None:
                # Bestehendem Hindernis zuweisen und Confidence erhöhen
                best_obs.confidence = min(1.0, best_obs.confidence + 0.01)

                # Position des Hindernisses mit einem Tiefpassfilter (EMA) anpassen,
                # um schneller, genauer und sensibler auf Positionsänderungen zu reagieren
                ox, oy = best_obs.position
                ox = ox + self.position_alpha * (cx - ox)
                oy = oy + self.position_alpha * (cy - oy)

                best_obs.position = [ox, oy]
                confirmed_obstacles.add(best_obs)
            else:
                # Neues Hindernis anlegen
                new_obs = Obstacle([cx, cy])
                self.obstacles.append(new_obs)
                confirmed_obstacles.add(new_obs)

        # Schritt 3: Sichtbarkeitsprüfung & Decay
        for obs in list(self.obstacles):
            if obs in confirmed_obstacles:
                continue

            # Prüfen, ob im Sichtbereich (Distanz < 2.0m)
            dist = math.hypot(robot_x - obs.position[0], robot_y - obs.position[1])
            if dist < 2.0:
                # Prüfen, ob Sichtachse blockiert ist (Wände oder andere Hindernisse)
                if self.is_fully_visible(robot_x, robot_y, obs):
                    obs.confidence -= 0.01
                    if obs.confidence <= 0.0:
                        if obs in self.obstacles:
                            self.obstacles.remove(obs)

    def update_obstacle_colors(self, camera_image, robot_pose):
        """
        Aktualisiert die Farbstimmen der Hindernisse basierend auf dem Kamerabild.
        """
        if camera_image is None:
            return

        h_img, w_img, _ = camera_image.shape
        rx, ry, robot_yaw = robot_pose
        
        # FOV und Brennweiten-Berechnung
        FOV = 2.7925268
        f_y = w_img / (2.0 * math.tan(FOV / 2.0))
        
        # alpha = robot_yaw + pi/2
        alpha = robot_yaw + math.pi / 2.0
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        
        for obs in self.obstacles:
            # 1. Farbe bereits gesperrt?
            if obs.color in ["red", "green"]:
                continue
                
            # 2. Sichtbarkeit prüfen (Wände und andere Hindernisse)
            ox, oy = obs.position
            if not self.is_fully_visible(rx, ry, obs):
                continue
                
            # 3. Transformation ins Roboterkoordinatensystem (mit alpha)
            dx = ox - rx
            dy = oy - ry
            x_local = dx * cos_alpha + dy * sin_alpha
            y_local = -dx * sin_alpha + dy * cos_alpha
            
            # 4. Transformation ins Kamerakoordinatensystem (Kamera ist 0.09m vor dem Lidar)
            x_cam = x_local - 0.09
            y_cam = y_local
            
            # 5. Liegt das Hindernis vor der Kamera?
            if x_cam <= 0:
                continue
                
            # 6. Horizontaler Winkel
            theta_cam = math.atan2(y_cam, x_cam)
            
            # 7. Im Sichtfeld der Kamera?
            if abs(theta_cam) >= FOV / 2.0:
                continue
                
            # 8. Projektion auf Bildkoordinaten u, v (Pinhole-Projektion, Hinderniszentrum ist 0.01m über Kamera)
            u = int(w_img / 2.0 - f_y * y_cam / x_cam)
            v = int(h_img / 2.0 - f_y * 0.01 / x_cam)
            
            # 9. ROI-Größe bestimmen (Rechteck: höher und etwas schmaler, z. B. Breite 0.03m und Höhe 0.06m)
            # Enforce minimum ROI size to prevent empty ROIs with wide-angle cameras
            box_w = max(3, int(f_y * 0.02 / x_cam))
            box_h = max(6, int(f_y * 0.08 / x_cam))
                
            # ROI Boxgrenzen bestimmen
            u_min = u - box_w // 2
            u_max = u + box_w // 2
            v_min = v - box_h // 2
            v_max = v + box_h // 2
            
            # Beschränken auf Bildgrenzen
            u_min_clamped = max(0, u_min)
            u_max_clamped = min(w_img - 1, u_max)
            v_min_clamped = max(0, v_min)
            v_max_clamped = min(h_img - 1, v_max)
            
            # Falls die ROI leer oder ungültig ist, überspringen
            if u_max_clamped <= u_min_clamped or v_max_clamped <= v_min_clamped:
                continue
                
            # ROI ausschneiden
            roi = camera_image[v_min_clamped:v_max_clamped+1, u_min_clamped:u_max_clamped+1]
            
            # ROI in HSV konvertieren
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Sättigung (S) und Helligkeit (V) Filter:
            # Ignoriere S < 50, V < 40, V > 245
            s_channel = roi_hsv[:, :, 1]
            v_channel = roi_hsv[:, :, 2]
            valid_mask = (s_channel >= 50) & (v_channel >= 40) & (v_channel <= 245)
            
            h_channel = roi_hsv[:, :, 0]
            
            # Masken für Rot und Grün
            green_mask = valid_mask & (h_channel >= 35) & (h_channel <= 85)
            red_mask = valid_mask & ((h_channel <= 10) | (h_channel >= 170))
            
            n_green = np.sum(green_mask)
            n_red = np.sum(red_mask)
            
            # Gesamtzahl der Pixel im ROI
            total_pixels = roi.shape[0] * roi.shape[1]
            min_area_pixels = 0.10 * total_pixels
            
            # Voting-Entscheidung
            if n_red > 2 * n_green and n_red >= min_area_pixels:
                obs.votes_red += 1
            elif n_green > 2 * n_red and n_green >= min_area_pixels:
                obs.votes_green += 1
                
            # Temporaler Akkumulator / Lock
            if obs.votes_red >= 10:
                obs.color = "red"
            elif obs.votes_green >= 10:
                obs.color = "green"

    def render_camera(self, camera_image, robot_pose):
        """
        Erstellt ein Debug-Kamerabild mit Overlays für projizierte Hindernisse.
        """
        if camera_image is None:
            return None
            
        img = camera_image.copy()
        h_img, w_img, _ = img.shape
        rx, ry, robot_yaw = robot_pose
        
        FOV = 2.7925268
        f_y = w_img / (2.0 * math.tan(FOV / 2.0))
        
        alpha = robot_yaw + math.pi / 2.0
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        
        def draw_text(img, text, pos, color):
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
        for obs in self.obstacles:
            ox, oy = obs.position
            dx = ox - rx
            dy = oy - ry
            
            # Transformation ins Roboterkoordinatensystem
            x_local = dx * cos_alpha + dy * sin_alpha
            y_local = -dx * sin_alpha + dy * cos_alpha
            
            # Transformation ins Kamerakoordinatensystem (Kamera ist 0.09m vor dem Lidar)
            x_cam = x_local - 0.09
            y_cam = y_local
            
            # Vor der Kamera?
            if x_cam <= 0:
                continue
                
            # Horizontaler Winkel
            theta_cam = math.atan2(y_cam, x_cam)
            
            # Im FOV?
            if abs(theta_cam) >= FOV / 2.0:
                continue
                
            # Projektion auf Bildkoordinaten u, v (Pinhole-Projektion, Hinderniszentrum ist 0.01m über Kamera)
            u = int(w_img / 2.0 - f_y * y_cam / x_cam)
            v = int(h_img / 2.0 - f_y * 0.01 / x_cam)
            
            # ROI Box-Größe (Rechteck: höher und etwas schmaler, z. B. Breite 0.03m und Höhe 0.06m)
            # Enforce minimum ROI size to prevent empty ROIs with wide-angle cameras
            box_w = max(3, int(f_y * 0.02 / x_cam))
            box_h = max(6, int(f_y * 0.08 / x_cam))
                
            # Sichtbarkeit prüfen
            if not self.is_fully_visible(rx, ry, obs):
                continue
            
            # Bounding Box Farbe bestimmen
            if obs.color == "red":
                bbox_color = (0, 0, 255) # Rot
            elif obs.color == "green":
                bbox_color = (0, 255, 0) # Grün
            else:
                bbox_color = (128, 128, 128) # Grau
                
            # 1. Vertikale Spaltenlinie
            cv2.line(img, (u, 0), (u, h_img), (255, 255, 0), 1)
            
            # 2. Kurze horizontale Markierungslinie
            cv2.line(img, (max(0, u - 20), v), (min(w_img - 1, u + 20), v), (255, 255, 0), 1)
            
            # 3. Bounding Box (ROI) zeichnen
            cv2.rectangle(img, (u - box_w//2, v - box_h//2), (u + box_w//2, v + box_h//2), bbox_color, 2)
            
            # 4. Text Overlay zeichnen
            dist = math.hypot(dx, dy)
            text_x = u + box_w // 2 + 5
            text_y = v - box_h // 2 + 12
            
            # Zeile 1: Distanz
            draw_text(img, f"d: {dist:.2f}m", (text_x, text_y), (255, 255, 255))
            text_y += 12
            
            # Zeile 2: Stimmen
            draw_text(img, f"R:{obs.votes_red} G:{obs.votes_green}", (text_x, text_y), (255, 255, 255))
            text_y += 12
            
            # Zeile 3: Status
            if obs.color in ["red", "green"]:
                draw_text(img, f"Locked: {obs.color.upper()}", (text_x, text_y), bbox_color)
            else:
                draw_text(img, "Scanning...", (text_x, text_y), (0, 255, 255))
                
        return img

    def render(self, img, robot_pose, scale, window_size):
        """
        Zeichnet die Hindernisse als gefüllte Quadrate auf das Debug-Bild
        und zeichnet ggf. Sichtverbindungslinien zum Roboter.
        """
        robot_x, robot_y, robot_yaw = robot_pose

        for obs in self.obstacles:
            ox, oy = obs.position

            # Bestimme die Farbe basierend auf der klassifizierten Farbe oder Confidence
            if obs.color == "red":
                color = (0, 0, 255)
            elif obs.color == "green":
                color = (0, 255, 0)
            else:
                gray_value = int(200 * (1 - obs.confidence))
                color = (gray_value, gray_value, gray_value)

            # Quadratecken berechnen (in Pixeln)
            pt1_x = int((ox - 0.025) * scale)
            pt1_y = int(window_size - (oy + 0.025) * scale)
            pt2_x = int((ox + 0.025) * scale)
            pt2_y = int(window_size - (oy - 0.025) * scale)

            # Gefülltes Quadrat zeichnen
            cv2.rectangle(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, -1)

            # Sichtlinie zeichnen, falls im Sichtbereich und mindestens ein Strahl frei
            dist = math.hypot(robot_x - ox, robot_y - oy)
            if dist < 2.0:
                visible_left, visible_right, pt_left, pt_right = self.get_edge_visibilities(robot_x, robot_y, obs)
                
                # Wenn beide Strahlen rot/blockiert sind, zeichnen wir sie nicht
                if visible_left or visible_right:
                    # Roboter-Pixelkoordinaten
                    rx_px = int(robot_x * scale)
                    ry_px = int(window_size - robot_y * scale)
                    
                    # Pixelkoordinaten der Ränder
                    pl_x_px = int(pt_left[0] * scale)
                    pl_y_px = int(window_size - pt_left[1] * scale)
                    pr_x_px = int(pt_right[0] * scale)
                    pr_y_px = int(window_size - pt_right[1] * scale)
                    
                    # Linken Strahl zeichnen (Grün wenn frei, Rot wenn blockiert)
                    color_l = (0, 255, 0) if visible_left else (0, 0, 255)
                    cv2.line(img, (rx_px, ry_px), (pl_x_px, pl_y_px), color_l, 1, lineType=cv2.LINE_AA)
                    
                    # Rechten Strahl zeichnen (Grün wenn frei, Rot wenn blockiert)
                    color_r = (0, 255, 0) if visible_right else (0, 0, 255)
                    cv2.line(img, (rx_px, ry_px), (pr_x_px, pr_y_px), color_r, 1, lineType=cv2.LINE_AA)

        return img
