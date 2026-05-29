import math
import os
import numpy as np
import cv2

class TranslationICPLocalizer:
    def __init__(self):
        # Initial-Pose
        self.X_real = 1.5
        self.Y_real = 0.5
        self.yaw_real = 0.0
        
        # Offset zwischen IMU-Yaw (Webots-Welt) und Arena-Yaw
        # Wird bei set_initial_pose berechnet: yaw_init - imu_yaw_init
        self.imu_to_world_offset = 0.0
        
        self.window_size = 600
        # Spielfeld ist 3.0m x 3.0m. Bei 600x600 Pixeln: 200 Pixel/Meter.
        self.scale = 200.0
        
        # Statische Segmente der Arena (Wände) definieren
        # Format: (Typ, Limit1, Limit2, Konstante_Koordinate)
        # Typ "H": y = Konstante_Koordinate, x zwischen Limit1 und Limit2
        # Typ "V": x = Konstante_Koordinate, y zwischen Limit1 und Limit2
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
        
        # Hintergrundbild (Spielfeld.png) laden
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, "..", "..", "..", "worlds", "textures", "Spielfeld.png")
        
        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                # In Graustufen konvertieren
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                # Auf 600x600 skalieren
                self.bg_img = cv2.resize(img_gray, (self.window_size, self.window_size))
            else:
                print(f"[TranslationICPLocalizer] Warnung: Konnte '{img_path}' nicht laden.")
                self.bg_img = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
        else:
            print(f"[TranslationICPLocalizer] Warnung: Datei existiert nicht: '{img_path}'")
            self.bg_img = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
            
        # Letzte klassifizierte Punkte für die Visualisierung speichern
        self.point_types = []  # Liste von (x_glob, y_glob, is_inlier)
        self.trajectory_history = []  # Liste von (x_real, y_real) für die Fahrspur

    def set_initial_pose(self, x, y, yaw):
        """
        Setzt die Startpose nach der Kalibrierung.
        Da die IMU genullt ist (wie eine echte IMU), ist yaw_init direkt der Offset.
        """
        self.X_real = x
        self.Y_real = y
        self.yaw_real = yaw
        # IMU ist genullt (startet bei 0), daher: world_yaw = yaw_init + imu_yaw
        self.imu_to_world_offset = yaw
        self.trajectory_history = [(x, y)]

    def update(self, lidar_ranges, imu_yaw, max_range=2.0):
        """
        Aktualisiert die geschätzte Translation (X_real, Y_real) über 3 ICP-Iterationen.
        Der absolute Winkel wird aus dem IMU-Yaw und dem gespeicherten Offset berechnet.
        
        Parameter:
            lidar_ranges: LiDAR-Distanzmessungen
            imu_yaw: Aktueller absoluter IMU-Yaw aus Webots
            max_range: Maximale LiDAR-Reichweite
        """
        ranges = np.array(lidar_ranges)
        n_rays = len(ranges)
        if n_rays == 0:
            return self.X_real, self.Y_real, self.yaw_real, []

        # 1. LiDAR-Strahlen filtern (valide Reichweite [0.01, max_range])
        valid = (ranges > 0.01) & (ranges < max_range) & ~np.isinf(ranges) & ~np.isnan(ranges)
        if not np.any(valid):
            return self.X_real, self.Y_real, self.yaw_real, []

        r_valid = ranges[valid]
        indices = np.arange(n_rays)[valid]

        # 2. Lokale kartesische Koordinaten berechnen (wie in project_lidar_to_template)
        angle_inc = -2.0 * math.pi / n_rays
        rel_angles = -math.pi + indices * angle_inc
        
        x_local = r_valid * np.cos(rel_angles)
        y_local = r_valid * np.sin(rel_angles)

        # 3. Arena-Yaw aus IMU-Yaw berechnen
        # world_yaw = imu_to_world_offset + imu_yaw
        # (offset wurde bei set_initial_pose aus yaw_init - imu_yaw_init berechnet)
        world_yaw = self.imu_to_world_offset + imu_yaw
        # Rotation: angle = pi/2 + world_yaw (identisch zu project_lidar_to_template)
        angle = math.pi / 2 + world_yaw
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Punkte einmal rotieren (da Rotation fix ist)
        dx_rot = x_local * cos_a - y_local * sin_a
        dy_rot = x_local * sin_a + y_local * cos_a

        x_est = self.X_real
        y_est = self.Y_real
        outliers = []

        # 4. Translation-Only ICP-Schleife über genau 3 Iterationen
        for iteration in range(3):
            # Projiziere Punkte mit der aktuellen Positionsschätzung
            x_glob = x_est + dx_rot
            y_glob = y_est + dy_rot

            dx_list = []
            dy_list = []

            # Speichere die Punktklassifikationen in der letzten Iteration für die Visualisierung
            if iteration == 2:
                self.point_types = []
                outliers = []

            for px, py in zip(x_glob, y_glob):
                min_dist = float('inf')
                best_seg_type = None
                best_diff = 0.0

                # Finde das nächstgelegene Segment
                for seg_type, lim1, lim2, val in self.segments:
                    if seg_type == "H":
                        proj_x = max(lim1, min(lim2, px))
                        dist = math.hypot(px - proj_x, py - val)
                        if dist < min_dist:
                            min_dist = dist
                            best_seg_type = "H"
                            best_diff = val - py
                    else: # "V"
                        proj_y = max(lim1, min(lim2, py))
                        dist = math.hypot(px - val, py - proj_y)
                        if dist < min_dist:
                            min_dist = dist
                            best_seg_type = "V"
                            best_diff = val - px

                is_inlier = (min_dist < 0.15)
                if is_inlier:
                    if best_seg_type == "H":
                        dy_list.append(best_diff)
                    else:
                        dx_list.append(best_diff)

                if iteration == 2:
                    self.point_types.append((px, py, is_inlier))
                    if not is_inlier:
                        if 0.0 <= px <= 3.0 and 0.0 <= py <= 3.0:
                            outliers.append([px, py])

            # Korrektur mittels Median berechnen und anwenden
            if len(dx_list) > 0:
                x_est += np.median(dx_list)
            if len(dy_list) > 0:
                y_est += np.median(dy_list)

        # Pose innerhalb der Arena begrenzen
        self.X_real = max(0.0, min(3.0, x_est))
        self.Y_real = max(0.0, min(3.0, y_est))
        # Normalisieren auf [-pi, pi]
        self.yaw_real = (world_yaw + math.pi) % (2.0 * math.pi) - math.pi

        # Fahrspur aktualisieren
        if len(self.trajectory_history) == 0:
            self.trajectory_history.append((self.X_real, self.Y_real))
        else:
            last_x, last_y = self.trajectory_history[-1]
            dist = math.hypot(self.X_real - last_x, self.Y_real - last_y)
            if dist > 0.02:
                self.trajectory_history.append((self.X_real, self.Y_real))
                if len(self.trajectory_history) > 5000:
                    self.trajectory_history.pop(0)

        return self.X_real, self.Y_real, self.yaw_real, outliers

    def render(self, max_range=2.0):
        """
        Erzeugt das Debug-Bild auf Basis der Graustufen-Textur von Spielfeld.png.
        """
        # Konvertiere Graustufen-Hintergrund in BGR für farbige Overlays
        img = cv2.cvtColor(self.bg_img, cv2.COLOR_GRAY2BGR)

        # 1. LiDAR-Punkte zeichnen
        for px, py, is_inlier in self.point_types:
            px_img = int(px * self.scale)
            py_img = int(self.window_size - py * self.scale)
            
            # Farbe: Grün für Inliers (Wände), Rot für Outliers (Hindernisse/Rauschen)
            color = (0, 255, 0) if is_inlier else (0, 0, 255)
            cv2.circle(img, (px_img, py_img), 2, color, -1)

        # 1.5. Fahrspur (Trajektorie) zeichnen
        if len(self.trajectory_history) > 1:
            pts = []
            for tx, ty in self.trajectory_history:
                tx_px = int(tx * self.scale)
                ty_px = int(self.window_size - ty * self.scale)
                pts.append([tx_px, ty_px])
            pts = np.array(pts, dtype=np.int32)
            # Orange-farbene Spur mit Anti-Aliasing (dünnere Linie)
            cv2.polylines(img, [pts], isClosed=False, color=(0, 140, 255), thickness=1, lineType=cv2.LINE_AA)

        # 2. Roboter-Position zeichnen
        rx_px = int(self.X_real * self.scale)
        ry_px = int(self.window_size - self.Y_real * self.scale)
        # Blauer Kreis für den Roboter (mit Anti-Aliasing)
        cv2.circle(img, (rx_px, ry_px), 8, (255, 100, 0), -1, lineType=cv2.LINE_AA)

        # 3. Ausrichtung (Yaw) zeichnen
        # Vektorlänge in Metern
        L = 0.25
        ex_px = int((self.X_real - L * math.sin(self.yaw_real)) * self.scale)
        ey_px = int(self.window_size - (self.Y_real + L * math.cos(self.yaw_real)) * self.scale)
        # Rote Linie für die Ausrichtung
        cv2.line(img, (rx_px, ry_px), (ex_px, ey_px), (0, 0, 255), 1, lineType=cv2.LINE_AA)

        # 4. LiDAR-Reichweitenkreis (max_range) zeichnen
        # Dünner grauer Kreis
        cv2.circle(img, (rx_px, ry_px), int(max_range * self.scale), (180, 180, 180), 1, lineType=cv2.LINE_AA)

        return img

