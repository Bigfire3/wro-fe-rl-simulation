import math
import numpy as np
import cv2

class OpenCVLocalizer:
    def __init__(self):
        self.X_real = 1.5
        self.Y_real = 0.5
        self.yaw_real = 0.0
        
        self.window_size = 600
        self.scale = 150.0
        self.lidar_img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)


    def update(self, lidar_ranges, max_range=2.0, angle_offset=0):
        """
        Wird in jedem Schleifendurchlauf vom Driver aufgerufen.
        Empfängt die aktuellen Lidar-Distanzen, verarbeitet sie und visualisiert sie.
        
        Parameter:
            lidar_ranges (list): Array/Liste von Distanzwerten des Lidar-Sensors
            max_range (float): Ignoriere Messwerte, die weiter entfernt sind
            angle_offset (float): Startwinkel für die Winkelberechnung (normalerweise pi beim Driver)
        
        Rückgabe:
            Tuple (float, float, float): Die geschätzte Pose (x_real, y_real, yaw_real)
        """
        ranges = np.array(lidar_ranges)
        n_rays = len(ranges)
        if n_rays == 0:
            return self.X_real, self.Y_real, self.yaw_real
            
        # 1. Berechnung des Winkels für jeden Strahl
        # Der Gesamtbereich von 360 Grad (2*pi) wird durch die Anzahl der Messungen geteilt.
        # Das negative Vorzeichen entspricht der Drehrichtung im Webots-Simulator.
        angle_inc = -2.0 * math.pi / n_rays
        
        # 2. Filterung ungültiger Werte
        # Wir betrachten nur Lidar-Strahlen, die sinnvolle Distanzen (z.B. > 1cm und < max_range) zurückgeben
        # und nicht unendlich oder NaN (Not a Number) sind.
        valid = (ranges > 0.01) & (ranges < max_range) & ~np.isinf(ranges) & ~np.isnan(ranges)
        
        if not np.any(valid):
            return self.X_real, self.Y_real, self.yaw_real
            
        # Nur valide Datenpunkte behalten
        r_valid = ranges[valid]
        indices = np.arange(n_rays)[valid]
        
        # Globale/Relative Winkel der Strahlen im Roboterkoordinatensystem berechnen
        rel_angles = angle_offset + indices * angle_inc
        
        # 3. Umrechnung von Polar- in lokale kartesische Koordinaten (in Metern)
        # Aus Sicht des Roboters:
        # X zeigt nach vorne (robot-forward)
        # Y zeigt nach links (robot-left)
        x_local = r_valid * np.cos(rel_angles)
        y_local = r_valid * np.sin(rel_angles)
        
        # TODO: Hier wird später die Berechnung der globalen Pose (X_real, Y_real, yaw_real) stattfinden.
        # Für diesen ersten Schritt lassen wir die Pose unverändert.
        
        # --- Visualisierung: Zeichnen der Lidar-Daten ---
        # Erstelle ein neues, schwarzes Bild (600x600, 3 Kanäle für RGB)
        self.lidar_img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        
        # Bildmittelpunkt (hier befindet sich der Roboter)
        cx = self.window_size // 2
        cy = self.window_size // 2
        
        # Zeichne den Roboter selbst als grünen Punkt in der Mitte
        cv2.circle(self.lidar_img, (cx, cy), 6, (0, 255, 0), -1)
        
        # Zeichne alle validen Lidar-Punkte in weiß
        for x, y in zip(x_local, y_local):
            # Koordinatentransformation von Meter in Pixel:
            # - Pixel-X: cx - y * scale (Da Y im Roboter nach links zeigt, schiebt ein positives Y das Pixel nach links, d.h. wir subtrahieren Y)
            # - Pixel-Y: cy - x * scale (Da X im Roboter nach vorne zeigt, schiebt ein positives X das Pixel nach oben, d.h. wir subtrahieren X)
            px = int(cx - y * self.scale)
            py = int(cy - x * self.scale)
            
            # Punkt nur zeichnen, wenn er innerhalb des Bildbereichs liegt
            if 0 <= px < self.window_size and 0 <= py < self.window_size:
                # Zeichne einen kleinen weißen Kreis für jeden Lidar-Punkt
                cv2.circle(self.lidar_img, (px, py), 2, (255, 255, 255), -1)
                
        return self.X_real, self.Y_real, self.yaw_real

    def render(self):
        """
        Gibt das gezeichnete Lidar-Bild für die Visualisierung zurück.
        Wird in wro_driver.py aufgerufen, um das Fenster anzuzeigen.
        """
        return self.lidar_img

    def create_arena_reference_image(self, padding=2.0):
        """
        Generates a reference image of the arena template using self.scale.
        """
        img_size = int((3.0 + 2.0 * padding) * self.scale)
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        
        def to_px(x, y):
            px = int((x + padding) * self.scale)
            py = int(img_size - (y + padding) * self.scale)
            return px, py

        # Draw outer boundary wall: (0,0) to (3,3) in real-world meters
        p_bl = to_px(0.0, 0.0)
        p_tr = to_px(3.0, 3.0)
        # OpenCV rectangle uses top-left and bottom-right points
        cv2.rectangle(img, (p_bl[0], p_tr[1]), (p_tr[0], p_bl[1]), 255, 2)
        
        # Draw center obstacle: (1,1) to (2,2) in real-world meters
        p_ob_bl = to_px(1.0, 1.0)
        p_ob_tr = to_px(2.0, 2.0)
        cv2.rectangle(img, (p_ob_bl[0], p_ob_tr[1]), (p_ob_tr[0], p_ob_bl[1]), 255, 2)
        
        return img

    def project_lidar_to_template(self, ranges, yaw=0.0):
        """
        Projects raw LiDAR ranges onto a template image with a given candidate yaw orientation, using self.scale.
        """
        img = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
        cx = self.window_size / 2.0
        cy = self.window_size / 2.0
        
        ranges = np.array(ranges)
        n_rays = len(ranges)
        if n_rays == 0:
            return img

        # Der Gesamtbereich von 360 Grad (2*pi) wird durch die Anzahl der Messungen geteilt.
        # Das negative Vorzeichen entspricht der Drehrichtung im Webots-Simulator.
        angle_inc = -2.0 * math.pi / n_rays

        # Filter out invalid LiDAR ranges: within [0.01, 2.0] and not nan or inf
        valid = (ranges > 0.01) & (ranges < 2.0) & ~np.isinf(ranges) & ~np.isnan(ranges)
        if not np.any(valid):
            return img
            
        r_valid = ranges[valid]
        indices = np.arange(n_rays)[valid]
        
        # Calculate angles relative to the robot (index 180 is FRONT)
        rel_angles = -math.pi + indices * angle_inc
        
        # Convert polar coordinates to Cartesian local coordinates
        x_local = r_valid * np.cos(rel_angles)
        y_local = r_valid * np.sin(rel_angles)
        
        # Rotate the local coordinates by the yaw offset candidate (where yaw=0 is North)
        # To align yaw=0 with North (+Y), we add pi/2 to the rotation angle.
        angle = math.pi / 2 - yaw
        cos_y = math.cos(angle)
        sin_y = math.sin(angle)
        x_rot = x_local * cos_y - y_local * sin_y
        y_rot = x_local * sin_y + y_local * cos_y
        
        # Project onto template image coordinates
        px = cx + x_rot * self.scale
        py = cy - y_rot * self.scale
        
        # Draw points on the image
        for x_p, y_p in zip(px, py):
            ix = int(round(x_p))
            iy = int(round(y_p))
            if 0 <= ix < self.window_size and 0 <= iy < self.window_size:
                cv2.circle(img, (ix, iy), 2, 255, -1)
                
        return img

