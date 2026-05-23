import math
import cv2

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

class ObstacleMapper:
    def __init__(self):
        """
        Verwaltet die Liste der erkannten Hindernisse und führt Sichtbarkeitsprüfungen durch.
        """
        self.obstacles = []
        
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

    def is_visible(self, rx, ry, ox, oy):
        """
        Prüft, ob die Sichtlinie zwischen Roboter (rx, ry) und Hindernis (ox, oy)
        durch eines der statischen Wandsegmente verdeckt wird.
        """
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
        return True

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

                # Minimaler Shift der Position des Hindernisses
                ox, oy = best_obs.position
                half_size = 0.025  # 50mm / 2
                
                if cx > ox + half_size:
                    ox = cx - half_size
                elif cx < ox - half_size:
                    ox = cx + half_size

                if cy > oy + half_size:
                    oy = cy - half_size
                elif cy < oy - half_size:
                    oy = cy + half_size

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
                # Prüfen, ob Sichtachse durch Wände blockiert ist
                if self.is_visible(robot_x, robot_y, obs.position[0], obs.position[1]):
                    obs.confidence -= 0.01
                    if obs.confidence <= 0.0:
                        if obs in self.obstacles:
                            self.obstacles.remove(obs)

    def render(self, img, robot_pose, scale, window_size):
        """
        Zeichnet die Hindernisse als gefüllte Quadrate auf das Debug-Bild
        und zeichnet ggf. Sichtverbindungslinien zum Roboter.
        """
        robot_x, robot_y, robot_yaw = robot_pose

        for obs in self.obstacles:
            ox, oy = obs.position

            # Bestimme die Grautönung basierend auf der Confidence
            gray_value = int(200 * (1 - obs.confidence))
            color = (gray_value, gray_value, gray_value)

            # Quadratecken berechnen (in Pixeln)
            pt1_x = int((ox - 0.025) * scale)
            pt1_y = int(window_size - (oy + 0.025) * scale)
            pt2_x = int((ox + 0.025) * scale)
            pt2_y = int(window_size - (oy - 0.025) * scale)

            # Gefülltes Quadrat zeichnen
            cv2.rectangle(img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, -1)

            # Sichtlinie zeichnen, falls im Sichtbereich und Sichtachse frei
            dist = math.hypot(robot_x - ox, robot_y - oy)
            if dist < 2.0:
                if self.is_visible(robot_x, robot_y, ox, oy):
                    # Roboter-Pixelkoordinaten
                    rx_px = int(robot_x * scale)
                    ry_px = int(window_size - robot_y * scale)
                    # Hindernis-Mittelpunkt
                    ox_px = int(ox * scale)
                    oy_px = int(window_size - oy * scale)
                    # Dünne grüne Linie
                    cv2.line(img, (rx_px, ry_px), (ox_px, oy_px), (0, 255, 0), 1, lineType=cv2.LINE_AA)

        return img
