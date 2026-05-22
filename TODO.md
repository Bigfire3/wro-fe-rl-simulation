# WRO Future Engineers 2026 - Calibration Plan & TODOs

Dieses Dokument fasst den Implementierungsplan für die initiale Yaw- und Positionskalibrierung mittels globalem Template Matching zusammen und definiert den nächsten Vertical Slice.

## 📋 Status der Implementierung

- [x] **Referenzkarte generieren:** `OpenCVLocalizer.create_arena_reference_image` ist bereits implementiert.
- [x] **LiDAR-Projektion:** `OpenCVLocalizer.project_lidar_to_template` ist bereits implementiert.
- [x] **Akkumulation im Driver:** `wro_driver.py` sammelt bereits 10 LiDAR-Scans und berechnet den Mittelwert.
- [x] **Anzeige-Integration:** Der Driver ruft die Kalibrierung auf, zeigt die Überlagerung an und blockiert das Losfahren bis zur Freigabe.
- [x] **Core-Template-Matching:** Hochperformante zweistufige Pyramiden-Suche über $0^\circ$ bis $359^\circ$ ist implementiert.
- [x] **Symmetrie-Auflösung:** Die Filterung nach $y_{\text{real}} < 1.1\,\text{m}$ zur Erkennung des Startkorridors ist aktiv.
- [x] **Richtungserkennung:** Zuordnung von `CW` und `CCW` basierend auf dem gefundenen Yaw-Winkel.

---

## 🎯 Nächster Vertical Slice: Core-Template-Matching & Symmetrie-Auflösung

Der nächste logische und funktionale Schnitt (Vertical Slice) besteht darin, die eigentliche Kalibrierungs-Engine zu implementieren und sie so zu integrieren, dass der Roboter beim Start seine echte Pose und Fahrtrichtung erkennt.

### Kernaufgaben für diesen Slice:
1. **Implementierung der Matching-Logik in `opencv_localizer.py`:**
   - Erstellen einer Methode `calibrate_initial_pose(self, avg_ranges, angle_offset, angle_inc)`:
     - Iteriert über Yaw-Winkel von $0^\circ$ bis $359^\circ$ in Schritten von $1^\circ$.
     - Projiziert die LiDAR-Daten für jeden Winkel.
     - Führt ein OpenCV Template-Matching (`cv2.matchTemplate` mit `cv2.TM_CCOEFF`) auf der `ref_img` durch.
     - Speichert den besten Score, die Pixelposition und den Winkel für die Top-Kandidaten.
2. **Symmetrie-Auflösung & Richtungsbestimmung:**
   - Filtert die Top-Kandidaten nach der Y-Koordinate: $y_{\text{real}} < 1.1\,\text{m}$.
   - Bestimmt die Fahrtrichtung:
     - Zeigt der gefundene Yaw-Winkel nach Osten ($\approx 90^\circ$ bzw. $\pi/2$), ist die Richtung `CCW`.
     - Zeigt er nach Westen ($\approx 270^\circ$ bzw. $-\pi/2$), ist die Richtung `CW`.
3. **Integration & Visualisierung:**
   - Einzeichnen der kalibrierten Pose und des LiDAR-Templates als Overlay auf der Referenzkarte zur visuellen Überprüfung.
   - Aktualisierung des Drivers, sodass er die echte Pose und Richtung übernimmt und anwendet.

---

## 📅 Detaillierte Task-Liste (TODO)

### 1. Modul: `opencv_localizer.py`
- [x] Methode `set_initial_pose(self, x, y, yaw)` implementieren, um Startwerte festzulegen.
- [x] Methode `calibrate_initial_pose(self, avg_ranges, angle_offset, angle_inc)` implementieren.
- [x] Methode `draw_calibration_result(self, ref_img, tpl_img, x, y, yaw)` zur Überlagerung für die Visualisierung implementieren.
- [x] Logik zur Bestimmung von `CW` / `CCW` aus dem Yaw-Winkel integrieren.

### 2. Integration in `wro_driver.py`
- [x] Die Dummy-Zuweisung in Stage 2 durch den Aufruf von `localizer.calibrate_initial_pose` ersetzen.
- [x] Initial-Pose an den Localizer übergeben (`localizer.set_initial_pose`).
- [x] Gefundene Fahrtrichtung (`driving_direction`) für Stage 3 (Planning) setzen.

### 3. Verifikation & Test
- [x] Roboter in verschiedenen Winkeln im Süd-Korridor in Webots platzieren.
- [x] Überprüfen, ob die berechnete Start-Pose und Fahrtrichtung exakt stimmen.
- [x] Prüfen, ob die Visualisierung im cv2-Fenster das LiDAR-Overlay perfekt deckungsgleich anzeigt.
