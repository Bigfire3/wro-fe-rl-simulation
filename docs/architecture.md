# WRO Future Engineers 2026 - Eingebettete Softwarearchitektur

## 1. Die 4-Stufen-Software-Pipeline

Um eine saubere Code-Trennung zu gewährleisten und Best Practices der Embedded-Software-Entwicklung einzuhalten, MUSS die Regelschleife des Roboters in jedem Frame sequenziell über 4 klar abgegrenzte Stufen ausgeführt werden. 

Die Planungslogik ist dabei vollständig in **STAGE 3: PLANNING** gekapselt. Die Stufen 1, 2 und 4 enthalten keine Planungsentscheidungen (wobei Stage 2 beim Start die einmalige Kalibrierung zur Pose- und Richtungsschätzung übernimmt).

* **[ STAGE 1: PERCEPTION ]** (Wahrnehmung)
  * *Verantwortung:* Einlesen der Rohdaten von den Sensoren (LiDAR-Distanzen, IMU-Werte, Kamerabild) und Anwendung grundlegender Filter (z. B. zur Entfernung von Ausreißern).
  * *Einschränkung:* Diese Stufe enthält **keine** Zustandsprüfungen, Wartezeiten oder Kontrollflusssteuerungen (keine State-Machine-Logik).
  
* **[ STAGE 2: ESTIMATION ]** (Zustandsschätzung)
  * *Verantwortung:* Berechnung der Roboterpose `(x, y, yaw)` durch geometrischen Abgleich (Template-Matching zur Kalibrierung, Translation-Only ICP zum kontinuierlichen Tracking). Klassifiziert LiDAR-Ausreißer und führt ein dynamisches Hindernis-Mapping (Obstacle Mapping) durch.
  
* **[ STAGE 3: PLANNING ]** (Pfadplanung)
  * *Verantwortung:* Generierung der Soll-Trajektorie (Geschwindigkeits- und Lenkwinkelvorgaben) aus der Reinforcement-Learning-Inferenz.
  
* **[ STAGE 4: CONTROL ]** (Regelung)
  * *Verantwortung:* Berechnung der konkreten Stellgrößen für die Aktuatoren (Ackermann-Lenkgeometrie für die Servomotoren, Soll-Geschwindigkeiten für die Antriebsmotoren) basierend auf den Vorgaben aus Stage 3.
  * *Features:*
    * **Elektronisches Differenzial**: Berechnet unterschiedliche Sollgeschwindigkeiten für das linke und rechte Hinterrad basierend auf dem aktuellen Lenkwinkel (kurveninneres Rad fährt langsamer, kurvenäußeres Rad schneller).
    * **Direkte Ansteuerung**: Verzicht auf Tiefpassfilter für maximale Reaktionsfähigkeit.
  * *Einschränkung:* Diese Stufe arbeitet rein zustandsunabhängig und führt lediglich mathematische Regelungsberechnungen und Sicherheitsbegrenzungen (Clamping) durch.

---

## 2. Spezifikationen des Koordinatensystems und Visualisierungsraums

Die Codebasis verwendet ein strikt positives globales Koordinatensystem für die Positionierung und Ausrichtung in der Arena.

* **Realer kontinuierlicher Raum ($P_r$):**
  * Der Ursprung $(0.0, 0.0)$ befindet sich exakt in der **südwestlichen Innenecke** der Außenbegrenzungen.
  * X-Achse: $0.0\,\text{m}$ bis $3.0\,\text{m}$ (Westen nach Osten / Easting).
  * Y-Achse: $0.0\,\text{m}$ bis $3.0\,\text{m}$ (Süden nach Norden / Northing).
  * Orientierung (Yaw): $0.0\,\text{rad}$ zeigt nach Norden (+Y). Positive Winkel drehen im Gegenuhrzeigersinn (CCW).
* **OpenCV-Visualisierungsraum ($P_{cv}$):**
  * Ein Bildfenster der Größe $600 \times 600$ Pixel, welches die LiDAR-Messungen aus der Roboter-Perspektive darstellt.
  * Der Roboter befindet sich im Zentrum des Bildes $(cx, cy) = (300, 300)$.
  * Die Skalierung beträgt $150\,\text{Pixel/Meter}$ (`scale = 150.0`).
  * Transformation für LiDAR-Punkte:
    * `px = cx - y_local * scale`
    * `py = cy - x_local * scale`
    (Da die lokale X-Achse des Roboters nach vorne zeigt und die lokale Y-Achse nach links).
* **Template-Matching / Kalibrierungs-Raum ($P_{tpl}$):**
  * Die Referenzkarte der Arena wird auf ein Bild gezeichnet, dessen Größe durch den Suchbereich mit zusätzlichem Padding bestimmt wird: `img_size = int((3.0 + 2.0 * padding) * scale)` Pixel (mit `scale = 150.0` und `padding = 2.0` entspricht dies $1050 \times 1050$ Pixeln).
  * Transformation einer globalen Position $(x, y)$ in Pixel auf dieser Karte:
    * `px = int((x + padding) * scale)`
    * `py = int(img_size - (y + padding) * scale)` (Y-Achse invertiert).

---

## 3. Modulspezifikation: Schätzung & Mapping (Stage 2)

### 3.1. `OpenCVLocalizer` (`opencv_localizer.py`)
* **`calibrate_initial_pose(avg_ranges, angle_offset, angle_inc)`**: Bestimmt die Anfangspose im Startkorridor mittels Template Matching und leitet die Fahrtrichtung (`CW`/`CCW`) ab.

### 3.2. `TranslationICPLocalizer` (`trans_icp_localizer.py`)
* **`update(lidar_ranges, imu_yaw, max_range=2.0, ego_v_x=0.0)`**: Berechnet die Pose `(x, y, yaw)` über 3 Iterationen Translation-Only ICP und gibt LiDAR-Ausreißer (Wandabstand $\ge 15\,\text{cm}$) zurück. Zeichnet den Trajektorienverlauf auf.
* **`render()`**: Erzeugt das Debug-Bild mit LiDAR-Punkten (Inlier=grün, Outlier=rot) und Roboterpose. Der Trajektorienverlauf wird **geschwindigkeitsabhängig eingefärbt** (von Cyan/Blau für langsam über Grün zu Orange/Rot für Höchstgeschwindigkeiten bis 1.6 m/s) inklusive einer eingeblendeten Farblegende.

### 3.3. `ObstacleMapper` (`obstacle_mapper.py`)
* **`update(robot_pose, outlier_points)`**: Clustert Outliers (Schwelle $10\,\text{cm}$, Rauschfilter $\ge 2$ Punkte), gleicht sie mit Hindernissen ($50\,\text{mm}$ Boxen) ab und passt die Position (mittels Tiefpassfilter $\alpha = 0.1$) sowie die Confidence (+0.01) an. Verringert die Confidence (-0.01) bei Nicht-Erkennung nur im freien Sichtfeld (Radius $< 2.0\,\text{m}$, Sichtachse nicht durch Wände oder andere Hindernisse verdeckt).
* **`render(img, robot_pose, scale, window_size)`**: Zeichnet Hindernisse (Farbe entspricht der klassifizierten Farbe rot/grün oder Grauton basierend auf der Confidence) und Sichtlinien (grün) zum Roboter.

---

## 4. Modulspezifikation: Planung (Stage 3)

Die Verhaltenssteuerung des Roboters wird über Pfadplaner realisiert, die in **STAGE 3: PLANNING** ausgeführt werden. Der Controller nutzt:

* **Reinforcement Learning (RL)**: Nutzt den `RLPlanner` zur Auswertung eines ONNX-Modells (`wro_model.onnx` via `onnxruntime`). Verwendet eine **inkrementelle Aktionssteuerung** (Steering und Speed werden über diskrete Inkremente angepasst und auf Grenzwerte limitiert).

---

## 5. Flussdiagramm: Gesamtprogrammablauf (Control Loop)

Dieses Diagramm zeigt den sequenziellen Durchlauf der 4 Stufen der Software-Pipeline in jedem Simulations-Zeitschritt:

```mermaid
graph TD
    Start([Start des Steuerungsprogramms]) --> Step[Frame-Schritt: robot.step]
    Step --> Stage1[STAGE 1: PERCEPTION<br/>- Rohdaten einlesen<br/>- Ausreißer filtern]
    Stage1 --> Stage2[STAGE 2: ESTIMATION<br/>- Pose schätzen / kalibrieren]
    Stage2 --> Stage3[STAGE 3: PLANNING<br/>- Trajektorie berechnen]
    Stage3 --> Stage4[STAGE 4: CONTROL<br/>- Ackermann-Winkel berechnen<br/>- Motordrehzahl regeln]
    Stage4 --> Check{Simulation aktiv?}
    Check -- Ja --> Step
    Check -- Nein --> End([Programm-Ende])
```
