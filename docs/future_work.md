# Future Work: Lokalisierung & Hindernis-Mapping

Dieses Dokument fasst die Ergebnisse unserer Brainstorming-Session für die Lokalisierungs- und Mapping-Architektur des WRO-Fahrzeugs auf einem Raspberry Pi 4 zusammen.

---

## 1. Lokalisierung (Estimation)

Da die Rotation über die IMU als absolut präzise und driftfrei angenommen wird, verbleibt nur die Schätzung der Translation ($X, Y$). Hierfür vergleichen wir zwei unterschiedliche Ansätze.

### Ansatz A: Translation-Only ICP
Iterativer Abgleich der rotierten LiDAR-Punkte mit den Liniensegmenten der statischen Karte.

*   **Ablauf (pro Zyklus):**
    1.  Rotieren der LiDAR-Punkte mittels IMU-Yaw und Projezieren in globale Koordinaten unter Verwendung der aktuellen Pose $(X_{est}, Y_{est})$.
    2.  Für jeden Punkt das nächstgelegene Liniensegment der statischen Karte (Außen- und Innenwände) bestimmen.
    3.  Berechnen des Fehlervektors (Abstand des Punktes zum Segment).
    4.  Berechnung des Korrekturvektors $(\Delta x, \Delta y)$ mittels Least-Squares oder Median der Abweichungen.
    5.  Verschieben der geschätzten Pose und Wiederholung (Schritte 2–4) bis zur Konvergenz (meist 1–3 Iterationen).
*   **Vorteile:**
    *   Sehr robust gegenüber größeren Schätzfehlern (z. B. nach Rucklern oder Kollisionen).
    *   Automatische Zuweisung auch bei schrägen oder komplexeren Wandverläufen.
*   **Nachteile:**
    *   Höhere CPU-Last durch iterative Suche nach dem nächsten Liniensegment für jeden Punkt.

### Ansatz B: One-Shot 1D-Wandprojektion
Direkte, nicht-iterative Zuweisung von Punkten zu den bekannten, achsenparallelen Wänden der Arena.

*   **Ablauf (pro Zyklus):**
    1.  Rotieren der LiDAR-Punkte mittels IMU-Yaw und Projezieren in globale Koordinaten.
    2.  Direkte Zuweisung der Punkte anhand ihrer globalen Koordinaten zu den Wänden (z. B. wenn $x_g \approx 3.0$ und $y_g \in [0, 3]$, gehört der Punkt zur rechten Außenwand).
    3.  Berechnung der Translation: $\Delta x$ als Median aller Abweichungen der vertikalen Wandpunkte; $\Delta y$ für die horizontalen Wandpunkte.
    4.  Einmalige (One-Shot) Korrektur der Pose $(X_{est}, Y_{est})$ um $(\Delta x, \Delta y)$.
*   **Vorteile:**
    *   Extrem performant ($O(n)$ ohne iterative Schleifen und komplexe Distanzprüfungen).
    *   Hervorragend geeignet für eingebettete Systeme wie den Raspberry Pi 4.
*   **Nachteile:**
    *   Weniger robust bei sehr großen Positionsfehlern, da Punkte fälschlicherweise falschen Wänden zugeordnet werden könnten.

---

## 2. Hindernis-Mapping (`List[Obstacle]`)

Anstelle eines rechenintensiven Gitters (Occupancy Grid) wird eine dynamische Liste von `Obstacle`-Objekten gepflegt.

### Struktur der Klasse `Obstacle`
*   `position`: Globale Position $(x, y)$.
*   `size`: Konstante Größe von $45\text{mm} \times 45\text{mm}$.
*   `confidence` (Sicherheit): Ein numerischer Wert, der bei Bestätigung steigt und ohne Bestätigung sinkt.
*   `color` (Farbe): Initialisiert als `grau` (unknown), Auswertung zu `rot`/`grün` erfolgt später.

### Ablauf des Mapping-Algorithmus
1.  **Ausreißer-Erkennung (Karte abziehen):**
    *   Nach der Lokalisierung werden alle LiDAR-Punkte, die nahe an den statischen Wänden liegen, herausgefiltert.
    *   Die verbleibenden Punkte im Freiraum sind Hindernis-Kandidaten.
2.  **Rauschfilter (Clustering):**
    *   Punkte werden geclustert (z. B. einfache Distanz-Nachbarschaft).
    *   Cluster der Größe 1 (einzelne isolierte Punkte) werden als Rauschen verworfen. Nur Cluster mit $\ge 2$ Punkten werden weiterverarbeitet.
3.  **Datenassoziation (Karten-Update):**
    *   Für jedes gefundene Cluster wird geprüft, ob sich bereits ein bekanntes `Obstacle` in der Nähe befindet.
    *   **Falls ja:** Die Confidence des Hindernisses wird erhöht. Die Position des Hindernisses wird so angepasst, dass das neue Cluster im $45\text{mm} \times 45\text{mm}$ Quadrat liegt. Liegt es bereits im Quadrat, bleibt die Position unverändert.
    *   **Falls nein:** Ein neues `Obstacle` wird mit niedriger Start-Confidence erzeugt.
4.  **Sichtfeld-basiertes Vergessen (Decay):**
    *   In jedem Zyklus verringert sich die Confidence aller bekannten Hindernisse leicht.
    *   **Wichtige Einschränkung:** Dieser Decay-Schritt wird *nur* auf Hindernisse angewendet, deren Abstand zum Roboter kleiner als der LiDAR-Radius ($2.0\text{m}$) ist oder deren Sichtachse verdeckt ist. Hindernisse außerhalb des Sichtfelds behalten ihre Confidence, damit sie beim Umrunden der Strecke nicht gelöscht werden.
    *   Fällt die Confidence eines Hindernisses unter einen Schwellenwert, wird es aus der Liste gelöscht.
