Implementation Plan - Umstellung RL-Architektur (High-Speed & Generalisierung)
Dieses Dokument beschreibt die geplante Anpassung des Reinheitsgrad-Lernens (RL) für das WRO Future Engineers Roboterprojekt, basierend auf den Anforderungen in 
TODO.md
.

User Review Required
IMPORTANT

Driver als Supervisor: Der wro_driver.py wird von Robot auf Supervisor umgestellt, um über getVelocity() die Eigengeschwindigkeit ego_v_x des Fahrzeugs exakt zu bestimmen.
Bestehende Belohnungen entfernt: Gemäß Absprache werden alle alten Checkpoint-Belohnungen sowie der Zeitschritt-Malus (-0.01) entfernt. Der Reward setzt sich ausschließlich aus ego_v_x * 1.0 - abs(lateral_error) * 0.5 zusammen.
Timeout (Stillstand): Der Stagnations-Check wird rein auf die Eigengeschwindigkeit umgestellt (ego_v_x < 0.01 m/s für 50 konsekutive Schritte).
Proposed Changes
Gymnasium Environment
[MODIFY] 
wro_gym_env.py
Action Space:
Konvertieren in spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32).
In step(): Dimension 0 steuert die Lenkung (skaliert mit self.MAX_STEERING = 0.8), Dimension 1 steuert die Geschwindigkeit (skaliert mit self.max_motor_velocity).
Observation Space:
Festlegen auf spaces.Box mit 12 Elementen (Low/High Grenzen entsprechend auf [-1.0, 1.0] oder [0.0, 1.0] angepasst).
Berechnung der Eigengeschwindigkeit ego_v_x via self.robot_node.getVelocity() und Transformation in lokale Koordinaten.
Definition der quadratischen Mittellinie als 2.0m x 2.0m Quadrat um $(1.5, 1.5)$.
Berechnung von lateral_error (Projektionsabstand zur Mittellinie, normiert mit /0.5 und Capping auf 1.0).
Berechnung von heading_error (Differenz zwischen Fahrzeug-Yaw und Linientangente, normiert mit /(pi/2)).
Berechnung der Lookahead-Punkte in 30 cm und 60 cm entlang der Mittellinie. Transformation in lokale Koordinaten und Normierung der Y-Koordinate durch Division mit 0.3 bzw. 0.6 und anschließendem Capping auf [-1.0, 1.0].
Sortierung der Hindernisse nach kleinstem positiven rel_x (in Fahrzeugrichtung voraus). Normierung von rel_x und rel_y durch Division mit 2.0. Dummy-Werte falls < 2 Hindernisse vorhanden: rel_x = 2.0, rel_y = 0.0, color = 0.0.
Belohnungssystem & Resets:
reward = ego_v_x * 1.0 - abs(lateral_error) * 0.5.
Harter Bestrafungs-Reward von -20.0 und sofortiger Abbruch (terminated = True) bei Kollision oder Falschumfahrung einer Säule.
Abbruch nach einer vollständigen Runde (terminated = True).
Abbruch bei Stillstand (truncated = True), wenn ego_v_x < 0.01 m/s für 50 Schritte.
Robot Controller & Inference
[MODIFY] 
wro_driver.py
Klasse Robot zu Supervisor ändern.
Ermittlung des robot_node über robot.getSelf().
Berechnung des 12-elementigen Observation-Vektors analog zum Gym-Environment (inklusive Eigengeschwindigkeit, Mittellinien-Features und Hindernis-Projektionen).
Ausführung der ONNX-Inferenz mit dem neuen Eingangsvektor.
Anwendung der 2-dimensionalen Ausgabe auf Lenk-Servos und Motoren (analog zur Skalierung im Environment).
Training Script & Hyperparameters
[MODIFY] 
train.py
Hyperparameter des PPO-Algorithmus anpassen:
learning_rate = 2e-4
n_steps = 8192
batch_size = 256
gamma = 0.98
ent_coef = 0.01
policy_kwargs = dict(net_arch=[64, 64])
Model Export Script
[MODIFY] 
export_onnx.py
Anpassung des Dummy-Inputs im Export-Skript von 8 auf 12 Elemente (dummy_input = torch.randn(1, 12, dtype=torch.float32)).
Verification Plan
Automated Tests
Testen des modifizierten Gym-Environments durch Ausführen von:
powershell

.venv\Scripts\python controllers/wro_driver/test_env.py
Dabei wird überprüft, ob die Observations korrekt berechnet werden, die Simulation läuft und keine Fehler auftreten.
Manual Verification
Trainieren des Modells für 50.000 Schritte, Exportieren ins ONNX-Format und Testen im Driver, um das korrekte Inferenzverhalten zu verifizieren:
powershell

.venv\Scripts\python controllers/wro_driver/train.py --timesteps 50000 --no-render
.venv\Scripts\python controllers/wro_driver/export_onnx.py