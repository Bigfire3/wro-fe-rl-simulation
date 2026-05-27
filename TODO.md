# TODO: Reinforcement Learning Integration (WRO FE Simulation)

Dieses Projekt enthält eine fertige **Gymnasium-Umgebung** zur Verbindung von Webots mit RL-Algorithmen (Stable-Baselines3). Die Verbindung und die physikalischen Resets wurden erfolgreich getestet.

Die folgenden Schritte beschreiben, wie ein nachfolgender Agent oder Entwickler das Modell trainieren, exportieren und in den Roboter integrieren kann.

---

## Aktueller Stand
- **Gymnasium Env:** [`controllers/wro_driver/wro_gym_env.py`](file:///c:/Users/fabia/Documents/WRO_FE_SIM/controllers/wro_driver/wro_gym_env.py) – Implementiert Observation Space (8-D), Action Space (1-D), Belohnungen (Checkpoints, richtiges Umfahren) und automatische Webots-Pfad-Erkennung unter Windows.
- **Verbindungstest:** [`controllers/wro_driver/test_env.py`](file:///c:/Users/fabia/Documents/WRO_FE_SIM/controllers/wro_driver/test_env.py) – Testet die Verbindung und führt den Roboter mit zufälligen Aktionen aus.

---

## Wichtiger Hinweis: Normalisierung der Observations (Eingangsdaten)
Um das Lernen zu beschleunigen und die Stabilität des neuronalen Netzes zu garantieren, sollten alle Werte im Observation-Vektor vor dem Training normalisiert werden (Skalierung auf den Bereich `[0.0, 1.0]` oder `[-1.0, 1.0]`).

Der aktuelle 8-dimensionale Observation-Vektor in `wro_gym_env.py` ist in Metern ausgedrückt. Für das Training empfiehlt sich folgende Normalisierung in `_get_obs()`:
1. **Lidar-Abstände (Index 0 bis 4):** Teilen durch `1.5` (die maximale Reichweite/Cap), um Werte im Bereich `[0.0, 1.0]` zu erhalten.
2. **Nächstes Hindernis X/Y (Index 5 und 6):** Teilen durch `2.0` (die maximale Sichtweite/Cap), um Werte im Bereich `[-1.0, 1.0]` zu erhalten.
3. **Hindernisfarbe (Index 7):** Liegt bereits perfekt im Bereich `[-1.0, 1.0]` (-1.0 für Grün/links, 1.0 für Rot/rechts, 0.0 für kein Hindernis/grau).

*Tipp für den nächsten Agenten:* Passe entweder die Methode `_get_obs()` in `wro_gym_env.py` an, um diese Skalierung direkt durchzuführen (vergiss nicht, die `low`/`high` Grenzen im `observation_space` in `__init__` auf `0.0`/`1.0` bzw. `-1.0`/`1.0` anzupassen!), oder verwende Stable-Baselines3-Wrapper bzw. Gymnasium-Wrapper wie `gym.wrappers.NormalizeObservation` im Trainings-Skript.

---

## Nächste Schritte

### 1. Trainings-Skript erstellen (`train.py`)
Erstelle ein Skript (z. B. `controllers/wro_driver/train.py`), das den Lernprozess steuert.
```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from wro_gym_env import WebotsWroEnv

def main():
    # Umgebung instanziieren
    env = WebotsWroEnv()
    
    # PPO-Modell erstellen (CPU-Training reicht voellig aus!)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./tb_logs/"
    )
    
    print("Starte Training...")
    # Ca. 100.000 bis 150.000 Timesteps sollten für den statischen Track reichen
    model.learn(total_timesteps=150000, tb_log_name="ppo_wro")
    
    # Modell speichern
    model.save("wro_ppo_model")
    print("Modell erfolgreich unter 'wro_ppo_model.zip' gespeichert!")
    
    env.close()

if __name__ == "__main__":
    main()
```

### 2. Trainingsfortschritt überwachen (TensorBoard)
Öffne ein Terminal im Projektverzeichnis und starte Tensorboard:
```bash
.venv\Scripts\tensorboard --logdir ./tb_logs/
```
Achte auf die Kurven von `rollout/ep_rew_mean` (durchschnittliche Belohnung pro Versuch) und `rollout/ep_len_mean` (Überlebenszeit). Die Belohnung sollte stetig ansteigen.

### 3. Modell nach ONNX exportieren (`export_onnx.py`)
Erstelle ein Skript, das die gelernte Policy extrahiert und in das ONNX-Format übersetzt, damit die Ausführung auf der Embedded-Hardware ohne PyTorch möglich ist.
```python
import torch
from stable_baselines3 import PPO
import numpy as np

# PPO-Modell laden
model = PPO.load("wro_ppo_model")

# Stable-Baselines3 Policy ist ein PyTorch-Modell
policy = model.policy

# Dummy-Input erzeugen (Entspricht unserem 8-dimensionalen Observation Space)
dummy_input = torch.randn(1, 8)

# ONNX-Export ausführen
torch.onnx.export(
    policy,
    dummy_input,
    "wro_model.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("ONNX-Modell erfolgreich als 'wro_model.onnx' exportiert!")
```

### 4. Integration in den echten Roboter-Controller (`wro_driver.py`)
Ersetze die Planungs-Phase (Stage 3) im Controller durch die ONNX-Inferenz:
1. **Laden des Modells beim Start:**
   ```python
   import onnxruntime as ort
   import numpy as np
   
   ort_session = ort.InferenceSession("wro_model.onnx")
   ```
2. **Inferenz in der Planning-Schleife (10 Hz):**
   * Baue den gleichen 8-dimensionalen Observation-Vektor wie in `_get_obs()` aus `wro_gym_env.py` auf.
   * Führe das Modell aus:
     ```python
     obs_input = np.array([obs], dtype=np.float32)
     # Inferenz ausführen
     ort_inputs = {ort_session.get_inputs()[0].name: obs_input}
     ort_outs = ort_session.run(None, ort_inputs)
     
     # ONNX Stable-Baselines3 Policy gibt 3 Werte zurück (Action, Value, Log_Prob).
     # Wir benötigen nur den ersten Wert (die Aktion).
     action = ort_outs[0][0][0] 
     
     # Skaliere auf maximalen Lenkwinkel
     target_steering = action * MAX_STEERING
     target_speed = TARGET_SPEED
     ```
