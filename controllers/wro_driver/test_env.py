import time
import numpy as np
import sys
import os

# Ensure the script directory is in Python path to import wro_gym_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wro_gym_env import WebotsWroEnv

def main():
    print("=" * 60)
    print("WRO FUTURE ENGINEERS - RL ENVIRONMENT TEST SCRIPT")
    print("=" * 60)
    print("Voraussetzungen:")
    print("1. Webots R2025a (oder kompatibel) ist geoeffnet.")
    print("2. Die Welt 'track_training.wbt' ist geladen.")
    print("3. Der Controller des Roboters steht auf '<extern>' (im Scene Tree).")
    print("4. Die Simulation steht auf PAUSE oder läuft langsam.")
    print("-" * 60)
    print("Starte Verbindung zum Webots-Simulator...")
    
    try:
        # Gymnasium Environment instanziieren
        env = WebotsWroEnv()
        print("[SUCCESS] Verbindung zu Webots erfolgreich hergestellt!")
        
        for episode in range(3):
            print(f"\n--- Starte Test-Episode {episode + 1} ---")
            obs, info = env.reset()
            print(f"Reset abgeschlossen. Erste Observation:\n{obs}")
            
            step = 0
            done = False
            total_reward = 0.0
            
            while not done:
                # Waehle eine zufaellige Aktion (Lenkwinkel-Offset)
                action = env.action_space.sample()
                
                # Schritt in der Simulation ausführen
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
                
                # Visualisierung rendern
                env.render()
                
                # Konsolen-Logs alle 10 Schritte
                if step % 10 == 0:
                    print(f"Schritt {step:03d} | Pose: x={info['x']:.2f}, y={info['y']:.2f}, yaw={info['yaw']:.2f} | "
                          f"Checkpoint: {info['checkpoint']} | Reward: {reward:+.2f} | Total Reward: {total_reward:+.2f}")
                    print(f"          Obs (Wand_L, Wand_R, Wand_F, Obst_X, Obst_Y, Color): "
                          f"{obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}, {obs[5]:.2f}, {obs[6]:.2f}, {obs[7]:.1f}")
                
                # Kurze Pause, um die Konsole lesbar zu halten
                time.sleep(0.02)
                
            print(f"\n--- Episode {episode + 1} beendet nach {step} Schritten ---")
            print(f"End-Pose: x={info['x']:.2f}, y={info['y']:.2f}")
            print(f"Erreichte Checkpoints: {info['checkpoint']}")
            print(f"Gesamt-Belohnung: {total_reward:+.2f}")
            print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n[Test Env] Test durch Benutzer abgebrochen.")
    except Exception as e:
        print(f"\n[ERROR] Fehler bei der Simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nSchließe Fenster...")
        try:
            env.close()
        except NameError:
            pass
        print("Test beendet.")

if __name__ == "__main__":
    main()
