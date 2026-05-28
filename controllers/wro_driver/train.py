import sys
import os
import cv2
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Ensure the script directory is in Python path to import wro_gym_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wro_gym_env import WebotsWroEnv

class TrainingAbortException(Exception):
    """Custom exception raised to abort training without saving."""
    pass

class RenderCallback(BaseCallback):
    def __init__(self, render_training=True, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_training = render_training

    def _on_step(self) -> bool:
        if self.render_training:
            # Render the environment
            try:
                self.training_env.env_method("render")
            except Exception as e:
                print(f"[RenderCallback] Error calling render: {e}")
            
            # Check for user keyboard inputs ('q' or 'Esc') in the OpenCV window.
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q'), ord('Q')]:
                print("\n[RenderCallback] Abort requested by user (Keypress 'q' or 'Esc' in OpenCV).")
                raise TrainingAbortException("User aborted training via OpenCV window.")
        return True

def main():
    parser = argparse.ArgumentParser(description="WRO Reinforcement Learning Training")
    parser.add_argument("--timesteps", type=int, default=150000, help="Total training timesteps")
    parser.add_argument("--no-render", action="store_true", help="Disable OpenCV rendering during training")
    args = parser.parse_args()

    render_training = not args.no_render
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "wro_ppo_model")
    
    print("=" * 60)
    print("WRO FUTURE ENGINEERS - RL TRAINING SCRIPT")
    print("=" * 60)
    print(f"Rendering: {render_training}")
    print(f"Total Timesteps: {args.timesteps}")
    print("-" * 60)
    
    # Instantiate environment
    env = WebotsWroEnv()
    
    # Check if tensorboard is installed
    tb_log = "./tb_logs/"
    try:
        import tensorboard
    except ImportError:
        print("[Warning] TensorBoard ist nicht installiert. Training wird ohne TensorBoard-Logging fortgesetzt.")
        tb_log = None
        
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tb_log
    )
    
    # Initialize callback
    callback = RenderCallback(render_training=render_training)
    
    try:
        print("Starte Training...")
        model.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name="ppo_wro" if tb_log is not None else None)
        
        # Save model if completed normally
        model.save(model_save_path)
        print(f"\nModell erfolgreich unter '{model_save_path}.zip' gespeichert!")
    except KeyboardInterrupt:
        print("\n[Training] Training wurde vom Benutzer abgebrochen (Ctrl+C). Modell wurde NICHT gespeichert.")
    except TrainingAbortException as e:
        print(f"\n[Training] {e}")
    except Exception as e:
        print(f"\n[Training] Unerwarteter Fehler beim Training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Schließe Umgebung...")
        try:
            env.close()
        except Exception:
            pass
        print("Training-Skript beendet.")

if __name__ == "__main__":
    main()
