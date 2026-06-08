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
    def __init__(self, render_training=True, forced_stage=None, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_training = render_training
        self.forced_stage = forced_stage

    def _on_step(self) -> bool:
        # Check global step count to update curriculum stage (Transition at 200,000 steps)
        if self.forced_stage is not None:
            stage = self.forced_stage
        else:
            stage = 1
            if self.num_timesteps >= 200000:
                stage = 2
            
        try:
            self.training_env.env_method("set_curriculum_stage", stage)
        except Exception:
            pass
            
        self.logger.record("train/curriculum_stage", stage)

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
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--no-render", action="store_true", help="Disable OpenCV rendering during training")
    parser.add_argument("--continue-training", "-c", action="store_true", help="Continue training the existing model if it exists")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None, help="Force curriculum stage (1 or 2). If not specified, standard automatic curriculum is used.")
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
    print(f"Continue Training: {args.continue_training}")
    print(f"Forced Curriculum Stage: {args.stage if args.stage is not None else 'Auto'}")
    print("-" * 60)
    
    # Instantiate environment
    env = WebotsWroEnv()
    if args.stage is not None:
        env.set_curriculum_stage(args.stage)
    
    # Check if tensorboard is installed
    tb_log = "./tb_logs/"
    try:
        import tensorboard
    except ImportError:
        print("[Warning] TensorBoard ist nicht installiert. Training wird ohne TensorBoard-Logging fortgesetzt.")
        tb_log = None
        
    # Check if we should load the existing model
    model_zip_path = model_save_path + ".zip"
    if args.continue_training and os.path.exists(model_zip_path):
        print(f"Lade bereits vorhandenes Modell für das Weitertraining: {model_zip_path}")
        # Overwrite hyperparameters to force exploration under new reward settings
        model = PPO.load(
            model_save_path,
            env=env,
            tensorboard_log=tb_log,
            ent_coef=0.01,       # Lowered to prevent standard deviation explosion (was 0.04)
            learning_rate=3e-4
        )
        # Reset exploration noise with standard deviation of 0.3 to force exploration of faster steering without destabilizing the policy
        import torch
        with torch.no_grad():
            model.policy.log_std.fill_(-1.2)  # Reset std to exp(-1.2) ≈ 0.3 (was -0.693 ≈ 0.5)
        print("Explorations-Rauschen (log_std) erfolgreich auf -1.2 (std ≈ 0.3) zurückgesetzt!")
    else:
        if args.continue_training:
            print(f"[Warning] Kein Modell unter '{model_zip_path}' gefunden. Starte neues Training von vorne.")
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=4096,             # Doubled for more stable gradient estimates
            batch_size=256,           # Larger batches for better statistics
            n_epochs=15,              # More passes over each batch
            gamma=0.995,              # Long-horizon planning for strategic corner-cutting
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,           # Lowered to prevent standard deviation explosion (was 0.02)
            vf_coef=1.0,              # Stronger value loss weight (fixes explained_var=0.29)
            verbose=1,
            tensorboard_log=tb_log,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[256, 128])  # Separate, larger value network
            )
        )
    
    # Initialize callback
    callback = RenderCallback(render_training=render_training, forced_stage=args.stage)
    
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
