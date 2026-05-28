import sys
import os
# Reconfigure stdout/stderr to utf-8 on Windows to avoid UnicodeEncodeError when printing emojis/Unicode
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
from stable_baselines3 import PPO

def main():
    print("=" * 60)
    print("WRO FUTURE ENGINEERS - ONNX MODEL EXPORT SCRIPT")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Path of the trained SB3 model zip
    model_path = os.path.join(models_dir, "wro_ppo_model.zip")
    # Path where to save the exported ONNX model
    onnx_path = os.path.join(models_dir, "wro_model.onnx")
    
    if not os.path.exists(model_path):
        # Check old root path as fallback
        fallback_path = os.path.join(project_root, "wro_ppo_model.zip")
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            print(f"[ERROR] Could not find trained model zip at {model_path} or {fallback_path}.")
            print("Bitte trainiere zuerst ein Modell mit 'python train.py'.")
            sys.exit(1)
            
    print(f"Lade PPO-Modell von: {model_path}")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"[ERROR] Fehler beim Laden des Modells: {e}")
        sys.exit(1)
        
    print("Extrahiere Policy-Netzwerk und erstelle exportierbares Modell...")
    policy = model.policy
    policy.to("cpu")
    policy.eval()
    
    # Wrap the policy in a custom Module that only extracts the deterministic action (mean_actions)
    # to avoid tracing the Normal distribution class which fails in newer PyTorch versions.
    class OnnxablePolicy(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, obs):
            latent_pi, _ = self.mlp_extractor(obs)
            actions = self.action_net(latent_pi)
            return actions

    onnxable_model = OnnxablePolicy(policy)
    onnxable_model.eval()
    
    # Dummy-Input representing 8-dimensional observation vector (batch size 1)
    dummy_input = torch.randn(1, 8, dtype=torch.float32)
    
    print(f"Exportiere ONNX-Modell nach: {onnx_path}")
    try:
        torch.onnx.export(
            onnxable_model,
            dummy_input,
            onnx_path,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print("[SUCCESS] ONNX-Modell erfolgreich exportiert!")
    except Exception as e:
        print(f"[ERROR] Fehler beim ONNX-Export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
