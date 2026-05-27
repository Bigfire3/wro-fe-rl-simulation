import sys
import os
import torch
from stable_baselines3 import PPO

def main():
    print("=" * 60)
    print("WRO FUTURE ENGINEERS - ONNX MODEL EXPORT SCRIPT")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path of the trained SB3 model zip
    model_path = os.path.join(script_dir, "wro_ppo_model.zip")
    # Path where to save the exported ONNX model
    onnx_path = os.path.join(script_dir, "wro_model.onnx")
    
    if not os.path.exists(model_path):
        # Check parent or current directory too
        model_path_cwd = "wro_ppo_model.zip"
        if os.path.exists(model_path_cwd):
            model_path = model_path_cwd
        else:
            print(f"[ERROR] Could not find trained model zip at {model_path} or current directory.")
            print("Bitte trainiere zuerst ein Modell mit 'python train.py'.")
            sys.exit(1)
            
    print(f"Lade PPO-Modell von: {model_path}")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"[ERROR] Fehler beim Laden des Modells: {e}")
        sys.exit(1)
        
    print("Extrahiere Policy-Netzwerk...")
    policy = model.policy
    
    # Send policy to CPU and put in evaluation mode
    policy.to("cpu")
    policy.eval()
    
    # Dummy-Input representing 8-dimensional observation vector (batch size 1)
    dummy_input = torch.randn(1, 8, dtype=torch.float32)
    
    print(f"Exportiere ONNX-Modell nach: {onnx_path}")
    try:
        # Wrap policy to export just the action selection
        # stable-baselines3 policy has an 'actor' or 'mlp_extractor' + 'action_net'.
        # However, policy itself can be exported. In SB3 PPO:
        # policy.forward(obs) returns (actions, values, log_prob).
        # We only need the actions. But wait, exporting the policy network directly works:
        # PPO's policy forward method is defined as:
        # def forward(self, obs: torch.Tensor, deterministic: bool = False):
        #     return self._predict(obs, deterministic=deterministic)
        # Wait, if we use policy, torch.onnx.export calls forward().
        # Let's verify what stable-baselines3 policy forward returns.
        # Yes, it returns action, value, log_prob, or action depending on methods.
        # But wait! If we export policy directly, torch.onnx.export exports the forward() method,
        # which in PPO returns (actions, values, log_prob).
        # When we load in onnxruntime, the output will contain 3 output arrays.
        # The first array will be the action.
        # Let's ensure the export is fully functional.
        
        # SB3 policy's forward() can be exported directly:
        torch.onnx.export(
            policy,
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
