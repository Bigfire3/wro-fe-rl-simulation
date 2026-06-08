<p align="center">
  <img src="docs/media/Track_Model_5.gif" width="85%" alt="Webots 3D Simulation" />
</p>

# WRO Future Engineers - RL Autonomous Racing Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-green.svg)](https://gymnasium.farama.org/)
[![Webots](https://img.shields.io/badge/Webots-R2025b-red.svg)](https://cyberbotics.com/)
[![ONNX](https://img.shields.io/badge/ONNX-runtime-orange.svg)](https://onnxruntime.ai/)

An autonomous racing robot for the **WRO (World Robot Olympiad) Future Engineers** competition, trained using **Deep Reinforcement Learning (PPO)** in a custom **Webots simulation**. The control pipeline features real-time **ICP-based localization** and **dynamic obstacle mapping** with ray-cast decay.

<p align="center">
  <img src="docs/media/Loc_Model_5.gif" width="60%" alt="ICP Localization & Path Visualizer" />
</p>

---

## 💡 Motivation & Background

In the 11th grade, I participated in the **WRO Future Engineers** competition, winning **1st place in the Regional and German National Finals**, and **13th in the World Finals** (see our old hardware-based robot: [wro-fe-MacRobot](https://github.com/Bigfire3/wro-fe-MacRobot)). At the time, we used a classical, model-based reflex agent with limited decision-making intelligence.

Now, as a **6th-semester Robotics student**, I wanted to revisit this challenge using **Deep Reinforcement Learning (RL)** and high-fidelity simulator testing. This repository represents a complete redesign of the project from scratch.

---

## 🛠️ Software Architecture

The control loop is executed sequentially in a **4-Stage Software Pipeline** at every simulation step:

```mermaid
graph LR
    Stage1[STAGE 1: PERCEPTION] --> Stage2[STAGE 2: ESTIMATION]
    Stage2 --> Stage3[STAGE 3: PLANNING]
    Stage3 --> Stage4[STAGE 4: CONTROL]
    Stage4 -.->|Feedback Loop| Stage1
```

Detailed documentation of coordinate transforms, localization mathematics, and mapping algorithms can be found in [docs/README.md](file:///c:/Users/fabia/Documents/WRO_FE_SIM/docs/README.md).

---

## ✨ Key Technical Highlights

*   **Curriculum Reinforcement Learning:** Trained using **PPO (Stable-Baselines3)** on a custom **Gymnasium** environment (see the detailed [wro_driver README](file:///c:/Users/fabia/Documents/WRO_FE_SIM/controllers/wro_driver/README.md) for specs on the 15D observation vector and reward shaping). The agent transitions from **Stage 1 (Safety focus)**, which teaches lane-keeping and basic driving at a constant speed, to **Stage 2 (Performance focus)**, optimizing lap times with variable speed control and curve-cutting.
*   **Translation-Only ICP:** Scan-to-map matching via a fast 3-iteration Iterative Closest Point algorithm to maintain an accurate estimate of the robot's local pose `(x, y, yaw)`.
*   **Dynamic Obstacle Mapping:** Clusters LiDAR outliers (representing the obstacle boxes) and filters them dynamically. Implements a ray-casting visibility decay method to fade out obstacles once they are no longer in the vehicle's line-of-sight.
*   **Template Matching for Initial Localization:** Calibrates the robot's initial pose `(x, y, yaw)` and driving direction (`CW`/`CCW`) using template matching on the start corridor, providing a robust initialization for the continuous ICP tracking loop.

---

## 📂 Project Directory Structure

*   [controllers/wro_driver/](file:///c:/Users/fabia/Documents/WRO_FE_SIM/controllers/wro_driver/): Core autonomy software.
    *   `wro_gym_env.py`: Gymnasium reinforcement learning environment wrapper.
    *   `train.py`: Training script for PPO with curriculum stage support.
    *   `export_onnx.py`: Utility script to export trained Stable-Baselines3 policies to ONNX.
    *   `wro_driver.py`: Webots supervisor node executing the ONNX model inference.
    *   `wro_core/`: Auxiliary estimation, perception, planning, and control modules.
*   [docs/](file:///c:/Users/fabia/Documents/WRO_FE_SIM/docs/): Technical documentation and visual assets.
*   [models/](file:///c:/Users/fabia/Documents/WRO_FE_SIM/models/): Exported ONNX model configurations (`wro_model.onnx`).
*   [worlds/](file:///c:/Users/fabia/Documents/WRO_FE_SIM/worlds/): Simulation arenas (`track.wbt` for testing, `track_training.wbt` for training).

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have [Webots](https://cyberbotics.com/) installed (tested with Webots R2025b).

### 2. Setup Virtual Environment
```bash
git clone https://github.com/Bigfire3/wro-fe-rl-simulation.git
cd wro-fe-rl-simulation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Running & Training

#### a) Inference (Using the Pre-trained Model)
The repository already includes a pre-trained model (`models/wro_model.onnx`), allowing you to test the autonomous driving immediately:
1. Open Webots and load the evaluation world `worlds/track.wbt`.
2. Press the **Play** button in Webots. The simulator will automatically launch the internal `wro_driver` controller, which loads the pre-trained ONNX model and drives autonomously.

#### b) Training (Train Your Own Model)
If you want to train your own reinforcement learning model from scratch:
1. Open Webots and load the training world `worlds/track_training.wbt`.
2. Ensure the robot's controller in the Scene Tree is set to `<extern>` (to connect to the external Gymnasium environment).
3. Run the training script in your terminal:
   ```bash
   python controllers/wro_driver/train.py --timesteps 150000 --no-render
   ```
4. Start TensorBoard to monitor training progress:
   ```bash
   tensorboard --logdir=tb_logs
   ```