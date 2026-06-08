# WRO Future Engineers - RL Autonomous Racing Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-green.svg)](https://gymnasium.farama.org/)
[![Webots](https://img.shields.io/badge/Webots-R2025b-red.svg)](https://cyberbotics.com/)
[![ONNX](https://img.shields.io/badge/ONNX-runtime-orange.svg)](https://onnxruntime.ai/)

An autonomous racing robot for the **WRO (World Robot Olympiad) Future Engineers** competition, trained using **Deep Reinforcement Learning (PPO)** in a custom **Webots simulation**. The control pipeline features real-time **ICP-based localization**, **dynamic obstacle mapping** with ray-cast decay, and an **electronic differential**.

---

## 📽️ Project Showcases

| 🏁 Webots 3D Simulation | 🗺️ OpenCV Path & Localization Visualizer | 👁️ Obstacle & Observation Debugger |
|:---:|:---:|:---:|
| ![Webots Track](docs/media/Track_Model_5.gif) | ![ICP Localization](docs/media/Loc_Model_5.gif) | ![Obstacle Mapper](docs/media/Obs_Model_5.gif) |

---

## 💡 Motivation & Background

Back in the 11th grade, I participated in the **WRO Future Engineers** competition, where my team won **1st place in the Regional and German National Finals**, and ranked **13th in the World Finals** (see our old hardware-based robot repository: [wro-fe-MacRobot](https://github.com/Bigfire3/wro-fe-MacRobot)). At the time, we solved the challenge using a classical, model-based reflex agent with limited decision-making intelligence.

Now, as a **6th-semester Robotics student**, I wanted to revisit this fascinating challenge through the lens of modern robotics and machine learning. This repository represents a complete redesign of the project from scratch, leveraging **Deep Reinforcement Learning (RL)** and high-fidelity simulator testing to achieve robust autonomous driving.

---

## 🛠️ Software Architecture

To match physical embedded systems constraints, the control loop is strictly separated into a modular **4-Stage Software Pipeline** executed sequentially at every simulation time step:

```mermaid
graph TD
    Start([Start of Control Loop]) --> Step[Frame Step: robot.step]
    Step --> Stage1[STAGE 1: PERCEPTION<br/>- Read Lidar, IMU & Camera<br/>- Filter sensor outliers]
    Stage1 --> Stage2[STAGE 2: ESTIMATION<br/>- Calibrate initial pose via Template Matching<br/>- Track pose via 3-Iteration Translation-Only ICP<br/>- Cluster obstacles & track with Low-Pass filter]
    Stage2 --> Stage3[STAGE 3: PLANNING<br/>- Compute 15D Observation vector<br/>- Evaluate ONNX Policy for Speed & Steer increments]
    Stage3 --> Stage4[STAGE 4: CONTROL<br/>- Compute Ackermann steering kinematics<br/>- Apply Electronic Differential for rear wheels]
    Stage4 --> Check{Simulation Active?}
    Check -- Yes --> Step
    Check -- No --> End([End Simulation])
```

Detailed documentation of coordinate transforms, localization mathematics, and mapping algorithms can be found in [docs/architecture.md](file:///c:/Users/fabia/Documents/WRO_FE_SIM/docs/architecture.md).

---

## ✨ Key Technical Highlights

*   **Curriculum Reinforcement Learning:** Trained using **PPO (Stable-Baselines3)** on a custom **Gymnasium** environment. The agent transitions from **Stage 1 (Safety focus)**, which teaches lane-keeping and basic driving at a constant speed, to **Stage 2 (Performance focus)**, optimizing lap times with variable speed control and aggressive curve-cutting.
*   **Translation-Only ICP:** Features scan-to-map matching via a fast 3-iteration Iterative Closest Point algorithm to maintain an accurate estimate of the robot's local pose `(x, y, yaw)`.
*   **Dynamic Obstacle Mapping:** Clusters LiDAR outliers (representing the $50\,\text{mm}$ colored obstacle boxes) and filters them dynamically. Implements a ray-casting visibility decay method to fade out obstacles once they are no longer in the vehicle's line-of-sight.
*   **Electronic Ackermann Differential:** Computes unequal velocities for the rear wheels based on the steering angle (slowing down the inner wheel, speeding up the outer wheel) for realistic cornering physics in Webots.
*   **Path Coloration Visualizer:** Draws a live trajectory path colored based on current velocity (from slow cyan/blue to fast orange/red) with an on-screen velocity legend.

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
Make sure you have [Webots](https://cyberbotics.com/) installed (tested with Webots R2023b or higher).

### 2. Setup Virtual Environment
Clone the repository and set up a virtual python environment:
```bash
git clone https://github.com/Bigfire3/wro-fe-rl-simulation.git
cd wro-fe-rl-simulation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
*(Make sure dependencies like `gymnasium`, `stable-baselines3`, `onnxruntime`, `opencv-python`, and `numpy` are installed.)*

### 3. Training the Agent
To start reinforcement learning training with curricular stages:
```bash
python controllers/wro_driver/train.py --timesteps 150000 --no-render
```
Monitor the training performance live via TensorBoard:
```bash
tensorboard --logdir=tb_logs
```

### 4. Running Simulation & Inference
Open the world file `worlds/track.wbt` in Webots. Press the **Play** button. The controller node will start automatically and read `models/wro_model.onnx` to drive the track.