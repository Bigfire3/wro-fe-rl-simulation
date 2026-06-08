<p align="center">
  <img src="media/Obs_Model_5.gif" width="60%" alt="Gym Environment Observation Debugger" />
</p>

# Reinforcement Learning Environment & Training Guide

This guide explains the Gymnasium reinforcement learning environment wrapper (`wro_gym_env.py`) interfacing with the Webots robot simulation, and outlines how to configure, start, and manage training runs.

---

## 📊 Environment Structure

### 1. Observation Space (15-Dimensional Vector)
The environment passes a normalized 15-dimensional float vector to the policy network, representing the ego-state, upcoming track geometry, and obstacle locations. All parameters are scaled and clipped to `[-1.0, 1.0]` using specific scaling factors (`config.NORM_FACTORS`):

| Index | Feature Name | Description | Default Value |
| :---: | :--- | :--- | :---: |
| **0** | `ego_v_x` | Current longitudinal velocity of the robot | $0.0\,\text{m/s}$ |
| **1** | `last_steering` | Steering angle from the previous control step | $0.0\,\text{rad}$ |
| **2** | `diff_yaw` | Heading error relative to the centerline tangent vector | $0.0\,\text{rad}$ |
| **3** | `inner_y_40` | Local Y-coordinate of the inner boundary at a $40\,\text{cm}$ lookahead | - |
| **4** | `inner_y_100` | Local Y-coordinate of the inner boundary at a $100\,\text{cm}$ lookahead | - |
| **5** | `outer_y_40` | Local Y-coordinate of the outer boundary at a $40\,\text{cm}$ lookahead | - |
| **6** | `outer_y_100` | Local Y-coordinate of the outer boundary at a $100\,\text{cm}$ lookahead | - |
| **7** | `corner_x_loc` | Local X-coordinate of the next inner corner beacon | - |
| **8** | `corner_y_loc` | Local Y-coordinate of the next inner corner beacon | - |
| **9** | `obs_1_x` | Local X-coordinate of the 1st (nearest) detected obstacle | $3.0\,\text{m}$ (empty) |
| **10** | `obs_1_y` | Local Y-coordinate of the 1st (nearest) detected obstacle | $0.0\,\text{m}$ (empty) |
| **11** | `obs_1_color` | Color category: `1.0` for Green (pass left), `-1.0` for Red (pass right) | $0.0$ (empty) |
| **12** | `obs_2_x` | Local X-coordinate of the 2nd nearest detected obstacle | $3.0\,\text{m}$ (empty) |
| **13** | `obs_2_y` | Local Y-coordinate of the 2nd nearest detected obstacle | $0.0\,\text{m}$ (empty) |
| **14** | `obs_2_color` | Color category: `1.0` for Green, `-1.0` for Red | $0.0$ (empty) |

### 2. Action Space (Continuous, 2D)
The policy outputs actions in the range `[-1.0, 1.0]` for two continuous dimensions:
*   **Steering Delta (Action 0):** Mapped to steer angle change in range `[-config.STEERING_DELTA_LIMIT, config.STEERING_DELTA_LIMIT]` radians. The absolute steering angle is clamped to `[-config.MAX_STEERING, config.MAX_STEERING]` radians.
*   **Speed Delta (Action 1):** Mapped to wheel speed change in range `[-10.0, 10.0]` rad/s. Active in Stage 2; target speed is clamped to `[0.0, config.MAX_MOTOR_VELOCITY]` rad/s.

---

## 📈 Curriculum Learning & Reward Shaping

The agent is trained using a **2-Stage Curriculum Learning** approach to decouple lane-keeping safety from speed optimization.

### Curriculum Stage 1: Safety & Line-Keeping
*   **Focus:** Drive safely around the track at a constant slow speed ($30\%$ of maximum motor velocity), ensuring correct obstacle bypass sides.
*   **Speed Reward:** $5.0 \times \text{speed-ratio}^2$.
*   **Steering Penalties:** Penalisations on absolute steering angle at speed to prevent swerving.
*   **Reward Equation:**
    $$R_{\text{Stage1}} = R_{\text{progress}} + R_{\text{speed}} + P_{\text{step}} + P_{\text{smooth}} + P_{\text{steer-amp}}$$
*   **Episode Termination:** Triggered instantly on any collision (absolute penalty: $-50.0$) or on passing an obstacle on the incorrect side (absolute penalty: $-50.0$).

### Curriculum Stage 2: Performance & Obstacle Avoidance
*   **Focus:** Maximize speed, optimize trajectories (cut corners), and dodge obstacles dynamically under active velocity control.
*   **Speed Reward:** $10.0 \times \text{speed-ratio}^2$ (quadratic encouragement to drive at the limits).
*   **Lap Completion Time Bonus:** Time-based reward awarded upon completing a lap ($R_{\text{bonus}} = \max(50.0, 400.0 - 4.0 \times \text{lap-steps})$) to reward speed.
*   **Reduced Steer/Smooth Penalties:** Allows agile, sudden maneuvers for corner-cutting and dodging.
*   **Heavy Step Penalty:** $P_{\text{step}} = -3.0$ per step, rendering elapsed time expensive.
*   **Reward Equation:**
    $$R_{\text{Stage2}} = R_{\text{progress}} + R_{\text{speed-quad}} + P_{\text{step-high}} + P_{\text{smooth-low}}$$
*   **Episode Termination:** Collision penalty is reduced to a smaller additive penalty ($-40.0$) rather than hard termination. This encourages the agent to aggressively explore the limits of the track walls during speed optimization.

---

## 🛠️ Key Scripts

*   [wro_gym_env.py](../controllers/wro_driver/wro_gym_env.py): Extends `gymnasium.Env` to interface with the Webots Supervisor. Handles physics resets, obstacle randomization, sensor updates, coordinate transformations, curriculum thresholds, and reward calculation.
*   [train.py](../controllers/wro_driver/train.py): Runs training using Stable-Baselines3 PPO. Implements automatic checkpoint saving and curriculum-stage transition parameters.
*   [export_onnx.py](../controllers/wro_driver/export_onnx.py): Serializes the trained PyTorch PPO policy network into a standalone ONNX model for deployment.

---

## 🚀 How to Start and Manage Your Own Training

To train your own Reinforcement Learning model from scratch, follow these instructions.

### 1. Load the Training World in Webots
1. Launch Webots.
2. Open the training world: `worlds/track_training.wbt`.
3. In the scene tree on the left, find the robot node (named `AckermannVehicle` or similar).
4. Expand the robot properties and ensure the `controller` field is set to `<extern>`. This configuration detaches the built-in controller and allows the external Gymnasium Python environment to drive the simulation.

### 2. Start the Training Script
Run the training script from the root of your project directory using the virtual environment:

```bash
# Windows
.venv\Scripts\activate
python controllers/wro_driver/train.py [parameters]

# Linux / macOS
source .venv/bin/activate
python controllers/wro_driver/train.py [parameters]
```

### 3. Training Script Parameters (`train.py`)
You can customize the training run using the following command-line flags:

*   `--timesteps <int>`: Total number of training steps (default: `500000`).
*   `--no-render`: Disables OpenCV window rendering during training. Excluding this flag opens debug visualization windows showing LiDAR inliers/outliers, camera feed, and the current observation vector state.
*   `--continue-training` / `-c`: Continues training an existing checkpoint file (`models/wro_ppo_model.zip`) if it exists. Note that continuing resets the standard deviation of exploration noise (setting `log_std` to `-1.2`, i.e., `std ≈ 0.3`) to promote safe exploration of faster trajectories without policy collapse.
*   `--stage <1|2>`: Forces a specific curriculum stage (1 or 2). If not specified, standard automatic curriculum logic is used (transits from Stage 1 to Stage 2 at 200,000 steps).

**Example Commands:**
```bash
# Start a default training run for 300,000 steps with active OpenCV rendering
python controllers/wro_driver/train.py --timesteps 300000

# Continue training from the last saved model and force Curriculum Stage 2
python controllers/wro_driver/train.py --continue-training --stage 2
```

### 4. Monitoring Progress with TensorBoard
TensorBoard logs are saved in the `tb_logs/` folder. You can launch TensorBoard to monitor training rewards, learning rate, entropy, and current curriculum stage:

```bash
tensorboard --logdir=tb_logs
```
Then open `http://localhost:6006` in your web browser.

### 5. Managing the Training Loop
*   **Aborting Training Cleanly:** If rendering is enabled, you can focus the OpenCV window and press **`q`** or **`Esc`**. This raises a clean abort signal, terminating the loop without saving corrupted states.
*   **Saving Checkpoints:** When the script finishes successfully, the trained model is automatically saved as `models/wro_ppo_model.zip`.
*   **Exporting to ONNX:** To run inference inside Webots using the default simulator controller, convert the saved zip file into an ONNX model:
    ```bash
    python controllers/wro_driver/export_onnx.py
    ```
    This script reads `models/wro_ppo_model.zip` and outputs `models/wro_model.onnx` which is utilized by `wro_driver.py` during inference.
