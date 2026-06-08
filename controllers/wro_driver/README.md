<p align="center">
  <img src="../../docs/media/Obs_Model_5.gif" width="60%" alt="Gym Environment Observation Debugger" />
</p>

# Gymnasium Reinforcement Learning Environment (`WebotsWroEnv`)

This directory houses the core Gymnasium environment wrapper (`wro_gym_env.py`) interfacing with the Webots robot simulation, along with training and evaluation tools.

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

*   [wro_gym_env.py](file:///c:/Users/fabia/Documents/WRO_FE_SIM/controllers/wro_driver/wro_gym_env.py): Extends `gymnasium.Env` to interface with the Webots Supervisor. Handles physics resets, obstacle randomization, sensor updates, coordinate transformations, curriculum thresholds, and reward calculation.
*   [train.py](file:///c:/Users/fabia/Documents/WRO_FE_SIM/controllers/wro_driver/train.py): Runs training using Stable-Baselines3 PPO. Implements automatic checkpoint saving and curriculum-stage transition parameters.
*   [export_onnx.py](file:///c:/Users/fabia/Documents/WRO_FE_SIM/controllers/wro_driver/export_onnx.py): Serializes the trained PyTorch PPO policy network into a standalone ONNX model for deployment.
