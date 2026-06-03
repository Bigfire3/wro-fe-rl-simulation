# WRO Future Engineers - Webots Simulation

This repository contains the Webots simulation for the WRO Future Engineers category.

## Project Structure

- `controllers/wro_driver/`: Main autonomous robot controller (supporting Rule-based and RL modes).
- `controllers/keyboard_driver/`: Keyboard controller for manual driving and testing.
- `worlds/`: Webots world files (`track.wbt` for evaluation, `track_training.wbt` for RL training).
- `protos/`: Custom robot and object prototypes.
- `docs/`: In-depth documentation of system architecture, state machine, and rules.
- `models/`: Trained ONNX models (`wro_model.onnx`).

## Features

- **Multi-Mode Navigation**: Supports both classic **Rule-based** path planning (PD control with obstacle avoidance) and **Reinforcement Learning (RL)** (ONNX inference with incremental action control).
- **Translation-Only ICP**: Scan-to-map matching via 3-iteration ICP with outlier classification.
- **Dynamic Obstacle Mapping**: Clustering-based obstacle mapper with visibility-based decay and confidence visualization.
- **Electronic Differential**: Rear-wheel velocity adjustment based on steering angle for realistic turning.
- **Speed-Dependent Path Coloration**: Live visualization path colored based on robot velocity (0.0 to 1.6 m/s) with a visual legend.

## Getting Started

1. Install [Webots](https://cyberbotics.com/).
2. Clone this repository.
3. Set up the virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt  # If applicable, or install onnxruntime, opencv-python, numpy
   ```
4. Open `worlds/track.wbt` in Webots. The controller `wro_driver` should start automatically.

### Running & Training RL

- To train the RL model:
  ```bash
  .venv\Scripts\python.exe controllers/wro_driver/train.py --timesteps 150000 --no-render
  ```

---
*Created for WRO Future Engineers.*