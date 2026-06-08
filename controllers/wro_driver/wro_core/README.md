# `wro_core` - Estimation, Perception, Mapping & Control Library

This library contains the modular components that form the foundational helper modules for **Stage 1 (Perception)**, **Stage 2 (Estimation)**, and **Stage 4 (Control)** of the autonomous software architecture.

---

## 🛠️ Core Module Specifications

### 1. Localization & Calibration (`Estimation`)
*   **Template Matching (`opencv_localizer.py`):** Resolves initial starting pose and driving direction (`CW`/`CCW`) using template matching on the reference arena.
*   **Translation-Only ICP (`trans_icp_localizer.py`):** 
    *   Estimates the vehicle coordinates $(x, y)$ via a 3-iteration Iterative Closest Point algorithm.
    *   Matches current LiDAR points against the 8 static wall segments of the track.
    *   Extracts non-matching wall points (outliers with wall distance $> 15\,\text{cm}$) to pass to the `ObstacleMapper`.
    *   Tracks trajectory history and draws colored speed-dependent paths ($0.0\,\text{m/s}$ cyan/blue to $1.6\,\text{m/s}$ orange/red) with a visual color legend.

### 2. Dynamic Obstacle Mapping (`obstacle_mapper.py`)
*   **Clustering:** Groups LiDAR outlier points using a distance threshold of $10\,\text{cm}$. A cluster is registered as an obstacle if it contains at least 2 points.
*   **Tracking Filter:** Matches clusters to existing obstacles. Positions are filtered over time using a low-pass filter ($\alpha = 0.1$):
    $$P_{\text{new}} = (1 - \alpha) P_{\text{old}} + \alpha P_{\text{measured}}$$
*   **Visibility-Based Confidence Decay:** 
    *   Increases confidence ($+0.01$) when matched; decays confidence ($-0.01$) when within sensor range ($2.0\,\text{m}$) but not detected.
    *   Uses raycasting to check line-of-sight and prevent decay when obstacles are blocked by walls or other obstacles.

### 3. Actuator Control (`control.py`)
*   **Ackermann Kinematics:** Transforms target speed and steering angle into individual angles for the left and right steering knuckles.
*   **Direct Control Loop:** Avoids steering low-pass filtering to allow maximum responsiveness during high-frequency RL correction commands.
