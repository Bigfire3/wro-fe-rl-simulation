# `wro_core` - Estimation, Perception, Mapping & Control Library

This library contains the modular components that form the foundational helper modules for **Stage 1 (Perception)**, **Stage 2 (Estimation)**, and **Stage 4 (Control)** of the autonomous software architecture.

---

## 🛠️ Core Module Specifications

### 1. Localization & Calibration (`Estimation`)
*   **Template Matching (`opencv_localizer.py`):** Resolves initial starting pose and driving direction (`CW`/`CCW`) using OpenCV template matching on the reference arena layout.
*   **Translation-Only ICP (`trans_icp_localizer.py`):** 
    *   Estimates the vehicle coordinate offsets $(x, y)$ relative to the arena origin $(0, 0)$ via a 3-iteration Iterative Closest Point algorithm.
    *   Matches current LiDAR points against the 8 static wall segments of the track (4 outer walls, 4 inner core walls).
    *   Extracts non-matching wall points (outliers with wall distance $> 15\,\text{cm}$) to pass as obstacle candidate points to the `ObstacleMapper`.
    *   Tracks trajectory history and draws colored speed-dependent paths ($0.0\,\text{m/s}$ cyan/blue to $1.6\,\text{m/s}$ orange/red) with a visual color legend overlay.

### 2. Dynamic Obstacle Mapping (`obstacle_mapper.py`)
*   **Clustering:** Groups LiDAR outlier points using a distance threshold of $10\,\text{cm}$. A cluster is registered as an obstacle candidate if it contains at least 2 points.
*   **Tracking Filter:** Matches clusters to existing obstacles or creates new tracking entries. Positions are filtered over time using a low-pass filter ($\alpha = 0.1$):
    $$P_{new} = (1 - \alpha) P_{old} + \alpha P_{measured}$$
*   **Visibility-Based Confidence Decay:** 
    *   Every step, the confidence score of obstacles increases ($+0.01$) when matched.
    *   If an obstacle is within the sensor range ($2.0\,\text{m}$) but not matched, its confidence decays ($-0.01$).
    *   To prevent decay of obstacles that are blocked from view, the mapper runs raycasting from the robot's pose to check line-of-sight. If a wall segment or another obstacle blocks the ray, the confidence decay is bypassed.

### 3. Actuator Control (`control.py`)
*   **Ackermann Kinematics:** Transforms target speed and target steering angle into individual angles for the left and right steering knuckles.
*   **Electronic Rear-Wheel Differential:** Corrects rear wheel velocities dynamically when cornering:
    *   Inner wheel slows down to prevent wheel spin and slipping.
    *   Outer wheel speeds up to maintain lateral torque stability.
    $$v_{inner} = v_{target} \left(1 - \frac{W \cdot \tan(\theta)}{2 \cdot L}\right)$$
    $$v_{outer} = v_{target} \left(1 + \frac{W \cdot \tan(\theta)}{2 \cdot L}\right)$$
    *(where $W$ is the wheel track width, $L$ is the wheelbase length, and $\theta$ is the steering angle).*
*   **Direct Control Loop:** Avoids steering low-pass filtering to allow maximum responsiveness during high-frequency RL correction commands.
