# WRO Future Engineers 2026 - Embedded Software Architecture

## 1. The 4-Stage Software Pipeline
To maintain clean code separation and adhere to embedded software engineering best practices, the robot control loop MUST execute sequentially across 4 distinct stages during each execution frame:

* **[ STAGE 1: PERCEPTION ]** -> Parse raw LiDAR ranges, apply basic outlier filters.
* **[ STAGE 2: ESTIMATION ]** -> Pure geometric alignment, dual-hypothesis matching, obstacle grid mapping.
* **[ STAGE 3: PLANNING ]** -> High-level State Machine (Exploration mode handling), trajectory path generation.
* **[ STAGE 4: CONTROL ]** -> Actuator outputs (Ackermann steering calculations, speed controller).

---

## 2. Coordinate System Specifications

The codebase uses a strictly positive coordinate layout to align mathematical formulas directly with grid storage indices, eliminating negative array slicing bugs.

* **Real World Continuous Space ($P_r$):**
    * Origin $(0.0, 0.0)$ is located precisely at the **South-West inner corner** of the outer boundaries.
    * X-Axis: $0.0\,\text{m}$ to $3.0\,\text{m}$ (West to East).
    * Y-Axis: $0.0\,\text{m}$ to $3.0\,\text{m}$ (South to North).
* **Discrete Occupancy Grid Space ($P_v$):**
    * A static 2D `numpy` array of size $60 \times 60$ matrix entries.
    * Resolution: $1 \text{ cell} = 5\,\text{cm} \times 5\,\text{cm}$ ($0.05\,\text{m}$).
    * Conversion: `cell_x = int(x_real * 20)`, `cell_y = int(y_real * 20)`.
* **OpenCV Visualization Space ($P_{cv}$):**
    * A native $600 \times 600$ pixel layout cropped strictly to the inner arena.
    * $1 \text{ cell} = 10 \times 10$ pixels.
    * Conversion: `pixel_x = cell_x * 10 + 5`, `pixel_y = (59 - cell_y) * 10 + 5` (Y-axis inverted for matrix rendering).

---

## 3. Module Specification: `OpenCVLocalizer` (`estimation.py`)

The estimation module is fully standalone and interchangeable. It exposes a strict, standardized function signature so it can be cleanly swapped out with alternative algorithms (e.g., Particle Filters) later.

### 3.1. Standard Class Interface
* **Initialization:** `__init__(self, grid_size=60, resolution=0.05)`
* **Pipeline Entry:** `def update(self, lidar_ranges, max_range=2.0, angle_offset=-math.pi)`
    * *Returns:* `(float x_real, float y_real, float yaw_real)`
