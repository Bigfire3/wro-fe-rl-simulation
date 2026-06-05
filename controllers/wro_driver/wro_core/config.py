import math
import numpy as np

# --- Physical Robot Dimensions ---
WHEELBASE = 0.14          # m
TRACK_FRONT = 0.12        # m
WHEEL_RADIUS = 0.03       # m
MAX_STEERING = 0.8        # rad

# --- Limits and Speeds ---
MAX_MOTOR_VELOCITY = 100.0  # rad/s max motor velocity

# --- Training / Gym Environment configuration ---
CONTROL_FREQ = 10          # Hz
TIME_STEP = 20             # ms

# --- Observation Space Normalization Factors ---
NORM_FACTORS = np.array([
    1.0,                         # ego_v_x (overwritten at runtime)
    1.0 / MAX_STEERING,          # smoothed_steering
    1.0 / (math.pi / 2.0),       # diff_yaw (heading_error)
    1.0 / 0.5,                   # inner_y_40
    1.0 / 0.5,                   # inner_y_100
    1.0 / 0.5,                   # outer_y_40
    1.0 / 0.5,                   # outer_y_100
    1.0 / 2.5,                   # corner_x_loc
    1.0 / 1.5,                   # corner_y_loc
    1.0 / 2.0,                   # obs1_x_loc
    1.0 / 2.0,                   # obs1_y_loc
    1.0,                         # obs1_color
    1.0 / 2.0,                   # obs2_x_loc
    1.0 / 2.0,                   # obs2_y_loc
    1.0                          # obs2_color
], dtype=np.float32)

