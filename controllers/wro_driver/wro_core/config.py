import math
import numpy as np

# --- Physical Robot Dimensions ---
WHEELBASE = 0.14          # m
TRACK_FRONT = 0.12        # m
WHEEL_RADIUS = 0.03       # m
MAX_STEERING = 0.8        # rad
STEERING_DELTA_LIMIT = 0.35 # rad max steering change per control step (was 0.2)

# --- Limits and Speeds ---
MAX_MOTOR_VELOCITY = 100.0  # rad/s max motor velocity

# --- Training / Gym Environment configuration ---
CONTROL_FREQ = 10          # Hz
TIME_STEP = 20             # ms

# --- Observation Space Normalization Factors ---
NORM_FACTORS = np.array([
    1.0,                         # ego_v_x (overwritten at runtime)
    1.0 / MAX_STEERING,          # last_steering
    1.0 / (math.pi / 2.0),       # diff_yaw (heading_error)
    1.0 / 0.5,                   # inner_y_40
    1.0 / 0.5,                   # inner_y_100
    1.0 / 0.5,                   # outer_y_40
    1.0 / 0.5,                   # outer_y_100
    1.0 / 2.5,                   # corner_x_loc
    1.0 / 1.5,                   # corner_y_loc
    1.0 / 3.0,                   # obs1_x_loc
    1.0 / 2.0,                   # obs1_y_loc
    1.0,                         # obs1_color
    1.0 / 3.0,                   # obs2_x_loc
    1.0 / 2.0,                   # obs2_y_loc
    1.0                          # obs2_color
], dtype=np.float32)

# --- PPO Hyperparameters ---
PPO_LEARNING_RATE = 3e-4      # Learning rate for the optimizer (default: 3e-4)
PPO_GAMMA = 0.995             # Discount factor (long-horizon planning for strategic corner-cutting)
PPO_BATCH_SIZE = 256          # Minibatch size for gradient descent updates
PPO_ENT_COEF = 0.005          # Entropy coefficient (lowered to 0.005 to prevent standard deviation explosion)
PPO_N_STEPS = 4096            # Number of steps to run for each environment per update (doubled for stable gradient estimation)
PPO_N_EPOCHS = 15             # Number of epochs when optimizing the surrogate loss
PPO_GAE_LAMBDA = 0.95         # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
PPO_CLIP_RANGE = 0.2          # Clipping parameter for the policy objective
PPO_VF_COEF = 1.0             # Value function coefficient in the loss calculation
PPO_NET_ARCH = dict(pi=[128, 128], vf=[256, 128])  # Network architecture (separate, larger value network)

# --- PPO Continue Training Settings ---
PPO_CONTINUE_LEARNING_RATE = 3e-4  # Learning rate during continued training
PPO_CONTINUE_ENT_COEF = 0.01       # Entropy coefficient during continued training (slightly higher to boost exploration)
PPO_CONTINUE_LOG_STD = -1.2        # Reset exploration noise standard deviation to exp(-1.2) ≈ 0.3 (was ≈ 0.5)
