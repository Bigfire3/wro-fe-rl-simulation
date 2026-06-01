import math
import numpy as np

from . import config

def ackermann_angles(target_angle):
    if abs(target_angle) < 1e-6:
        return 0.0, 0.0
    R = config.WHEELBASE / math.tan(abs(target_angle))
    inner = math.atan(config.WHEELBASE / (R - config.TRACK_FRONT / 2.0))
    outer = math.atan(config.WHEELBASE / (R + config.TRACK_FRONT / 2.0))
    
    # Positive target_angle turns right
    if target_angle > 0: 
        return outer, inner
    else: 
        return -inner, -outer

class Controller:
    def __init__(self):
        self.smoothed_steering = 0.0
        self.smoothed_speed = 0.0
        
    def reset(self):
        self.smoothed_steering = 0.0
        self.smoothed_speed = 0.0
        
    def apply(self, target_speed, target_steering, motor_left, motor_right, steer_left, steer_right, use_rl=True):
        """
        Processes target steering and speed targets using low-pass filtering,
        converts target steering into Ackermann steer angles, and applies 
        the controls to the Webots devices.
        
        Returns:
            speed (float): The actual velocity set on the motors.
            steering (float): The actual smoothed target steering angle.
        """
        self.smoothed_steering = target_steering
        self.smoothed_speed = target_speed

        if use_rl:
            speed = target_speed
        else:
            # Speed scaling based on steering angle
            speed_factor = 1.0 - (abs(self.smoothed_steering) / config.MAX_STEERING) * 0.3
            speed = target_speed * max(0.7, speed_factor)
            
        # Drive motors
        motor_right.setVelocity(speed)
        motor_left.setVelocity(speed)
        
        # Ackermann steering servos
        left_angle, right_angle = ackermann_angles(self.smoothed_steering)
        left_angle = float(np.clip(left_angle, -config.MAX_STEERING, config.MAX_STEERING))
        right_angle = float(np.clip(right_angle, -config.MAX_STEERING, config.MAX_STEERING))
        
        steer_left.setPosition(left_angle)
        steer_right.setPosition(right_angle)
        
        return speed, self.smoothed_steering
