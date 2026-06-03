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
        
    def apply(self, target_speed, target_steering, motor_left, motor_right, steer_left, steer_right):
        """
        Processes target steering and speed targets,
        converts target steering into Ackermann steer angles, and applies 
        the controls to the Webots devices.
        
        Returns:
            speed (float): The actual velocity set on the motors.
            steering (float): The actual smoothed target steering angle.
        """
        self.smoothed_steering = target_steering
        self.smoothed_speed = target_speed

        speed = target_speed
            
        # Electronic differential for rear wheels
        # If turning right (smoothed_steering > 0), the right wheel is inner (slower) and left is outer (faster).
        diff_factor = (config.TRACK_FRONT / (2.0 * config.WHEELBASE)) * math.tan(self.smoothed_steering)
        speed_left = speed * (1.0 + diff_factor)
        speed_right = speed * (1.0 - diff_factor)
        
        # Scale down if either wheel speed exceeds physical limits
        max_wheel_speed = max(abs(speed_left), abs(speed_right))
        if max_wheel_speed > config.MAX_MOTOR_VELOCITY:
            scale = config.MAX_MOTOR_VELOCITY / max_wheel_speed
            speed_left *= scale
            speed_right *= scale

        # Drive motors
        motor_right.setVelocity(speed_right)
        motor_left.setVelocity(speed_left)
        
        # Ackermann steering servos
        left_angle, right_angle = ackermann_angles(self.smoothed_steering)
        left_angle = float(np.clip(left_angle, -config.MAX_STEERING, config.MAX_STEERING))
        right_angle = float(np.clip(right_angle, -config.MAX_STEERING, config.MAX_STEERING))
        
        steer_left.setPosition(left_angle)
        steer_right.setPosition(right_angle)
        
        return speed, self.smoothed_steering
