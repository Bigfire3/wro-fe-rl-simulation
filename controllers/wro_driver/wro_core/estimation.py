import math
import numpy as np
import cv2
import warnings

from .opencv_localizer import OpenCVLocalizer
from .trans_icp_localizer import TranslationICPLocalizer
from .obstacle_mapper import ObstacleMapper

class StateEstimator:
    def __init__(self):
        self.localizer = OpenCVLocalizer()
        self.icp_localizer = TranslationICPLocalizer()
        self.obstacle_mapper = ObstacleMapper()
        self.initial_pose_found = False
        self.driving_direction = None
        self.collected_scans = []
        
    def reset(self, start_x=None, start_y=None, yaw=None, direction=None):
        self.obstacle_mapper.obstacles = []
        self.initial_pose_found = False
        self.driving_direction = direction
        self.collected_scans = []
        if start_x is not None and start_y is not None and yaw is not None:
            self.localizer.set_initial_pose(start_x, start_y, yaw)
            self.icp_localizer.set_initial_pose(start_x, start_y, yaw)
            self.initial_pose_found = True
            
    def calibrate_from_scans(self, avg_ranges):
        """
        Runs template matching initial calibration.
        """
        n_rays = len(avg_ranges)
        angle_inc = -2.0 * math.pi / n_rays if n_rays > 0 else 0.0
        angle_offset = math.pi / 2 # 90 degrees
        
        x_init, y_init, yaw_init, direction, debug_img = self.localizer.calibrate_initial_pose(
            avg_ranges=avg_ranges,
            angle_offset=angle_offset,
            angle_inc=angle_inc
        )
        return x_init, y_init, yaw_init, direction, debug_img
        
    def set_calibrated_pose(self, x, y, yaw, direction):
        self.localizer.set_initial_pose(x, y, yaw)
        self.icp_localizer.set_initial_pose(x, y, yaw)
        self.driving_direction = direction
        self.initial_pose_found = True

    def update(self, sensor_data):
        """
        Updates the pose estimate and obstacle map using the latest sensor data.
        Returns (x, y, yaw) pose.
        """
        if not self.initial_pose_found:
            raise RuntimeError("StateEstimator must be calibrated/initialized before calling update().")
            
        lidar_ranges = sensor_data["lidar_ranges"]
        imu_yaw = sensor_data["imu_yaw"]
        
        # Continuous ICP update
        rx, ry, ryaw, outliers = self.icp_localizer.update(
            lidar_ranges=lidar_ranges,
            imu_yaw=imu_yaw,
            max_range=2.0
        )
        
        # Update obstacle mapper
        self.obstacle_mapper.update([rx, ry, ryaw], outliers)
        
        if "camera_image" in sensor_data:
            self.obstacle_mapper.update_obstacle_colors(sensor_data["camera_image"], [rx, ry, ryaw])
            
        return rx, ry, ryaw
