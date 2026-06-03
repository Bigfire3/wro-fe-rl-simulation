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
        
    def add_calibration_scan(self, lidar_ranges):
        """
        Adds a LiDAR scan for initial calibration.
        Accumulates scans, filters invalid values, and performs template matching calibration
        when 10 scans are collected.
        Returns (x_init, y_init, yaw_init, direction, debug_img) if calibration is successful,
        otherwise returns None.
        """
        if self.initial_pose_found:
            return None
            
        if len(lidar_ranges) > 0:
            self.collected_scans.append(lidar_ranges)
            if len(self.collected_scans) >= 10:
                scans_arr = np.array(self.collected_scans)
                # Filter invalid values
                invalid_mask = (scans_arr <= 0.01) | (scans_arr >= 2.0) | np.isinf(scans_arr) | np.isnan(scans_arr)
                scans_arr[invalid_mask] = np.nan
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_ranges = np.nanmean(scans_arr, axis=0)
                avg_ranges = np.nan_to_num(avg_ranges, nan=0.0)
                
                x_init, y_init, yaw_init, direction, debug_img = self.calibrate_from_scans(avg_ranges)
                self.set_calibrated_pose(x_init, y_init, yaw_init, direction)
                return x_init, y_init, yaw_init, direction, debug_img
        return None
        
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
        ego_v_x = sensor_data.get("ego_v_x", 0.0)
        
        # Continuous ICP update
        rx, ry, ryaw, outliers = self.icp_localizer.update(
            lidar_ranges=lidar_ranges,
            imu_yaw=imu_yaw,
            max_range=2.0,
            ego_v_x=ego_v_x
        )
        
        # Update obstacle mapper
        self.obstacle_mapper.update([rx, ry, ryaw], outliers)
        
        if "camera_image" in sensor_data:
            self.obstacle_mapper.update_obstacle_colors(sensor_data["camera_image"], [rx, ry, ryaw])
            
        return rx, ry, ryaw
