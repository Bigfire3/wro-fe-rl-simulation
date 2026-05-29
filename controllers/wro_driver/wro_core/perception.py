import numpy as np
import cv2

def read_sensors(lidar, imu, camera, imu_yaw_initial=None):
    """
    Reads LiDAR ranges, IMU orientation, and Camera images from Webots devices.
    Normalizes IMU yaw to start at 0 if imu_yaw_initial is supplied or None.
    
    Returns:
        sensor_dict (dict): Dictionary with lidar_ranges, imu_yaw, and camera_image (if available).
        imu_yaw_initial (float): The base IMU value used for relative nulling.
    """
    lidar_data = lidar.getRangeImage()
    if lidar_data is None:
        lidar_data = []
        
    try:
        rpy = imu.getRollPitchYaw()
        imu_yaw_raw = rpy[2] if rpy else 0.0
    except Exception:
        imu_yaw_raw = 0.0
        
    if imu_yaw_initial is None:
        imu_yaw_initial = imu_yaw_raw
        
    imu_yaw = imu_yaw_raw - imu_yaw_initial
    
    img_bgr = None
    try:
        w = camera.getWidth()
        h = camera.getHeight()
        img_buffer = camera.getImage()
        if img_buffer:
            img_raw = np.frombuffer(img_buffer, dtype=np.uint8).reshape((h, w, 4))
            img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)
    except Exception:
        pass
        
    sensor_dict = {
        "lidar_ranges": lidar_data,
        "imu_yaw": imu_yaw
    }
    if img_bgr is not None:
        sensor_dict["camera_image"] = img_bgr
        
    return sensor_dict, imu_yaw_initial
