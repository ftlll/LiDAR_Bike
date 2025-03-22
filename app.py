import os
import re
import struct
import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

BASE_PATH = "data_q123"
LIDAR_PATH = os.path.join(BASE_PATH, "lidar/data/")
IMAGE_PATH = os.path.join(BASE_PATH, "image_02/data/")

#########################################
##
## Step 1: Reading Sensor Data
##
#########################################

# imu.json: {
#   timestamp: float,
#   angular_velocity: {
#     x, y, z
#   },
#   linear_acceleration: {
#     x, y, z
#   }
# }[]

# gps.json: {
#   timestamp: float,
#   latitude: float,
#   longitude: float,
#   altitude: float
#  }[]

def read_image(file_path):
    f = open(f'{file_path}', 'rb')
    width, height = struct.unpack("II", f.read(8))
    image_data = np.frombuffer(f.read(), dtype=np.uint8)

    # H * W * 3 => (R, G, B)
    image = image_data.reshape((height, width, 3))
    file_name = file_path.name
    match = re.search(r"image_(\d{10})_(\d{9})\.bin", file_name)
    if match:
        timestamp = float(f"{match.group(1)}.{match.group(2)}")

    return image, timestamp

def read_LiDAR(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

    file_name = file_path.name
    match = re.search(r"lidar_(\d{10})_(\d{9})\.bin", file_name)
    if match:
        timestamp = float(f"{match.group(1)}.{match.group(2)}")

    return points, timestamp

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_data(folder):
    gps_data = read_json(f"{folder}/gps.json")
    imu_data = read_json(f"{folder}/imu.json")

    gps_df = pd.DataFrame(gps_data)
    imu_df = pd.DataFrame(imu_data)

    images = []
    image_timestamps = []
    image_folder = Path(f'{folder}/images')
    for img_file in sorted(image_folder.iterdir()):
        if img_file.suffix == ".bin":
            img_data, timestamp = read_image(img_file)
            images.append(img_data)
            image_timestamps.append(timestamp)
    images = np.array(images)

    lidars = []
    lidar_timestamps = []
    lidar_folder = Path(f'{folder}/lidar')
    for lidar_file in sorted(lidar_folder.iterdir()):
        if lidar_file.suffix == ".bin":
            lidar_data, timestamp = read_LiDAR(lidar_file)
            lidars.append(lidar_data)
            lidar_timestamps.append(timestamp)
    lidars = np.array(lidars)

    return gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps

#########################################
##
## Step 2: Synchronizing Sensor Data
##
#########################################

def closest_index(timestamps, reference_timestamp):
    differences = [abs(t - reference_timestamp) for t in timestamps]
    return differences.index(min(differences))

def synchronize_sensor_data(gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps):
    # output: list of sync data contains all four sensors
    sync_data = []
    reference_timestamps = []

    ## compare frame rate
    frame_img = (image_timestamps[-1] - image_timestamps[0]) / len(image_timestamps)
    frame_lidar = (lidar_timestamps[-1] - lidar_timestamps[0]) / len(lidar_timestamps)

    reference_timestamps = max(frame_img, frame_lidar)
    # based on my observation, it seems images and lidars have close timestamp range
    # TODO: also, we need to make sure about the timestamp range

    for reference_timestamp in reference_timestamps:
        closest_image_idx = closest_index(image_timestamps, reference_timestamp)
        # print("closest_image_idx", closest_image_idx)
        closest_image = images[closest_image_idx]
        closest_lidar_idx = closest_index(lidar_timestamps, reference_timestamp)
        # print("closest_lidar_idx", closest_lidar_idx)
        closest_lidar = lidars[closest_lidar_idx]
        closest_gpu = gps_df.iloc[(gps_df['timestamp'] - reference_timestamp).abs().idxmin()]
        closest_imu = imu_df.iloc[(imu_df['timestamp'] - reference_timestamp).abs().idxmin()]
        sync_data.append({
            "gpu": closest_gpu,
            "imu": closest_imu,
            "image": closest_image,
            "lidar": closest_lidar,
        })

    return sync_data

# gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps = read_data(folder)
# sync_data = synchronize_sensor_data(gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps)

#########################################
##
## Step 3: Creating a Lightweight API
##
#########################################

app = Flask(__name__)

@app.route('/sync_data', methods=['POST'])
def sync_data():
    """API endpoint to synchronize sensor data from a given folder."""
    data = request.json
    folder_path = data.get("folder_path")
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({"error": "Invalid folder path"}), 400

    gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps = read_data(folder_path)
    synchronized_data = synchronize_sensor_data(gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps)

    return jsonify(synchronized_data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

#########################################
##
## Bonus: Synchronizing Sensor Data
##
#########################################
## Useful Links:
## https://leimao.github.io/blog/Camera-Intrinsics-Extrinsics/
## 'image_02': left rectified color image sequence
## https://github.com/yanii/kitti-pcl/blob/master/KITTI_README.TXT
## original paper: https://www.cvlibs.net/publications/Geiger2013IJRR.pdf
## reference: https://github.com/MikeS96/lidar_cam_sonar/blob/master/image_laser_fusion.ipynb
##
## calib_cam_to_cam.txt: Camera-to-camera calibration
## --------------------------------------------------
##   - S_xx: 1x2 size of image xx before rectification
##   - K_xx: 3x3 calibration matrix of camera xx before rectification
##   - D_xx: 1x5 distortion vector of camera xx before rectification
##   - R_xx: 3x3 rotation matrix of camera xx (extrinsic)
##   - T_xx: 3x1 translation vector of camera xx (extrinsic)
##   - S_rect_xx: 1x2 size of image xx after rectification
##   - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
##   - P_rect_xx: 3x4 projection matrix after rectification
## Note: When using this dataset you will most likely need to access only
## P_rect_xx, as this matrix is valid for the rectified image sequences.
##
## calib_velo_to_cam.txt: Velodyne-to-camera registration
## ------------------------------------------------------
##   - R: 3x3 rotation matrix
##   - T: 3x1 translation vector
##   - delta_f: deprecated
##   - delta_c: deprecated
## R|T takes a point in Velodyne coordinates and transforms it into the
## coordinate system of the left video camera. Likewise it serves as a
## representation of the Velodyne coordinate frame in camera coordinates.

