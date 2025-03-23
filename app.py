import os
import re
import json
import numpy as np
import pandas as pd
import cv2
import struct
from pathlib import Path
from flask import Flask, request, jsonify

#########################################
##
## Step 0: Preparation
##
#########################################
# What is IMU https://www.youtube.com/watch?v=fG-JQlzQxWQ
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
#   latitude: float,     x
#   longitude: float,    y
#   altitude: float      z
#  }[]
BASE_PATH = "data_q123"
BONUS_PATH = "data_bonus"

#########################################
##
## Step 1: Reading Sensor Data
##
#########################################

def load_image(file_path):
    f = open(f'{file_path}', 'rb')
    width, height = struct.unpack("II", f.read(8))
    image_data = np.frombuffer(f.read(), dtype=np.uint8)

    # H * W * 3 => (R, G, B)
    image = image_data.reshape((height, width, 3))
    file_name = file_path.name
    match = re.search(r"image_(\d{10})_(\d{9})\.bin", file_name)
    if match:
        timestamp = [float(f"{match.group(1)}.{match.group(2)}"), file_name]

    return image, timestamp

def load_LiDAR(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

    file_name = file_path.name
    match = re.search(r"lidar_(\d{10})_(\d{9})\.bin", file_name)
    if match:
        timestamp = [float(f"{match.group(1)}.{match.group(2)}"), file_name]

    return points, timestamp

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_files(folder_path):
    LIDAR_PATH = os.path.join(folder_path, "lidar/")
    IMAGE_PATH = os.path.join(folder_path, "images/")
    GPS_PATH = os.path.join(folder_path, "gps.json")
    IMU_PATH = os.path.join(folder_path, "imu.json")

    gps_df = load_json(GPS_PATH)
    imu_df = load_json(IMU_PATH)

    images = []
    image_timestamps = []
    image_folder = Path(IMAGE_PATH)
    for img_file in sorted(image_folder.iterdir()):
        if img_file.suffix == ".bin":
            img, timestamp = load_image(img_file)
            print(timestamp)
            images.append(img)
            image_timestamps.append(timestamp)
    images = np.array(images)

    lidars = []
    lidar_timestamps = []
    lidar_folder = Path(LIDAR_PATH)
    for lidar_file in sorted(lidar_folder.iterdir()):
        if lidar_file.suffix == ".bin":
            points, timestamp = load_LiDAR(lidar_file)
            lidars.append(points)
            lidar_timestamps.append(timestamp)
    lidars = np.array(lidars)

    return gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps

#########################################
##
## Step 2: Synchronizing Sensor Data
##
#########################################

def closest_index(timestamps, reference_timestamp):
    diff = [abs(t[0] - reference_timestamp) for t in timestamps]
    return diff.index(min(diff))

def linear_interp(v1, v2, t1, t2, t):
    return v1 + (v2 - v1) * (t - t1) / (t2 - t1)

def gps_interp(timestamp, gps_df):
    if timestamp < gps_df.iloc[0]["timestamp"]:
        return gps_df.iloc[0].to_dict()
    if timestamp > gps_df.iloc[-1]["timestamp"]:
        return gps_df.iloc[-1].to_dict()

    index = (gps_df["timestamp"] > timestamp).idxmax()
    gps1 = gps_df.iloc[index - 1]
    gps2 = gps_df.iloc[index]
    t1 = gps1['timestamp']
    t2 = gps2['timestamp']

    lat_interp = linear_interp(gps1["latitude"], gps2["latitude"], t1, t2, timestamp)
    lon_interp = linear_interp(gps1["longitude"], gps2["longitude"], t1, t2, timestamp)
    alt_interp = linear_interp(gps1["altitude"], gps2["altitude"], t1, t2, timestamp)

    return {
        "timestamp": timestamp,
        "latitude": lat_interp,
        "longitude": lon_interp,
        "altitude": alt_interp
    }

def synchronize_sensor_data(gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps):
    sync_data = []
    reference_timestamps = []

    ## compare average frame rate
    img_fps = len(image_timestamps) / (image_timestamps[-1][0] - image_timestamps[0][0])
    lidar_fps = len(lidar_timestamps) / (lidar_timestamps[-1][0] - lidar_timestamps[0][0])
    reference_timestamps = image_timestamps if lidar_fps > img_fps else lidar_timestamps

    for reference_timestamp, _ in reference_timestamps:
        closest_image_idx = closest_index(image_timestamps, reference_timestamp)
        closest_image = images[closest_image_idx]
        image_timestamp = image_timestamps[closest_image_idx]
        closest_lidar_idx = closest_index(lidar_timestamps, reference_timestamp)
        closest_lidar = lidars[closest_lidar_idx]
        lidar_timestamp = lidar_timestamps[closest_lidar_idx]
        closest_imu = imu_df.iloc[(imu_df['timestamp'] - reference_timestamp).abs().idxmin()]
        gps = gps_interp(reference_timestamp, gps_df)
        sync_data.append({
            "gps": gps,
            "imu": closest_imu.to_dict(),
            "image": image_timestamp[1],
            "lidar": lidar_timestamp[1],
        })

    return sync_data

# gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps = load_data(BASE_PATH)
# sync_data = synchronize_sensor_data(gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps)
# print(sync_data)

#########################################
##
## Step 3: Creating a Lightweight API
##
#########################################

app = Flask(__name__)

@app.route('/sync_data', methods=['GET'])
def sync_data():
    folder_path = request.args.get("folder_path")
    if not folder_path:
        return jsonify({"error": "Invalid folder path"}), 400

    gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps = load_files(folder_path)
    synchronized_data = synchronize_sensor_data(gps_df, imu_df, images, lidars, image_timestamps, lidar_timestamps)

    return jsonify(synchronized_data)

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

CALIB_PATH = os.path.join(BONUS_PATH, "calib/")
LIDAR_PATH = os.path.join(BONUS_PATH, "lidar/data/")
IMAGE_PATH = os.path.join(BONUS_PATH, "image_02/data/")
OUTPUT_PATH = os.path.join(BONUS_PATH, "output/")

# create output path
os.makedirs(OUTPUT_PATH, exist_ok=True)

# load matrices might be useful
def load_calib(file_path):
    data = {}
    with open(f'{file_path}', 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            if key == "P_rect_02":
                data[key] = np.array([float(x) for x in value.split()]).reshape(3, 4)
            elif key == "P_02" or key == "R":
                data[key] = np.array([float(x) for x in value.split()]).reshape(3, 3)
            elif key == "T":
                data[key] = np.array([float(x) for x in value.split()]).reshape(3, 1)
    return data

c2c_path = os.path.join(CALIB_PATH, "calib_cam_to_cam.txt")
cam_to_cam_data = load_calib(c2c_path)
v2c_path = os.path.join(CALIB_PATH, "calib_velo_to_cam.txt")
velo_to_cam_data = load_calib(v2c_path)

P_rect_02 = cam_to_cam_data['P_rect_02']
R_velo_to_cam = velo_to_cam_data['R']
T_velo_to_cam = velo_to_cam_data['T']

def transform_lidar_to_camera(lidar_points):
    # we only need (x, y, z) for each points
    xyz_points = lidar_points[:, :3]
    cam_points = (R_velo_to_cam @ xyz_points.T + T_velo_to_cam).T
    return cam_points

def project_to_image(points):
    # P_rect_02 is 3*4, we need homogenous coordinates
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_2s = (P_rect_02 @ points_hom.T).T
    points_2s = points_2s[:, :2] / points_2s[:, 2:]
    return points_2s

def overlay_lidar_on_image(image, pts_2d, depths):
    # adjust depth to range [0, 255]
    depth_min, depth_max = depths.min(), depths.max()
    depths_adjusted = ((depths - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    colors = cv2.applyColorMap(depths_adjusted.reshape(-1, 1), cv2.COLORMAP_JET).squeeze()

    for (x, y), color in zip(pts_2d.astype(int), colors):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 2, tuple(int(c) for c in color), -1)

    return image

@app.route('/bonus', methods=['GET'])
def bonus():
    frame = request.args.get('frame', type=int)
    if frame is None:
        return jsonify({"error": "Frame number required"}), 400

    image_file = os.path.join(IMAGE_PATH, f"{frame:010d}.png")
    lidar_file = os.path.join(LIDAR_PATH, f"{frame:010d}.bin")
    output_file = os.path.join(OUTPUT_PATH, f"{frame:010d}.png")

    if not os.path.exists(image_file) or not os.path.exists(lidar_file):
        return jsonify({"error": "Frame not found"}), 404

    lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    cam_points = transform_lidar_to_camera(lidar_points)
    cam_points = cam_points[cam_points[:, 2] > 0]
    img_points = project_to_image(cam_points)
    depths = cam_points[:, 2]

    image = cv2.imread(image_file)
    image_with_lidar = overlay_lidar_on_image(image, img_points, depths)
    cv2.imwrite(output_file, image_with_lidar)

    return jsonify({"output_image": output_file})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)