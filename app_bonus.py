import os
import numpy as np
import cv2
import struct
from flask import Flask, request, jsonify

# Flask app
app = Flask(__name__)

# Paths to KITTI dataset folders (modify as needed)
BASE_PATH = "data_bonus"
CALIB_PATH = os.path.join(BASE_PATH, "calib/")
LIDAR_PATH = os.path.join(BASE_PATH, "lidar/data/")
IMAGE_PATH = os.path.join(BASE_PATH, "image_02/data/")
OUTPUT_PATH = os.path.join(BASE_PATH, "output/")

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_calib(file_path):
    """Load KITTI calibration file into a dictionary."""
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

# Convert LiDAR points to camera coordinates
def transform_lidar_to_camera(velo_points):
    """Transform LiDAR points to the camera coordinate system."""
    lidar_hom = np.hstack((velo_points[:, :3], np.ones((velo_points.shape[0], 1))))
    cam_points = (R_velo_to_cam @ lidar_hom[:, :3].T + T_velo_to_cam).T
    return cam_points

# Project 3D points onto 2D image
def project_to_image(pts_3d):
    """Project 3D points into 2D image space."""
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    pts_2d = (P_rect_02 @ pts_3d_hom.T).T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]  # Normalize
    return pts_2d

# Read LiDAR binary file
def read_LiDAR(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

def read_image(file_path):
    return cv2.imread(file_path)

def overlay_lidar_on_image(image, pts_2d, depths):
    """Overlay projected LiDAR points onto the image with color based on depth."""
    # Normalize depth values to range 0-255
    depth_min, depth_max = depths.min(), depths.max()
    depths_normalized = ((depths - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    # Apply colormap (JET: blue = close, red = far)
    colors = cv2.applyColorMap(depths_normalized.reshape(-1, 1), cv2.COLORMAP_JET).squeeze()

    # Draw points with color
    for (x, y), color in zip(pts_2d.astype(int), colors):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 2, tuple(int(c) for c in color), -1)  # Color based on depth

    return image

def overlay_lidar():
    """API endpoint to overlay LiDAR points onto a camera image."""
    # frame = request.args.get('frame', type=int)
    frame = 2
    # if frame is None:
    #     return jsonify({"error": "Frame number required"}), 400

    # File paths
    image_file = os.path.join(IMAGE_PATH, f"{frame:010d}.png")
    lidar_file = os.path.join(LIDAR_PATH, f"{frame:010d}.bin")
    output_file = os.path.join(OUTPUT_PATH, f"overlay_{frame:010d}.png")

    print(image_file)

    # Check if files exist
    # if not os.path.exists(image_file) or not os.path.exists(lidar_file):
    #     return jsonify({"error": "Frame not found"}), 404

    # Load data
    lidar_points = read_LiDAR(lidar_file)
    cam_points = transform_lidar_to_camera(lidar_points)
    cam_points = cam_points[cam_points[:, 2] > 0]  # Remove points behind camera
    img_points = project_to_image(cam_points)

    depths = cam_points[:, 2]
    print(cam_points)

    # Read image and overlay points
    image = read_image(image_file)
    image_with_lidar = overlay_lidar_on_image(image, img_points, depths)

    # Save and return output image path
    cv2.imwrite(output_file, image_with_lidar)
    # return jsonify({"output_image": output_file})
    return str(output_file)

print(overlay_lidar())

# @app.route('/overlay_lidar', methods=['GET'])


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
