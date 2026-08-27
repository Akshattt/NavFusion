from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud

from navfusion.dataset.nuscenes_loader import NuScenesLoader
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points



DATASET_ROOT = Path(
    r"C:\Users\Mettle\Documents\PythonProjects\NavFusion\Datasets\nuScenes"
)

#nuScenes sensor chanel names:
CAMERA_CHANNEL = "CAM_FRONT"
LIDAR_CHANNEL = "LIDAR_TOP"

# Only display LiDAR points within +/- 50 meters in x and y.
VISUALIZATION_RANGE_M = 50.0

def getSensorRecordAndPath(loader, sample, sensor_channel):
    if sensor_channel not in sample["data"]:
        raise KeyError(
            f"Sensor Channel '{sensor_channel}' is not present in sample data."
        )

    sample_data_token = sample["data"][sensor_channel]

    sensor_record = loader.nusc.get(
        "sample_data",
        sample_data_token)

    sensor_path = loader.dataroot / sensor_record["filename"]

    if not sensor_path.is_file():
        raise FileNotFoundError(
            f"{sensor_channel} file not found: {sensor_path}"
        )

    return sensor_record, sensor_path


#from lidar to car

def lidarToCar(pointCloud, lidarCalibration):
    #transform lidar from to car coordinate frame

    lidarRotation = Quaternion(
        lidarCalibration["rotation"]
    ).rotation_matrix

    lidarTranslation = np.array(
        lidarCalibration["translation"]
    )

    pointCloud.rotate(lidarRotation)
    pointCloud.translate(lidarTranslation)

    return pointCloud

#car coordinates to global coordinates
def carToGlobal(pointCloud, carPose):
    carRotation = Quaternion(
        carPose["rotation"]
    ).rotation_matrix

    carTranslation = np.array(
        carPose["translation"]
    )

    pointCloud.rotate(carRotation)
    pointCloud.translate(carTranslation)

    return pointCloud

def globalToCar(pointCloud, carPose):
    #transform global frame to car coordinates frame.

    carTranslation = np.array(
        carPose["translation"]
    )

    carRotation = Quaternion(
        carPose["rotation"]
    ).rotation_matrix

    #to make the points negative

    pointCloud.translate(
        -carTranslation
    )

    # Undo the car's global rotation.
    # For a rotation matrix: inverse(R) = transpose(R)

    pointCloud.rotate(
        carRotation.T
    )

    return pointCloud

def carToCamera(pointCloud, cameraCalibration):
    #Transform car frame point into the camera coordinate frame"

    cameraTranslation = np.array(
        cameraCalibration["translation"]
    )

    cameraRotation = Quaternion(
        cameraCalibration["rotation"]
    ).rotation_matrix

    #cameraCalibration describes camera 
    #we need car, camera, so we apply the inverse transformation

    pointCloud.translate(
        -cameraTranslation
    )

    pointCloud.rotate(
        cameraRotation.T
    )

    return pointCloud


def cameraToImage(pointCloud, cameraCalibration):

    #here z is the depth of the camera

    depths = pointCloud.points[2,:]

    #nuScenes states the camera intinrsic matrix as a python list
    #convert it to a numpy array firsts

    cameraIntrinsic = np.array(
        cameraCalibration["camera_intrinsic"]
    )

    points2d = view_points(
        pointCloud.points[:3, :],
        cameraIntrinsic,
        normalize=True
    )

    return points2d, depths



# Run the complete LiDAR -> camera projection pipeline.
def projectLidarToCamera(
    loader,
    lidarPointCloud,
    lidarRecord,
    cameraRecord
):
    """Transform LiDAR points into camera pixels."""

    # Make a copy because rotate() and translate() modify the point cloud.
    pointCloud = LidarPointCloud(
        lidarPointCloud.points.copy()
    )

    # Get LiDAR mounting position/orientation relative to the car.
    lidarCalibration = loader.nusc.get(
        "calibrated_sensor",
        lidarRecord["calibrated_sensor_token"]
    )

    # Get the car pose when the LiDAR scan was captured.
    lidarCarPose = loader.nusc.get(
        "ego_pose",
        lidarRecord["ego_pose_token"]
    )

    # Get the car pose when the camera image was captured.
    cameraCarPose = loader.nusc.get(
        "ego_pose",
        cameraRecord["ego_pose_token"]
    )

    # Get camera mounting calibration and camera intrinsic matrix.
    cameraCalibration = loader.nusc.get(
        "calibrated_sensor",
        cameraRecord["calibrated_sensor_token"]
    )

    # LiDAR coordinates -> car coordinates at LiDAR timestamp.
    pointCloud = lidarToCar(
        pointCloud,
        lidarCalibration
    )

    # Car coordinates at LiDAR timestamp -> global coordinates.
    pointCloud = carToGlobal(
        pointCloud,
        lidarCarPose
    )

    # Global coordinates -> car coordinates at camera timestamp.
    pointCloud = globalToCar(
        pointCloud,
        cameraCarPose
    )

    # Car coordinates -> camera 3D coordinates.
    pointCloud = carToCamera(
        pointCloud,
        cameraCalibration
    )

    # Camera 3D coordinates -> image pixel coordinates.
    points2d, depths = cameraToImage(
        pointCloud,
        cameraCalibration
    )

    return points2d, depths
    










def main():
    """ Load and visualize the first sample from the first scene in the nuScenes dataset. """

    #to use the Navfusion dataset again and again.

    loader = NuScenesLoader(
        dataroot=DATASET_ROOT,
        version="v1.0-mini",
        verbose=True
    )

    #loader.nusc.scene is a python list of dictionaries, where each dictionary represents a scene in the nuScenes dataset.
    scene = loader.nusc.scene[0]

    first_sample_token = scene["first_sample_token"]

    sample = loader.nusc.get(
        "sample",
        first_sample_token
    )

    #now we have to get the camera metadata and the physical JPEG Path
    camera_record, camera_path = getSensorRecordAndPath(
        loader,
        sample,
        CAMERA_CHANNEL
    )

    lidar_record, lidar_path = getSensorRecordAndPath(
        loader,
        sample,
        LIDAR_CHANNEL
    )

        # ------------------------------------------------------------------
    # CAMERA DATA
    # ------------------------------------------------------------------

    # Read the JPEG and convert it to RGB.
    camera_image = Image.open(camera_path).convert("RGB")

        # ------------------------------------------------------------------
    # LIDAR DATA
    # ------------------------------------------------------------------

    # Read the binary LiDAR file using the official nuScenes SDK.
    
    lidar_pointCloud = LidarPointCloud.from_file(
        str(lidar_path)
    )

    points2d, depths = projectLidarToCamera(
        loader,
        lidar_pointCloud,
        lidar_record,
        camera_record
    )

    # Get the camera image dimensions.
    imageWidth, imageHeight = camera_image.size

    # Keep only projected LiDAR points that:
    # 1. are in front of the camera,
    # 2. fall inside the horizontal image boundary,
    # 3. fall inside the vertical image boundary.
    fusionMask = (
        (depths > 1.0)
        & (points2d[0, :] >= 0)
        & (points2d[0, :] < imageWidth)
        & (points2d[1, :] >= 0)
        & (points2d[1, :] < imageHeight)
    )

        # points is a NumPy array with shape (4, N):
    #
    # row 0 -> x
    # row 1 -> y
    # row 2 -> z
    # row 3 -> intensity
    #
    # N = number of LiDAR returns.

    points = lidar_pointCloud.points

    # Extract each row into a separate 1-D Numpy array.
    x = points[0, :]
    y = points[1, :]
    z = points[2, :]
    intensity = points[3, :]

    range_mask = (
        (x > -VISUALIZATION_RANGE_M) 
        &(y > -VISUALIZATION_RANGE_M)
        &(x < VISUALIZATION_RANGE_M)
        &(y < VISUALIZATION_RANGE_M) 
    )

    x_visible = x[range_mask]
    y_visible = y[range_mask]

    #For printing information

    print("Navfusion Sensor Visualization")
    print("================================")
    print(f"Scene Name: {scene['name']}")
    print(f"Sample Token: {sample['token']}")

    print("\nCAM_FRONT Sensor Record:")
    print("-------------------------")
    print(f"File             : {camera_path}")
    print(f"Image size       : {camera_image.size}")
    print(f"Timestamp        : {camera_record['timestamp']}")

    print("\nLIDAR_TOP Sensor Record:")
    print("-------------------------")
    print(f"File             : {lidar_path}")
    print(f"Point array      : {points.shape}")
    print(f"Number of points : {points.shape[1]}")
    print(f"Visible points   : {x_visible.shape[0]}")
    print(f"Timestamp        : {lidar_record['timestamp']}")
    print("\nLiDAR -> Camera Projection:")
    print("---------------------------")
    print(f"2D point array   : {points2d.shape}")
    print(f"Depth array      : {depths.shape}")
    print("First 5 projected points:")
    print(points2d[:, :5].T)
    print("First 5 depths:")
    print(depths[:5])
    print(f"Points inside image: {fusionMask.sum()}")

    print("Print 5 lidar points: ")
    print("[x y z intensity]")
    print(points[:, :5].T)

    print("\nCoordinate ranges:")
    print(f"x         : {x.min():.2f} to {x.max():.2f} m")
    print(f"y         : {y.min():.2f} to {y.max():.2f} m")
    print(f"z         : {z.min():.2f} to {z.max():.2f} m")
    print(f"intensity : {intensity.min():.2f} to {intensity.max():.2f}")

        # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(22, 7)
    )

    # Left side: front-camera image.
    axes[0].imshow(camera_image)
    axes[0].set_title("CAM_FRONT")
    axes[0].axis("off")

    # Right side: raw LiDAR viewed from above.
    #
    # Horizontal plotting axis = LiDAR y
    # Vertical plotting axis   = LiDAR x
    #
    # This makes vehicle-forward (+x) appear toward the top.
    axes[1].scatter(
        y_visible,
        x_visible,
        s=0.5
    )

    # In the LiDAR's own coordinate system, the sensor is at (0, 0).
    axes[1].scatter(
        0,
        0,
        marker="x",
        s=80,
        label="LiDAR sensor"
    )

    axes[1].set_title("LIDAR_TOP - Raw Sensor-Frame XY View")
    axes[1].set_xlabel("y - left/right (m)")
    axes[1].set_ylabel("x - forward/backward (m)")

    axes[1].set_xlim(
        -VISUALIZATION_RANGE_M,
        VISUALIZATION_RANGE_M
    )

    axes[1].set_ylim(
        -VISUALIZATION_RANGE_M,
        VISUALIZATION_RANGE_M
    )

    # One meter on the x-axis should visually equal one meter on the y-axis.
    axes[1].set_aspect("equal")

    axes[1].grid(True)
    axes[1].legend()

    # Third panel: projected LiDAR points over the camera image.
    axes[2].imshow(camera_image)

    fusionScatter = axes[2].scatter(
        points2d[0, fusionMask],
        points2d[1, fusionMask],
        c=depths[fusionMask],
        s=6,
        cmap="viridis"
    )

    axes[2].set_title("CAM_FRONT + LIDAR_TOP Fusion")
    axes[2].axis("off")

    # Show what the point colors mean.
    figure.colorbar(
        fusionScatter,
        ax=axes[2],
        label="Depth (m)"
    )

    figure.tight_layout()

    # ------------------------------------------------------------------
    # SAVE OUTPUT
    # ------------------------------------------------------------------

    # Current file:
    # NavFusion\Code\scripts\02_visualize_sensors.py
    #
    # parents[0] = scripts
    # parents[1] = Code
    # parents[2] = NavFusion
    project_root = Path(__file__).resolve().parents[2]

    output_directory = project_root / "Outputs"

    # Create Outputs if it does not already exist.
    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_directory
        / "scene_0_first_sample_sensor_visualization.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"\nVisualization saved to:\n{output_path}")

    # Keep the Matplotlib visualization window open.
    plt.show()


if __name__ == "__main__":
    main()





