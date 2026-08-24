from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud

from navfusion.dataset.nuscenes_loader import NuScenesLoader

DATASET_ROOT = Path(
    r"C:\Users\Mettle\Documents\PythonProjects\NavFusion\Datasets\nuScenes"
)

#nuScenes sensor chanel names:
CAMERA_CHANNEL = "CAM_FRONT"
LIDAR_CHANNEL = "LIDAR_TOP"

# Only display LiDAR points within +/- 50 meters in x and y.
VISUALIZATION_RANGE_M = 50.0

def get_sensor_record_and_path(loader, sample, sensor_channel):
    if sensor_channel not in sample["data"]:
        raise KeyError(
            f"Sensor Channel '{sensor_channel}' is not present in sameple data."
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
    camera_record, camera_path = get_sensor_record_and_path(
        loader,
        sample,
        CAMERA_CHANNEL
    )

    lidar_record, lidar_path = get_sensor_record_and_path(
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
    
    lidar_point_cloud = LidarPointCloud.from_file(
        str(lidar_path)
    )

        # points is a NumPy array with shape (4, N):
    #
    # row 0 -> x
    # row 1 -> y
    # row 2 -> z
    # row 3 -> intensity
    #
    # N = number of LiDAR returns.

    points = lidar_point_cloud.points

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
        2,
        figsize=(16, 7)
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





