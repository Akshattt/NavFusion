from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from ultralytics import YOLO
import numpy as np

from navfusion.dataset.nuscenes_loader import NuScenesLoader


DATASET_ROOT = Path(
    r"C:\Users\Mettle\Documents\PythonProjects\NavFusion\Datasets\nuScenes"
)

# nuScenes sensor channel names.
CAMERA_CHANNEL = "CAM_FRONT"
LIDAR_CHANNEL = "LIDAR_TOP"

# Only display raw LiDAR points within +/- 50 metres.
VISUALIZATION_RANGE_M = 50.0

# --------------------------------------------------------------
# YOLO CONFIGURATION
# --------------------------------------------------------------

# YOLO11 nano is small and fast enough for our first implementation.
YOLO_MODEL_NAME = "yolo11n.pt"

# Ignore weak detections below this confidence.
YOLO_CONFIDENCE = 0.35

# Vehicle classes from the standard COCO labels used by YOLO.
VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}

# Shrink each YOLO bounding box before associating LiDAR points.
#
# The previous Visual Fusion project showed why:
# LiDAR points near the edges of a rectangular detection box may
# actually belong to the background or another nearby object.
#
# 0.10 means remove 10% from EACH side.
BOUNDING_BOX_SHRINK_FACTOR = 0.10

# Require at least this many LiDAR points before trusting an
# object-distance estimate.
MIN_LIDAR_POINTS_PER_OBJECT = 3

# MAD threshold used for robust depth-outlier rejection.
MAD_SCALE = 3.0


def getSensorRecordAndPath(
    loader,
    sample,
    sensorChannel
):
    """Return sensor metadata and its physical data-file path."""

    if sensorChannel not in sample["data"]:
        raise KeyError(
            f"Sensor channel '{sensorChannel}' "
            f"is not present in sample data."
        )

    sampleDataToken = sample["data"][sensorChannel]

    sensorRecord = loader.nusc.get(
        "sample_data",
        sampleDataToken
    )

    sensorPath = (
        loader.dataroot
        / sensorRecord["filename"]
    )

    if not sensorPath.is_file():
        raise FileNotFoundError(
            f"{sensorChannel} file not found: "
            f"{sensorPath}"
        )

    return sensorRecord, sensorPath


# --------------------------------------------------------------
# LIDAR -> CAMERA GEOMETRIC TRANSFORMATIONS
# --------------------------------------------------------------

def lidarToCar(
    pointCloud,
    lidarCalibration
):
    """Transform LiDAR-frame points into the car coordinate frame."""

    lidarRotation = Quaternion(
        lidarCalibration["rotation"]
    ).rotation_matrix

    lidarTranslation = np.array(
        lidarCalibration["translation"]
    )

    pointCloud.rotate(
        lidarRotation
    )

    pointCloud.translate(
        lidarTranslation
    )

    return pointCloud


def carToGlobal(
    pointCloud,
    carPose
):
    """Transform car-frame points into the global coordinate frame."""

    carRotation = Quaternion(
        carPose["rotation"]
    ).rotation_matrix

    carTranslation = np.array(
        carPose["translation"]
    )

    pointCloud.rotate(
        carRotation
    )

    pointCloud.translate(
        carTranslation
    )

    return pointCloud


def globalToCar(
    pointCloud,
    carPose
):
    """Transform global-frame points into a car coordinate frame."""

    carTranslation = np.array(
        carPose["translation"]
    )

    carRotation = Quaternion(
        carPose["rotation"]
    ).rotation_matrix

    # Undo the car's global translation.
    pointCloud.translate(
        -carTranslation
    )

    # Undo its global rotation.
    #
    # For a rotation matrix:
    #
    # inverse(R) = transpose(R)
    pointCloud.rotate(
        carRotation.T
    )

    return pointCloud


def carToCamera(
    pointCloud,
    cameraCalibration
):
    """Transform car-frame points into the camera coordinate frame."""

    cameraTranslation = np.array(
        cameraCalibration["translation"]
    )

    cameraRotation = Quaternion(
        cameraCalibration["rotation"]
    ).rotation_matrix

    # cameraCalibration describes camera -> car.
    #
    # We need car -> camera, so apply the inverse.
    pointCloud.translate(
        -cameraTranslation
    )

    pointCloud.rotate(
        cameraRotation.T
    )

    return pointCloud


def cameraToImage(
    pointCloud,
    cameraCalibration
):
    """Project camera-frame 3D points onto the 2D image plane."""

    # In the camera coordinate frame, z is depth.
    depths = pointCloud.points[2, :]

    cameraIntrinsic = np.array(
        cameraCalibration["camera_intrinsic"]
    )

    points2d = view_points(
        pointCloud.points[:3, :],
        cameraIntrinsic,
        normalize=True
    )

    return points2d, depths


def projectLidarToCamera(
    loader,
    lidarPointCloud,
    lidarRecord,
    cameraRecord
):
    """Run the complete LiDAR -> camera projection pipeline."""

    # rotate() and translate() modify the object in place,
    # so use a copy.
    pointCloud = LidarPointCloud(
        lidarPointCloud.points.copy()
    )

    lidarCalibration = loader.nusc.get(
        "calibrated_sensor",
        lidarRecord["calibrated_sensor_token"]
    )

    lidarCarPose = loader.nusc.get(
        "ego_pose",
        lidarRecord["ego_pose_token"]
    )

    cameraCarPose = loader.nusc.get(
        "ego_pose",
        cameraRecord["ego_pose_token"]
    )

    cameraCalibration = loader.nusc.get(
        "calibrated_sensor",
        cameraRecord["calibrated_sensor_token"]
    )

    # LiDAR -> car at LiDAR time.
    pointCloud = lidarToCar(
        pointCloud,
        lidarCalibration
    )

    # Car at LiDAR time -> global.
    pointCloud = carToGlobal(
        pointCloud,
        lidarCarPose
    )

    # Global -> car at camera time.
    pointCloud = globalToCar(
        pointCloud,
        cameraCarPose
    )

    # Car -> camera.
    pointCloud = carToCamera(
        pointCloud,
        cameraCalibration
    )

    # Camera 3D -> image pixels.
    points2d, depths = cameraToImage(
        pointCloud,
        cameraCalibration
    )

    return points2d, depths


# --------------------------------------------------------------
# YOLO OBJECT DETECTION
# --------------------------------------------------------------

def runVehicleDetection(
    model,
    cameraImage
):
    """Detect vehicles in the front-camera image with YOLO11."""

    # Convert PIL RGB image into a NumPy RGB image.
    imageArray = np.array(
        cameraImage
    )

    results = model.predict(
        source=imageArray,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        classId = int(
            box.cls[0].item()
        )

        className = result.names[
            classId
        ]

        # For now only keep vehicle classes.
        if className not in VEHICLE_CLASSES:
            continue

        confidence = float(
            box.conf[0].item()
        )

        # xyxy gives:
        #
        # [left, top, right, bottom]
        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        detection = {
            "className": className,
            "confidence": confidence,
            "box": np.array(
                [x1, y1, x2, y2],
                dtype=np.float32
            )
        }

        detections.append(
            detection
        )

    return detections


# --------------------------------------------------------------
# BOUNDING-BOX SHRINKING
# --------------------------------------------------------------

def shrinkBoundingBox(
    box,
    shrinkFactor
):
    """
    Shrink a detection box inward from all four sides.

    This reduces the chance that LiDAR points near the outer
    boundary actually belong to the background.
    """

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    xShrink = (
        width
        * shrinkFactor
    )

    yShrink = (
        height
        * shrinkFactor
    )

    shrunkBox = np.array(
        [
            x1 + xShrink,
            y1 + yShrink,
            x2 - xShrink,
            y2 - yShrink
        ],
        dtype=np.float32
    )

    return shrunkBox


# --------------------------------------------------------------
# ROBUST DEPTH OUTLIER REMOVAL
# --------------------------------------------------------------

def filterDepthsWithMad(
    depths
):
    """
    Remove depth outliers using Median Absolute Deviation (MAD).

    MAD is more robust than mean/std when a YOLO box contains
    a few background LiDAR points.
    """

    if depths.size == 0:
        return np.zeros(
            0,
            dtype=bool
        )

    medianDepth = np.median(
        depths
    )

    absoluteDeviation = np.abs(
        depths - medianDepth
    )

    mad = np.median(
        absoluteDeviation
    )

    # If MAD is exactly zero, most depths are identical or nearly
    # identical. Do not reject them unnecessarily.
    if mad < 1e-6:
        return np.ones(
            depths.shape,
            dtype=bool
        )

    # 1.4826 makes MAD comparable to standard deviation for a
    # normally distributed variable.
    robustSigma = (
        1.4826
        * mad
    )

    inlierMask = (
        absoluteDeviation
        <= MAD_SCALE * robustSigma
    )

    return inlierMask


# --------------------------------------------------------------
# ASSOCIATE LIDAR WITH YOLO VEHICLES
# --------------------------------------------------------------

def associateLidarWithVehicles(
    detections,
    points2d,
    depths,
    fusionMask,
    originalLidarPoints
):
    """
    Associate projected LiDAR returns with each YOLO vehicle.

    The same point index is preserved between:
        originalLidarPoints[:, i]
        points2d[:, i]
        depths[i]

    Therefore a LiDAR point found inside a YOLO bounding box
    can also be located metrically in LiDAR x/y/z.
    """

    u = points2d[0, :]
    v = points2d[1, :]

    fusedObjects = []

    for detection in detections:

        shrunkBox = shrinkBoundingBox(
            detection["box"],
            BOUNDING_BOX_SHRINK_FACTOR
        )

        x1, y1, x2, y2 = shrunkBox

        # A LiDAR point belongs to this vehicle candidate when:
        #
        # 1. It survived the camera FOV mask.
        # 2. Its projected u coordinate lies inside the box.
        # 3. Its projected v coordinate lies inside the box.
        objectMask = (
            fusionMask
            & (u >= x1)
            & (u <= x2)
            & (v >= y1)
            & (v <= y2)
        )

        objectIndices = np.flatnonzero(
            objectMask
        )

        if (
            objectIndices.size
            < MIN_LIDAR_POINTS_PER_OBJECT
        ):
            continue

        objectDepths = depths[
            objectIndices
        ]

        # Remove depth outliers.
        depthInlierMask = filterDepthsWithMad(
            objectDepths
        )

        cleanIndices = objectIndices[
            depthInlierMask
        ]

        cleanDepths = depths[
            cleanIndices
        ]

        if (
            cleanIndices.size
            < MIN_LIDAR_POINTS_PER_OBJECT
        ):
            continue

        # ----------------------------------------------------------
        # OBJECT DISTANCE
        # ----------------------------------------------------------

        # Use the median rather than mean so a few remaining
        # unusual returns cannot pull the object distance strongly.
        distanceM = float(
            np.median(
                cleanDepths
            )
        )

        # ----------------------------------------------------------
        # ORIGINAL LIDAR XYZ POSITION
        # ----------------------------------------------------------

        # originalLidarPoints still has the same column ordering
        # as the projected points.
        objectLidarPoints = originalLidarPoints[
            :3,
            cleanIndices
        ]

        # Median x/y/z provides one representative object location.
        lidarXM = float(
            np.median(
                objectLidarPoints[0, :]
            )
        )

        lidarYM = float(
            np.median(
                objectLidarPoints[1, :]
            )
        )

        lidarZM = float(
            np.median(
                objectLidarPoints[2, :]
            )
        )

        fusedObject = {
            "className": detection["className"],
            "confidence": detection["confidence"],
            "box": detection["box"],
            "shrunkBox": shrunkBox,
            "distanceM": distanceM,
            "lidarXM": lidarXM,
            "lidarYM": lidarYM,
            "lidarZM": lidarZM,
            "rawLidarCount": objectIndices.size,
            "cleanLidarCount": cleanIndices.size,
            "cleanIndices": cleanIndices
        }

        fusedObjects.append(
            fusedObject
        )

    return fusedObjects


# --------------------------------------------------------------
# DRAW YOLO + LIDAR FUSION
# --------------------------------------------------------------

def drawVehicleFusion(
    axis,
    cameraImage,
    points2d,
    depths,
    fusionMask,
    fusedObjects
):
    """Draw YOLO boxes, LiDAR returns, and vehicle distances."""

    axis.imshow(
        cameraImage
    )

    # Show all camera-visible LiDAR returns faintly.
    axis.scatter(
        points2d[0, fusionMask],
        points2d[1, fusionMask],
        c=depths[fusionMask],
        s=3,
        cmap="viridis",
        alpha=0.35
    )

    for fusedObject in fusedObjects:

        x1, y1, x2, y2 = fusedObject[
            "box"
        ]

        width = x2 - x1
        height = y2 - y1

        rectangle = plt.Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2
        )

        axis.add_patch(
            rectangle
        )

        # Highlight only the robust LiDAR points associated with
        # this particular vehicle.
        cleanIndices = fusedObject[
            "cleanIndices"
        ]

        axis.scatter(
            points2d[0, cleanIndices],
            points2d[1, cleanIndices],
            s=12
        )

        label = (
            f"{fusedObject['className']} "
            f"{fusedObject['confidence']:.2f}\n"
            f"{fusedObject['distanceM']:.2f} m"
        )

        axis.text(
            x1,
            max(
                15,
                y1 - 8
            ),
            label,
            fontsize=9,
            bbox={
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none"
            }
        )

    axis.set_title(
        "YOLO11 + LiDAR Vehicle Distance"
    )

    axis.axis(
        "off"
    )


def main():
    """Load and visualize the first sample from the first scene."""

    # --------------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------------

    loader = NuScenesLoader(
        dataroot=DATASET_ROOT,
        version="v1.0-mini",
        verbose=True
    )

    scene = loader.nusc.scene[0]

    firstSampleToken = scene[
        "first_sample_token"
    ]

    sample = loader.nusc.get(
        "sample",
        firstSampleToken
    )

    cameraRecord, cameraPath = getSensorRecordAndPath(
        loader,
        sample,
        CAMERA_CHANNEL
    )

    lidarRecord, lidarPath = getSensorRecordAndPath(
        loader,
        sample,
        LIDAR_CHANNEL
    )

    # --------------------------------------------------------------
    # CAMERA DATA
    # --------------------------------------------------------------

    cameraImage = Image.open(
        cameraPath
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------------
    # LIDAR DATA
    # --------------------------------------------------------------

    lidarPointCloud = LidarPointCloud.from_file(
        str(lidarPath)
    )

    # Preserve the original LiDAR points before any geometric
    # projection transformations.
    originalLidarPoints = (
        lidarPointCloud.points.copy()
    )

    points2d, depths = projectLidarToCamera(
        loader,
        lidarPointCloud,
        lidarRecord,
        cameraRecord
    )

    imageWidth, imageHeight = (
        cameraImage.size
    )

    # --------------------------------------------------------------
    # CAMERA FOV MASK
    # --------------------------------------------------------------

    fusionMask = (
        (depths > 1.0)
        & (points2d[0, :] >= 0)
        & (points2d[0, :] < imageWidth)
        & (points2d[1, :] >= 0)
        & (points2d[1, :] < imageHeight)
    )

    # --------------------------------------------------------------
    # RAW LIDAR DATA
    # --------------------------------------------------------------

    points = lidarPointCloud.points

    x = points[0, :]
    y = points[1, :]
    z = points[2, :]
    intensity = points[3, :]

    rangeMask = (
        (x > -VISUALIZATION_RANGE_M)
        & (y > -VISUALIZATION_RANGE_M)
        & (x < VISUALIZATION_RANGE_M)
        & (y < VISUALIZATION_RANGE_M)
    )

    xVisible = x[
        rangeMask
    ]

    yVisible = y[
        rangeMask
    ]

    # --------------------------------------------------------------
    # YOLO11 DETECTION
    # --------------------------------------------------------------

    print("\nLoading YOLO11 model...")

    yoloModel = YOLO(
        YOLO_MODEL_NAME
    )

    detections = runVehicleDetection(
        yoloModel,
        cameraImage
    )

    # --------------------------------------------------------------
    # LIDAR + YOLO ASSOCIATION
    # --------------------------------------------------------------

    fusedObjects = associateLidarWithVehicles(
        detections,
        points2d,
        depths,
        fusionMask,
        originalLidarPoints
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------------

    print("\nNavFusion Sensor Visualization")
    print("================================")

    print(
        f"Scene Name       : "
        f"{scene['name']}"
    )

    print(
        f"Sample Token     : "
        f"{sample['token']}"
    )

    print("\nCAM_FRONT Sensor Record:")
    print("-------------------------")

    print(
        f"File             : "
        f"{cameraPath}"
    )

    print(
        f"Image size       : "
        f"{cameraImage.size}"
    )

    print(
        f"Timestamp        : "
        f"{cameraRecord['timestamp']}"
    )

    print("\nLIDAR_TOP Sensor Record:")
    print("-------------------------")

    print(
        f"File             : "
        f"{lidarPath}"
    )

    print(
        f"Point array      : "
        f"{points.shape}"
    )

    print(
        f"Number of points : "
        f"{points.shape[1]}"
    )

    print(
        f"Visible points   : "
        f"{xVisible.shape[0]}"
    )

    print(
        f"Timestamp        : "
        f"{lidarRecord['timestamp']}"
    )

    print("\nLiDAR -> Camera Projection:")
    print("---------------------------")

    print(
        f"2D point array   : "
        f"{points2d.shape}"
    )

    print(
        f"Depth array      : "
        f"{depths.shape}"
    )

    print(
        f"Points inside image: "
        f"{fusionMask.sum()}"
    )

    print("\nYOLO11 Vehicle Detection:")
    print("-------------------------")

    print(
        f"Vehicle detections : "
        f"{len(detections)}"
    )

    print(
        f"Fused vehicles     : "
        f"{len(fusedObjects)}"
    )

    for objectNumber, fusedObject in enumerate(
        fusedObjects,
        start=1
    ):

        print(
            f"\nObject {objectNumber}:"
        )

        print(
            f"Class             : "
            f"{fusedObject['className']}"
        )

        print(
            f"Confidence        : "
            f"{fusedObject['confidence']:.3f}"
        )

        print(
            f"Distance          : "
            f"{fusedObject['distanceM']:.2f} m"
        )

        print(
            f"LiDAR position    : "
            f"x={fusedObject['lidarXM']:.2f}, "
            f"y={fusedObject['lidarYM']:.2f}, "
            f"z={fusedObject['lidarZM']:.2f} m"
        )

        print(
            f"LiDAR points      : "
            f"{fusedObject['rawLidarCount']} raw -> "
            f"{fusedObject['cleanLidarCount']} filtered"
        )

    # --------------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------------

    # Keep the previous stages visible rather than replacing them.
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(28, 7)
    )

    # --------------------------------------------------------------
    # PANEL 1: CAMERA
    # --------------------------------------------------------------

    axes[0].imshow(
        cameraImage
    )

    axes[0].set_title(
        "CAM_FRONT"
    )

    axes[0].axis(
        "off"
    )

    # --------------------------------------------------------------
    # PANEL 2: RAW LIDAR
    # --------------------------------------------------------------

    axes[1].scatter(
        yVisible,
        xVisible,
        s=0.5
    )

    axes[1].scatter(
        0,
        0,
        marker="x",
        s=80,
        label="LiDAR sensor"
    )

    axes[1].set_title(
        "LIDAR_TOP - Raw Sensor-Frame XY"
    )

    axes[1].set_xlabel(
        "y - left/right (m)"
    )

    axes[1].set_ylabel(
        "x - forward/backward (m)"
    )

    axes[1].set_xlim(
        -VISUALIZATION_RANGE_M,
        VISUALIZATION_RANGE_M
    )

    axes[1].set_ylim(
        -VISUALIZATION_RANGE_M,
        VISUALIZATION_RANGE_M
    )

    axes[1].set_aspect(
        "equal"
    )

    axes[1].grid(
        True
    )

    axes[1].legend()

    # --------------------------------------------------------------
    # PANEL 3: ORIGINAL LIDAR-CAMERA PROJECTION
    # --------------------------------------------------------------

    axes[2].imshow(
        cameraImage
    )

    fusionScatter = axes[2].scatter(
        points2d[0, fusionMask],
        points2d[1, fusionMask],
        c=depths[fusionMask],
        s=6,
        cmap="viridis"
    )

    axes[2].set_title(
        "CAM_FRONT + LIDAR_TOP Fusion"
    )

    axes[2].axis(
        "off"
    )

    figure.colorbar(
        fusionScatter,
        ax=axes[2],
        label="Depth (m)"
    )

    # --------------------------------------------------------------
    # PANEL 4: YOLO + LIDAR DISTANCE
    # --------------------------------------------------------------

    drawVehicleFusion(
        axes[3],
        cameraImage,
        points2d,
        depths,
        fusionMask,
        fusedObjects
    )

    figure.tight_layout()

    # --------------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------------

    projectRoot = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    outputDirectory = (
        projectRoot
        / "Outputs"
    )

    outputDirectory.mkdir(
        parents=True,
        exist_ok=True
    )

    outputPath = (
        outputDirectory
        / "scene_0_first_sample_yolo_lidar_fusion.png"
    )

    figure.savefig(
        outputPath,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"\nVisualization saved to:\n"
        f"{outputPath}"
    )

    plt.show()


if __name__ == "__main__":
    main()