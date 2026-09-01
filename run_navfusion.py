from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud
from ultralytics import YOLO

from navfusion.dataset.nuscenes_loader import NuScenesLoader
from navfusion.perception.sensor_fusion import processSensorFusion
from navfusion.mapping.lidar_bev import processLidarBev
from navfusion.visualization.sensor_fusion_visualizer import (
    createSensorFusionFigure
)


# ==============================================================
# Project paths
# ==============================================================

# This file is located at:
#
# NavFusion\Code\run_navfusion.py
#
# .parent        -> Code
# .parent.parent -> NavFusion
projectRoot = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

datasetRoot = (
    projectRoot
    / "Datasets"
    / "nuScenes"
)

outputDirectory = (
    projectRoot
    / "Outputs"
)


# ==============================================================
# Dataset configuration
# ==============================================================

datasetVersion = "v1.0-mini"


# ==============================================================
# Data selection
# ==============================================================

# sceneIndex controls which scenes enter the pipeline.
#
# sceneIndex = 0
#     Process only the first scene.
#
# sceneIndex = 3
#     Process only scene index 3.
#
# sceneIndex = None
#     Process every scene in the dataset.
#
# For current verification:
sceneIndex = 0


# maxSamples is a global sample limit.
#
# maxSamples = 1
#     Process one sample total.
#
# maxSamples = 20
#     Process twenty samples total.
#
# maxSamples = None
#     Do not stop early.
#
# Entire dataset:
#
# sceneIndex = None
# maxSamples = None
#
# For current verification:
maxSamples = 1


# These strings are exact nuScenes channel names.
cameraChannel = "CAM_FRONT"

lidarChannel = "LIDAR_TOP"


# ==============================================================
# YOLO configuration
# ==============================================================

yoloModelName = "yolo11n.pt"

sensorFusionConfig = {
    "yoloConfidence": 0.35,

    "vehicleClasses": {
        "car",
        "truck",
        "bus",
        "motorcycle"
    },

    # Shrink 10% from each side of the detection box before
    # associating projected LiDAR points.
    "boundingBoxShrinkFactor": 0.10,

    "minimumLidarPoints": 3,

    "madScale": 3.0
}


# ==============================================================
# LiDAR BEV / RANSAC configuration
# ==============================================================

bevConfig = {
    # ----------------------------------------------------------
    # Physical BEV region
    # ----------------------------------------------------------

    # x:
    #     -20 m behind
    #      50 m forward
    "xMinM": -20.0,
    "xMaxM": 50.0,

    # y:
    #     -25 m to +25 m laterally
    "yMinM": -25.0,
    "yMaxM": 25.0,

    # Each BEV cell represents:
    #
    # 0.20 m x 0.20 m
    "resolutionM": 0.20,

    # ----------------------------------------------------------
    # Ground-candidate selection
    # ----------------------------------------------------------

    # Select the lower portion of the LiDAR z distribution as
    # possible ground points before RANSAC.
    "groundLowerPercentile": 5.0,

    "groundUpperPercentile": 35.0,

    # ----------------------------------------------------------
    # RANSAC configuration
    # ----------------------------------------------------------

    # Number of random plane hypotheses.
    "ransacIterations": 500,

    # A ground candidate is a RANSAC inlier if its perpendicular
    # distance from the proposed plane is <= 10 cm.
    "ransacDistanceThresholdM": 0.10,

    # Fixed random seed makes development runs reproducible.
    "ransacRandomSeed": 42,

    # ----------------------------------------------------------
    # Obstacle configuration
    # ----------------------------------------------------------

    # Points more than 15 cm above the estimated ground surface
    # become initial obstacle candidates.
    "minimumObstacleHeightM": 0.15,

    # ----------------------------------------------------------
    # Car / sensor self-return region
    # ----------------------------------------------------------

    # These coordinates are in the raw LIDAR_TOP sensor frame:
    #
    # +x = forward
    # +y = left
    # +z = up
    #
    # This region is intentionally small. It removes the dense
    # structure previously observed immediately around LIDAR_TOP
    # without masking a large arbitrary area around the car.
    "selfReturnXMinM": -0.8,
    "selfReturnXMaxM": 0.8,

    "selfReturnYMinM": -1.3,
    "selfReturnYMaxM": 1.9,

    "selfReturnZMinM": -1.2,
    "selfReturnZMaxM": 0.2
}


# ==============================================================
# Visualization configuration
# ==============================================================

visualizationRangeM = 50.0

saveImages = True

# Keep True while maxSamples = 1.
#
# For a whole-scene or whole-dataset run:
#
# showFigure = False
#
# Otherwise plt.show() blocks after every sample.
showFigure = True


# ==============================================================
# Get one selected sensor record
# ==============================================================

def getSensorRecordAndPath(
    loader,
    sample,
    sensorChannel
):
    """
    Get the sample_data record and physical file path for the
    sensor selected by this runner.

    Dataset access happens here rather than inside the perception
    or mapping algorithms.
    """

    if sensorChannel not in sample["data"]:
        raise KeyError(
            f"Sensor channel '{sensorChannel}' "
            f"is not available in this sample."
        )

    # sample["data"] maps sensor-channel names to sample_data
    # tokens.
    sampleDataToken = sample[
        "data"
    ][
        sensorChannel
    ]

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


# ==============================================================
# Load all source data for one selected sample
# ==============================================================

def loadFrameData(
    loader,
    sample
):
    """
    Load every source-data object required for one sample.

    This is the boundary between:

        dataset access

    and:

        perception / mapping algorithms.

    The algorithms receive frameData and therefore process only
    the exact data selected here.
    """

    # ----------------------------------------------------------
    # Select camera record
    # ----------------------------------------------------------

    cameraRecord, cameraPath = getSensorRecordAndPath(
        loader,
        sample,
        cameraChannel
    )

    # ----------------------------------------------------------
    # Select LiDAR record
    # ----------------------------------------------------------

    lidarRecord, lidarPath = getSensorRecordAndPath(
        loader,
        sample,
        lidarChannel
    )

    # ----------------------------------------------------------
    # Load camera image
    # ----------------------------------------------------------

    cameraImage = Image.open(
        cameraPath
    ).convert(
        "RGB"
    )

    # ----------------------------------------------------------
    # Load LiDAR point cloud
    # ----------------------------------------------------------

    lidarPointCloud = LidarPointCloud.from_file(
        str(
            lidarPath
        )
    )

    # ----------------------------------------------------------
    # Load LiDAR calibration
    # ----------------------------------------------------------

    lidarCalibration = loader.nusc.get(
        "calibrated_sensor",
        lidarRecord[
            "calibrated_sensor_token"
        ]
    )

    # ----------------------------------------------------------
    # Load camera calibration
    # ----------------------------------------------------------

    cameraCalibration = loader.nusc.get(
        "calibrated_sensor",
        cameraRecord[
            "calibrated_sensor_token"
        ]
    )

    # ----------------------------------------------------------
    # Load car pose at LiDAR timestamp
    # ----------------------------------------------------------

    # "ego_pose" is an exact nuScenes schema/table name.
    lidarCarPose = loader.nusc.get(
        "ego_pose",
        lidarRecord[
            "ego_pose_token"
        ]
    )

    # ----------------------------------------------------------
    # Load car pose at camera timestamp
    # ----------------------------------------------------------

    cameraCarPose = loader.nusc.get(
        "ego_pose",
        cameraRecord[
            "ego_pose_token"
        ]
    )

    # ----------------------------------------------------------
    # Package the exact data for this frame
    # ----------------------------------------------------------

    frameData = {
        "sample": sample,

        "cameraChannel": cameraChannel,
        "lidarChannel": lidarChannel,

        "cameraRecord": cameraRecord,
        "lidarRecord": lidarRecord,

        "cameraPath": cameraPath,
        "lidarPath": lidarPath,

        "cameraImage": cameraImage,
        "lidarPointCloud": lidarPointCloud,

        "cameraCalibration": cameraCalibration,
        "lidarCalibration": lidarCalibration,

        "cameraCarPose": cameraCarPose,
        "lidarCarPose": lidarCarPose
    }

    return frameData


# ==============================================================
# Print sensor-fusion results
# ==============================================================

def printSensorFusionSummary(
    scene,
    sampleIndex,
    sensorResult
):
    """
    Print camera, LiDAR, YOLO and LiDAR-object fusion results for
    one synchronized sample.
    """

    print()
    print("=" * 70)

    print(
        f"Scene        : "
        f"{scene['name']}"
    )

    print(
        f"Sample index : "
        f"{sampleIndex}"
    )

    print(
        f"Sample token : "
        f"{sensorResult['sample']['token']}"
    )

    print("=" * 70)

    print(
        f"Camera       : "
        f"{sensorResult['cameraChannel']}"
    )

    print(
        f"LiDAR        : "
        f"{sensorResult['lidarChannel']}"
    )

    print(
        f"Image size   : "
        f"{sensorResult['cameraImage'].size}"
    )

    print(
        f"Camera time  : "
        f"{sensorResult['cameraRecord']['timestamp']}"
    )

    print(
        f"LiDAR time   : "
        f"{sensorResult['lidarRecord']['timestamp']}"
    )

    print(
        f"LiDAR points : "
        f"{sensorResult['lidarPointCloud'].points.shape[1]}"
    )

    print(
        f"2D points    : "
        f"{sensorResult['points2d'].shape}"
    )

    print(
        f"Depth array  : "
        f"{sensorResult['depths'].shape}"
    )

    print(
        f"In camera    : "
        f"{sensorResult['fusionMask'].sum()}"
    )

    print(
        f"YOLO vehicles: "
        f"{len(sensorResult['detections'])}"
    )

    print(
        f"Fused objects: "
        f"{len(sensorResult['fusedObjects'])}"
    )

    for objectNumber, fusedObject in enumerate(
        sensorResult[
            "fusedObjects"
        ],
        start=1
    ):

        print()

        print(
            f"Object {objectNumber}:"
        )

        print(
            f"  Class        : "
            f"{fusedObject['className']}"
        )

        print(
            f"  Confidence   : "
            f"{fusedObject['confidence']:.3f}"
        )

        print(
            f"  Camera depth : "
            f"{fusedObject['distanceM']:.2f} m"
        )

        print(
            f"  LiDAR xyz    : "
            f"x={fusedObject['lidarXM']:.2f}, "
            f"y={fusedObject['lidarYM']:.2f}, "
            f"z={fusedObject['lidarZM']:.2f} m"
        )

        print(
            f"  LiDAR points : "
            f"{fusedObject['rawLidarCount']} raw -> "
            f"{fusedObject['cleanLidarCount']} filtered"
        )


# ==============================================================
# Print RANSAC / BEV results
# ==============================================================

def printLidarBevSummary(
    bevResult
):
    """
    Print RANSAC ground estimation and self-return filtering
    diagnostics for one LiDAR sample.
    """

    groundA, groundB, groundC = bevResult[
        "planeCoefficients"
    ]

    # Some points inside the self-return box may already have been
    # classified as ground/non-obstacles.
    #
    # The exact number actually removed from the obstacle layer is
    # the intersection:
    #
    # obstacleCandidateMask AND selfReturnMask
    removedSelfObstacleMask = (
        bevResult[
            "obstacleCandidateMask"
        ]
        & bevResult[
            "selfReturnMask"
        ]
    )

    removedSelfObstacleCount = int(
        removedSelfObstacleMask.sum()
    )

    lidarSensorX, lidarSensorY, lidarSensorZ = bevResult[
        "lidarSensorPositionCarM"
    ]

    print()
    print("LiDAR -> car calibration:")
    print("-------------------------")

    print(
        f"LiDAR position in car frame: "
        f"x={lidarSensorX:.3f}, "
        f"y={lidarSensorY:.3f}, "
        f"z={lidarSensorZ:.3f} m"
    )

    print()
    print("RANSAC ground plane:")
    print("--------------------")

    print(
        f"Plane: "
        f"z = {groundA:.6f}x "
        f"+ {groundB:.6f}y "
        f"+ {groundC:.6f}"
    )

    print(
        f"Ground candidates   : "
        f"{bevResult['groundCandidateCount']}"
    )

    print(
        f"RANSAC inliers      : "
        f"{bevResult['ransacInlierCount']}"
    )

    print(
        f"Mean inlier error   : "
        f"{bevResult['ransacMeanInlierDistanceM']:.4f} m"
    )

    print()
    print("Obstacle filtering:")
    print("-------------------")

    print(
        f"Obstacle candidates : "
        f"{bevResult['obstacleCandidateMask'].sum()}"
    )

    print(
        f"Points in self area : "
        f"{bevResult['selfReturnMask'].sum()}"
    )

    print(
        f"Self obstacles removed: "
        f"{removedSelfObstacleCount}"
    )

    print(
        f"Clean obstacles     : "
        f"{bevResult['obstacleMask'].sum()}"
    )

    print()
    print("Multi-channel BEV:")
    print("------------------")

    print(
        f"Tensor shape        : "
        f"{bevResult['multiChannelBev'].shape}"
    )

    print(
        f"Occupied cells      : "
        f"{int(bevResult['occupancyGrid'].sum())}"
    )

    print(
        f"Maximum cell density: "
        f"{bevResult['obstacleDensity'].max():.0f}"
    )

    print(
        f"Maximum height      : "
        f"{bevResult['maxHeightGrid'].max():.2f} m"
    )

    print(
        f"Maximum mean intensity: "
        f"{bevResult['meanIntensityGrid'].max():.2f}"
    )


# ==============================================================
# Save one frame visualization
# ==============================================================

def saveSensorFusionFigure(
    figure,
    scene,
    sampleIndex
):
    """
    Save the complete visualization for one processed sample.
    """

    outputDirectory.mkdir(
        parents=True,
        exist_ok=True
    )

    outputPath = (
        outputDirectory
        / (
            f"{scene['name']}_"
            f"sample_{sampleIndex:04d}_"
            f"sensor_fusion.png"
        )
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


# ==============================================================
# Main NavFusion runner
# ==============================================================

def main():
    """
    Run the NavFusion processing pipeline.

    This runner controls:

        dataset loading,
        scene selection,
        sample selection,
        camera selection,
        LiDAR selection,
        sensor-file loading,
        calibration loading,
        car-pose loading,
        YOLO model loading,
        processing order.

    Processing modules receive already-selected frameData.
    """

    # ----------------------------------------------------------
    # Load nuScenes once
    # ----------------------------------------------------------

    loader = NuScenesLoader(
        dataroot=datasetRoot,
        version=datasetVersion,
        verbose=True
    )

    # ----------------------------------------------------------
    # Load YOLO once
    # ----------------------------------------------------------

    print(
        "\nLoading YOLO11 once..."
    )

    yoloModel = YOLO(
        yoloModelName
    )

    # ----------------------------------------------------------
    # Decide which scenes enter the pipeline
    # ----------------------------------------------------------

    if sceneIndex is None:

        # None means every scene in the dataset.
        selectedScenes = (
            loader.nusc.scene
        )

    else:

        if (
            sceneIndex < 0
            or sceneIndex >= len(
                loader.nusc.scene
            )
        ):
            raise IndexError(
                f"sceneIndex={sceneIndex} is invalid. "
                f"The dataset contains "
                f"{len(loader.nusc.scene)} scenes."
            )

        selectedScenes = [
            loader.nusc.scene[
                sceneIndex
            ]
        ]

    # Global count across all selected scenes.
    processedSampleCount = 0

    # ----------------------------------------------------------
    # Loop through selected scenes
    # ----------------------------------------------------------

    for scene in selectedScenes:

        print()
        print("=" * 70)

        print(
            f"Processing scene: "
            f"{scene['name']}"
        )

        print("=" * 70)

        sampleToken = scene[
            "first_sample_token"
        ]

        sampleIndex = 0

        # ------------------------------------------------------
        # Loop through every sample in the current scene
        # ------------------------------------------------------

        while sampleToken:

            # maxSamples is only a development/testing limit.
            #
            # None means no early stopping.
            if (
                maxSamples is not None
                and processedSampleCount >= maxSamples
            ):
                break

            # --------------------------------------------------
            # Select exactly one sample
            # --------------------------------------------------

            sample = loader.nusc.get(
                "sample",
                sampleToken
            )

            # --------------------------------------------------
            # Load exact data for this sample
            # --------------------------------------------------

            frameData = loadFrameData(
                loader,
                sample
            )

            # --------------------------------------------------
            # Camera + LiDAR + YOLO fusion
            # --------------------------------------------------

            sensorResult = processSensorFusion(
                frameData,
                yoloModel,
                sensorFusionConfig
            )

            # --------------------------------------------------
            # LiDAR + RANSAC + self-return filtering
            # --------------------------------------------------

            # Both algorithms receive the same frameData.
            bevResult = processLidarBev(
                frameData,
                bevConfig
            )

            # --------------------------------------------------
            # Print diagnostics
            # --------------------------------------------------

            printSensorFusionSummary(
                scene,
                sampleIndex,
                sensorResult
            )

            printLidarBevSummary(
                bevResult
            )

            # --------------------------------------------------
            # Create combined visualization
            # --------------------------------------------------

            figure = createSensorFusionFigure(
                sensorResult,
                bevResult,
                visualizationRangeM
            )

            # --------------------------------------------------
            # Save visualization
            # --------------------------------------------------

            if saveImages:

                saveSensorFusionFigure(
                    figure,
                    scene,
                    sampleIndex
                )

            # --------------------------------------------------
            # Display or close visualization
            # --------------------------------------------------

            if showFigure:

                plt.show()

            else:

                plt.close(
                    figure
                )

            # --------------------------------------------------
            # Move to next sample
            # --------------------------------------------------

            sampleToken = sample[
                "next"
            ]

            sampleIndex += 1

            processedSampleCount += 1

        # maxSamples is global, so break out of the scene loop too
        # when the requested development limit has been reached.
        if (
            maxSamples is not None
            and processedSampleCount >= maxSamples
        ):
            break

    # ----------------------------------------------------------
    # Final run summary
    # ----------------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"Total samples processed: "
        f"{processedSampleCount}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()