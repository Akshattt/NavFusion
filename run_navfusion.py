from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud
from ultralytics import YOLO

from navfusion.dataset.nuscenes_loader import NuScenesLoader
from navfusion.perception.sensor_fusion import processSensorFusion
from navfusion.mapping.lidar_bev import processLidarBev
from navfusion.mapping.semantic_bev import createSemanticBev
from navfusion.visualization.sensor_fusion_visualizer import (
    createSensorFusionFigure
)
from navfusion.mapping.camera_ipm import processCameraIpm
from navfusion.mapping.camera_lidar_fusion import processCameraLidarFusion


# ==============================================================
# OUTPUT MODE
# ==============================================================

# This is the ONLY mode flag.
#
# "photo"
#     Process exactly one sample:
#
#         first scene
#         first sample
#
#     Save one semantic-BEV PNG and display it.
#
# "video"
#     Process every scene and every sample in nuScenes-mini.
#
#     Do not display figures.
#     Do not save individual PNG files.
#     Save one protected semantic-BEV MP4.
#
# For the current full-dataset video run:
outputMode = "video"
#outputMode = "photo"


# ==============================================================
# Project paths
# ==============================================================

# This file:
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

    # Shrink the YOLO box by 10 percent on each side before
    # associating projected LiDAR returns.
    "boundingBoxShrinkFactor": 0.10,

    # A fused object must contain at least this many LiDAR points.
    "minimumLidarPoints": 3,

    # Median Absolute Deviation outlier-filter scale.
    "madScale": 3.0
}


# ==============================================================
# LiDAR BEV / RANSAC configuration
# ==============================================================

bevConfig = {
    # ----------------------------------------------------------
    # Physical car-frame BEV region
    # ----------------------------------------------------------

    # x:
    #
    # -20 m behind
    # +50 m forward
    "xMinM": -20.0,
    "xMaxM": 50.0,

    # y:
    #
    # -25 m right
    # +25 m left
    "yMinM": -25.0,
    "yMaxM": 25.0,

    # Each BEV cell represents:
    #
    # 0.20 m x 0.20 m
    #
    # Therefore:
    #
    # 70 / 0.20 = 350 rows
    # 50 / 0.20 = 250 columns
    "resolutionM": 0.20,

    # ----------------------------------------------------------
    # Ground candidate selection
    # ----------------------------------------------------------

    "groundLowerPercentile": 5.0,

    "groundUpperPercentile": 35.0,

    # ----------------------------------------------------------
    # RANSAC
    # ----------------------------------------------------------

    "ransacIterations": 500,

    # Ground candidate is an inlier when its perpendicular
    # distance from the candidate plane is <= 10 cm.
    "ransacDistanceThresholdM": 0.10,

    # Makes development runs deterministic.
    "ransacRandomSeed": 42,

    # ----------------------------------------------------------
    # Obstacle extraction
    # ----------------------------------------------------------

    # Point becomes an obstacle candidate when it is more than
    # 15 cm above the estimated ground plane.
    "minimumObstacleHeightM": 0.15,

    # ----------------------------------------------------------
    # LIDAR_TOP self-return region
    # ----------------------------------------------------------
    #
    # These values deliberately remain in RAW LIDAR_TOP
    # coordinates because this self-return region was already
    # validated in that coordinate frame.

    "selfReturnXMinM": -0.8,
    "selfReturnXMaxM": 0.8,

    "selfReturnYMinM": -1.3,
    "selfReturnYMaxM": 1.9,

    "selfReturnZMinM": -1.2,
    "selfReturnZMaxM": 0.2
}

# ==============================================================
# Camera IPM configuration
# ==============================================================

ipmConfig = {
    # Ground cells closer than this to/behind the camera optical
    # plane are not projected.
    "minimumCameraDepthM": 0.10
}

# ==============================================================
# Camera + LiDAR fusion configuration
# ==============================================================

cameraLidarFusionConfig = {
    # LiDAR obstacle with no known vehicle class.
    "unlabeledObstacleColorRgb": (
        160,
        160,
        160
    ),

    # Preserve some Camera-IPM texture under LiDAR obstacles.
    "unlabeledObstacleAlpha": 0.70,

    # Semantic vehicle cells should be visually dominant.
    "semanticAlpha": 0.95
}

# ==============================================================
# Visualization configuration
# ==============================================================

visualizationRangeM = 50.0


# ==============================================================
# Video configuration
# ==============================================================

# nuScenes keyframes are approximately 2 Hz.
videoFps = 2.0

# IMPORTANT:
#
# This filename is different from the earlier geometric-BEV video:
#
#     navfusion_full_dataset.mp4
#
# Therefore that earlier video is never touched by this run.
semanticVideoFileName = (
    "navfusion_semantic)_IPM_bev_only_front_full_dataset.mp4"
)


# ==============================================================
# Validate output mode
# ==============================================================

def validateOutputMode():
    """
    Validate the one user-controlled output mode.

    Allowed values:

        photo
        video
    """

    validModes = {
        "photo",
        "video"
    }

    if outputMode not in validModes:
        raise ValueError(
            f"Invalid outputMode='{outputMode}'. "
            f"Use either 'photo' or 'video'."
        )


# ==============================================================
# Get sensor record and path
# ==============================================================

def getSensorRecordAndPath(
    loader,
    sample,
    sensorChannel
):
    """
    Get one exact nuScenes sample_data record and its physical
    sensor-file path.

    Dataset selection remains the responsibility of this runner.
    """

    if sensorChannel not in sample[
        "data"
    ]:
        raise KeyError(
            f"Sensor channel '{sensorChannel}' "
            f"is not available in this sample."
        )

    # sample["data"] is a dictionary mapping channel names such as
    #
    # CAM_FRONT
    # LIDAR_TOP
    #
    # to sample_data tokens.
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
        / sensorRecord[
            "filename"
        ]
    )

    if not sensorPath.is_file():
        raise FileNotFoundError(
            f"{sensorChannel} file not found: "
            f"{sensorPath}"
        )

    return (
        sensorRecord,
        sensorPath
    )


# ==============================================================
# Load one synchronized frame
# ==============================================================

def loadFrameData(
    loader,
    sample
):
    """
    Load all source data required for exactly one selected sample.

    This function is the boundary between:

        dataset selection/loading

    and:

        perception/mapping algorithms.

    The downstream modules only process this supplied frameData.
    """

    # ----------------------------------------------------------
    # Camera
    # ----------------------------------------------------------

    cameraRecord, cameraPath = getSensorRecordAndPath(
        loader,
        sample,
        cameraChannel
    )

    # ----------------------------------------------------------
    # LiDAR
    # ----------------------------------------------------------

    lidarRecord, lidarPath = getSensorRecordAndPath(
        loader,
        sample,
        lidarChannel
    )

    # ----------------------------------------------------------
    # Camera image
    # ----------------------------------------------------------

    cameraImage = Image.open(
        cameraPath
    ).convert(
        "RGB"
    )

    # ----------------------------------------------------------
    # LiDAR point cloud
    # ----------------------------------------------------------

    lidarPointCloud = LidarPointCloud.from_file(
        str(
            lidarPath
        )
    )

    # ----------------------------------------------------------
    # LiDAR calibrated_sensor record
    # ----------------------------------------------------------

    lidarCalibration = loader.nusc.get(
        "calibrated_sensor",
        lidarRecord[
            "calibrated_sensor_token"
        ]
    )

    # ----------------------------------------------------------
    # Camera calibrated_sensor record
    # ----------------------------------------------------------

    cameraCalibration = loader.nusc.get(
        "calibrated_sensor",
        cameraRecord[
            "calibrated_sensor_token"
        ]
    )

    # ----------------------------------------------------------
    # Car pose at LiDAR timestamp
    # ----------------------------------------------------------

    # "ego_pose" is the exact nuScenes schema/table name.
    lidarCarPose = loader.nusc.get(
        "ego_pose",
        lidarRecord[
            "ego_pose_token"
        ]
    )

    # ----------------------------------------------------------
    # Car pose at camera timestamp
    # ----------------------------------------------------------

    cameraCarPose = loader.nusc.get(
        "ego_pose",
        cameraRecord[
            "ego_pose_token"
        ]
    )

    # ----------------------------------------------------------
    # Package synchronized source data
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
# Print sensor-fusion summary
# ==============================================================

def printSensorFusionSummary(
    scene,
    sampleIndex,
    sensorResult
):
    """
    Print camera, YOLO and LiDAR-object association diagnostics.
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
            f"  Raw LiDAR xyz: "
            f"x={fusedObject['lidarXM']:.2f}, "
            f"y={fusedObject['lidarYM']:.2f}, "
            f"z={fusedObject['lidarZM']:.2f} m"
        )

        print(
            f"  Car-frame xyz: "
            f"x={fusedObject['carXM']:.2f}, "
            f"y={fusedObject['carYM']:.2f}, "
            f"z={fusedObject['carZM']:.2f} m"
        )

        print(
            f"  LiDAR points : "
            f"{fusedObject['rawLidarCount']} raw -> "
            f"{fusedObject['cleanLidarCount']} filtered"
        )


# ==============================================================
# Print LiDAR BEV summary
# ==============================================================

def printLidarBevSummary(
    bevResult
):
    """
    Print LiDAR-to-car calibration, RANSAC, obstacle filtering and
    geometric BEV diagnostics.
    """

    groundA, groundB, groundC = bevResult[
        "planeCoefficients"
    ]

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

    (
        lidarSensorX,
        lidarSensorY,
        lidarSensorZ
    ) = bevResult[
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
    print("Geometric BEV:")
    print("--------------")

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
# Print semantic BEV summary
# ==============================================================

def printSemanticBevSummary(
    semanticResult
):
    """
    Print semantic-BEV diagnostics.
    """

    print()
    print("Semantic BEV:")
    print("-------------")

    print(
        f"Combined tensor shape: "
        f"{semanticResult['semanticMultiChannelBev'].shape}"
    )

    print(
        f"Semantic cells       : "
        f"{semanticResult['semanticCellCount']}"
    )

    print(
        f"Objects contributing : "
        f"{semanticResult['objectsContributing']}"
    )

    for className, cellCount in semanticResult[
        "classCellCounts"
    ].items():

        print(
            f"{className.capitalize():<20}: "
            f"{cellCount} cells"
        )

    for objectSummary in semanticResult[
        "objectSummaries"
    ]:

        print(
            f"Object "
            f"{objectSummary['objectNumber']} "
            f"({objectSummary['className']}): "
            f"{objectSummary['occupiedSemanticPoints']} "
            f"LiDAR points -> "
            f"{objectSummary['semanticCellCount']} "
            f"semantic cells"
        )


# ==============================================================
# Save one semantic-BEV photo
# ==============================================================

def saveSemanticBevPhoto(
    figure,
    scene,
    sampleIndex
):
    """
    Save one semantic-BEV figure.

    The filename is different from the earlier sensor-fusion PNG,
    so the previous visualization is not replaced.
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
            f"semantic_bev.png"
        )
    )

    figure.savefig(
        outputPath,
        dpi=150,
        bbox_inches="tight"
    )

    print()
    print(
        f"Semantic BEV photo saved to:\n"
        f"{outputPath}"
    )


# ==============================================================
# Convert Matplotlib figure to RGB frame
# ==============================================================

def figureToRgbFrame(
    figure
):
    """
    Render one Matplotlib figure and convert it to an RGB NumPy
    image for FFmpeg.
    """

    # Force complete canvas rendering.
    figure.canvas.draw()

    # Matplotlib canvas:
    #
    # height x width x RGBA
    rgbaFrame = np.asarray(
        figure.canvas.buffer_rgba()
    )

    # FFmpeg is configured for RGB24.
    #
    # Therefore remove the alpha channel.
    rgbFrame = rgbaFrame[
        :,
        :,
        0:3
    ]

    # FFmpeg requires the image bytes to occupy contiguous memory.
    rgbFrame = np.ascontiguousarray(
        rgbFrame,
        dtype=np.uint8
    )

    return rgbFrame


# ==============================================================
# Create protected semantic-BEV video writer
# ==============================================================

def createVideoWriter(
    videoPath,
    frameWidth,
    frameHeight
):
    """
    Start FFmpeg.

    Existing video files are NEVER overwritten.
    """

    # ----------------------------------------------------------
    # Python-level overwrite protection
    # ----------------------------------------------------------

    if videoPath.exists():
        raise FileExistsError(
            f"Video already exists and will not be overwritten:\n"
            f"{videoPath}"
        )

    # ----------------------------------------------------------
    # Locate FFmpeg
    # ----------------------------------------------------------

    ffmpegPath = shutil.which(
        "ffmpeg"
    )

    if ffmpegPath is None:
        raise FileNotFoundError(
            "FFmpeg was not found in Windows PATH."
        )

    outputDirectory.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------------------
    # FFmpeg command
    # ----------------------------------------------------------

    ffmpegCommand = [
        ffmpegPath,

        # Never overwrite an existing file.
        "-n",

        # Only print FFmpeg errors.
        "-loglevel",
        "error",

        # Python supplies raw frames.
        "-f",
        "rawvideo",

        # Three bytes per pixel:
        #
        # R G B
        "-pix_fmt",
        "rgb24",

        # Input frame dimensions.
        "-s",
        f"{frameWidth}x{frameHeight}",

        # Input frame rate.
        "-r",
        str(
            videoFps
        ),

        # Read from Python stdin.
        "-i",
        "-",

        # No audio.
        "-an",

        # H.264 encoding.
        "-c:v",
        "libx264",

        # Quality/speed balance.
        "-preset",
        "medium",

        # Lower CRF means better quality.
        "-crf",
        "18",

        # H.264 yuv420p requires even dimensions.
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",

        # Widely supported MP4 pixel format.
        "-pix_fmt",
        "yuv420p",

        # Put MP4 metadata near the beginning of the file.
        "-movflags",
        "+faststart",

        str(
            videoPath
        )
    ]

    print()
    print("=" * 70)
    print("Starting semantic BEV video")
    print("=" * 70)

    print(
        f"FFmpeg     : "
        f"{ffmpegPath}"
    )

    print(
        f"Output     : "
        f"{videoPath}"
    )

    print(
        f"Frame size : "
        f"{frameWidth} x {frameHeight}"
    )

    print(
        f"Frame rate : "
        f"{videoFps:.1f} FPS"
    )

    print("=" * 70)

    return subprocess.Popen(
        ffmpegCommand,
        stdin=subprocess.PIPE
    )


# ==============================================================
# Write RGB frame to video
# ==============================================================

def writeRgbFrameToVideo(
    videoProcess,
    rgbFrame,
    expectedFrameSize
):
    """
    Write exactly one RGB frame to FFmpeg.

    Every video frame must have identical height and width.
    """

    (
        frameHeight,
        frameWidth,
        frameChannels
    ) = rgbFrame.shape

    if frameChannels != 3:
        raise ValueError(
            f"Expected 3 RGB channels, "
            f"received {frameChannels}."
        )

    currentFrameSize = (
        frameHeight,
        frameWidth
    )

    if currentFrameSize != expectedFrameSize:
        raise ValueError(
            f"Video frame size changed from "
            f"{expectedFrameSize} to "
            f"{currentFrameSize}."
        )

    if videoProcess.stdin is None:
        raise RuntimeError(
            "FFmpeg input pipe is unavailable."
        )

    videoProcess.stdin.write(
        rgbFrame.tobytes()
    )


# ==============================================================
# Finalize video
# ==============================================================

def closeVideoWriter(
    videoProcess,
    videoPath
):
    """
    Close FFmpeg stdin and finalize the MP4 container.
    """

    if videoProcess is None:
        return

    if videoProcess.stdin is not None:
        videoProcess.stdin.close()

    returnCode = videoProcess.wait()

    if returnCode != 0:
        raise RuntimeError(
            f"FFmpeg exited with code {returnCode}."
        )

    print()
    print("=" * 70)

    print(
        f"Semantic BEV video saved to:\n"
        f"{videoPath}"
    )

    print("=" * 70)


# ==============================================================
# Decide which dataset samples to process
# ==============================================================

def getSelectedScenes(
    loader
):
    """
    Resolve dataset selection from the ONE outputMode flag.

    photo:
        first scene only

    video:
        every scene
    """

    if outputMode == "photo":

        return [
            loader.nusc.scene[
                0
            ]
        ]

    # outputMode == "video"
    return loader.nusc.scene


def shouldStopProcessing(
    processedSampleCount
):
    """
    Resolve sample count from the ONE outputMode flag.

    photo:
        stop after one sample

    video:
        never stop early
    """

    if outputMode == "photo":
        return (
            processedSampleCount
            >= 1
        )

    return False


# ==============================================================
# Main NavFusion runner
# ==============================================================

def main():
    """
    Run the NavFusion semantic-BEV pipeline.

    The ONE outputMode flag controls the complete execution mode.

    outputMode = "photo"

        first scene
            ↓
        first sample
            ↓
        sensor fusion
            ↓
        geometric BEV
            ↓
        semantic BEV
            ↓
        save PNG
            ↓
        display figure

    outputMode = "video"

        all scenes
            ↓
        all samples
            ↓
        sensor fusion
            ↓
        geometric BEV
            ↓
        semantic BEV
            ↓
        stream each figure into FFmpeg
            ↓
        save protected MP4
    """

    # ----------------------------------------------------------
    # Validate the only output-mode flag
    # ----------------------------------------------------------

    validateOutputMode()

    print()
    print("=" * 70)

    print(
        f"NavFusion output mode: "
        f"{outputMode}"
    )

    print("=" * 70)

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
    # Resolve scenes using outputMode
    # ----------------------------------------------------------

    selectedScenes = getSelectedScenes(
        loader
    )

    # ----------------------------------------------------------
    # Global processed-sample counter
    # ----------------------------------------------------------

    processedSampleCount = 0

    # ----------------------------------------------------------
    # Video state
    # ----------------------------------------------------------

    videoPath = (
        outputDirectory
        / semanticVideoFileName
    )

    videoProcess = None

    videoFrameSize = None

    # ----------------------------------------------------------
    # Process selected data
    # ----------------------------------------------------------

    try:

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

            # --------------------------------------------------
            # Loop through scene samples
            # --------------------------------------------------

            while sampleToken:

                # In photo mode:
                #
                # stop after the first global sample.
                #
                # In video mode:
                #
                # this remains False for the complete dataset.
                if shouldStopProcessing(
                    processedSampleCount
                ):
                    break

                # ------------------------------------------------
                # Fetch one sample
                # ------------------------------------------------

                sample = loader.nusc.get(
                    "sample",
                    sampleToken
                )

                # ------------------------------------------------
                # Load synchronized source data
                # ------------------------------------------------

                frameData = loadFrameData(
                    loader,
                    sample
                )

                # ------------------------------------------------
                # Camera + YOLO + LiDAR association
                # ------------------------------------------------

                sensorResult = processSensorFusion(
                    frameData,
                    yoloModel,
                    sensorFusionConfig
                )

                # ------------------------------------------------
                # 4-channel geometric car-frame LiDAR BEV
                # ------------------------------------------------
                #
                # D = density
                # H = maximum obstacle height
                # I = mean intensity
                # O = occupancy
                bevResult = processLidarBev(
                    frameData,
                    bevConfig
                )

                # ------------------------------------------------
                # Semantic BEV
                # ------------------------------------------------
                #
                # Adds:
                #
                # S = semantic vehicle class
                #
                # producing:
                #
                # [D, H, I, O, S]
                semanticResult = createSemanticBev(
                    sensorResult,
                    bevResult,
                    bevConfig
                )

                #CAM_FRONT inverse-perspective mapping

                ipmResult = processCameraIpm(
                    frameData,
                    bevResult,
                    bevConfig,
                    ipmConfig
                )

                # ------------------------------------------------
                # Camera IPM + LiDAR + semantic fusion
                # ------------------------------------------------

                cameraLidarFusionResult = processCameraLidarFusion(
                    ipmResult,
                    bevResult,
                    semanticResult,
                    cameraLidarFusionConfig
                )

                print()
                print("Camera IPM:")
                print("-----------")

                print(
                    f"IPM shape       : "
                    f"{ipmResult['cameraIpmRgb'].shape}"
                )

                print(
                    f"Valid BEV cells : "
                    f"{ipmResult['validCellCount']} / "
                    f"{ipmResult['totalCellCount']}"
                )

                print(
                    f"Camera coverage : "
                    f"{ipmResult['coveragePercent']:.2f}%"
                )


                print()
                print("Camera + LiDAR fused BEV:")
                print("-------------------------")

                print(
                    f"Fused shape              : "
                    f"{cameraLidarFusionResult['fusedBevRgb'].shape}"
                )

                print(
                    f"LiDAR obstacle cells     : "
                    f"{cameraLidarFusionResult['occupiedCellCount']}"
                )

                print(
                    f"Semantic vehicle cells   : "
                    f"{cameraLidarFusionResult['semanticCellCount']}"
                )

                print(
                    f"Unlabeled obstacle cells : "
                    f"{cameraLidarFusionResult['unlabeledObstacleCellCount']}"
                )

                print(
                    f"Camera-visible obstacles : "
                    f"{cameraLidarFusionResult['cameraVisibleObstacleCellCount']}"
                )
                # ------------------------------------------------
                # Save and display Camera + LiDAR fused BEV
                # ------------------------------------------------

                if outputMode == "photo":

                    outputDirectory.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    fusedOutputPath = (
                        outputDirectory
                        / (
                            f"{scene['name']}_"
                            f"sample_{sampleIndex:04d}_"
                            f"camera_lidar_fused_bev.png"
                        )
                    )

                    fusedFigure, fusedAxis = plt.subplots(
                        1,
                        1,
                        figsize=(
                            7,
                            10
                        )
                    )

                    # ------------------------------------------------
                    # Display the actual fused RGB BEV
                    # ------------------------------------------------
                    #
                    # Camera IPM
                    #     +
                    # LiDAR obstacle occupancy
                    #     +
                    # semantic vehicle-class colors
                    fusedAxis.imshow(
                        cameraLidarFusionResult[
                            "fusedBevRgb"
                        ],
                        origin="upper"
                    )

                    # ------------------------------------------------
                    # Draw car origin
                    # ------------------------------------------------

                    fusedAxis.scatter(
                        bevResult[
                            "carColumn"
                        ],
                        bevResult[
                            "carRow"
                        ],
                        marker="^",
                        s=80,
                        label="Car"
                    )

                    # ------------------------------------------------
                    # Draw semantic object labels
                    # ------------------------------------------------

                    for objectSummary in semanticResult[
                        "objectSummaries"
                    ]:

                        centerRow = objectSummary[
                            "centerRow"
                        ]

                        centerColumn = objectSummary[
                            "centerColumn"
                        ]

                        if (
                            centerRow is None
                            or centerColumn is None
                        ):
                            continue

                        fusedAxis.text(
                            centerColumn,
                            centerRow,
                            objectSummary[
                                "className"
                            ].upper(),
                            fontsize=8,
                            ha="center",
                            va="center",
                            bbox={
                                "facecolor": "white",
                                "alpha": 0.85,
                                "edgecolor": "black"
                            }
                        )

                    # ------------------------------------------------
                    # Axis information
                    # ------------------------------------------------

                    fusedAxis.set_title(
                        "Fused Camera-IPM + LiDAR + Vehicle Semantics"
                    )

                    fusedAxis.set_xlabel(
                        "BEV column (+y left)"
                    )

                    fusedAxis.set_ylabel(
                        "BEV row (+x forward)"
                    )

                    fusedAxis.legend(
                        loc="lower right"
                    )

                    fusedFigure.tight_layout()

                    # ------------------------------------------------
                    # Save fused diagnostic image
                    # ------------------------------------------------

                    fusedFigure.savefig(
                        fusedOutputPath,
                        dpi=150,
                        bbox_inches="tight"
                    )

                    print()
                    print(
                        f"Fused Camera-LiDAR BEV saved to:\n"
                        f"{fusedOutputPath}"
                    )

                    # Show only in photo mode.
                    plt.show()

                    plt.close(
                        fusedFigure
                    )

                # ------------------------------------------------
                # Console diagnostics
                # ------------------------------------------------

                printSensorFusionSummary(
                    scene,
                    sampleIndex,
                    sensorResult
                )

                printLidarBevSummary(
                    bevResult
                )

                printSemanticBevSummary(
                    semanticResult
                )

                # ------------------------------------------------
                # Create complete 3 x 3 visualization
                # ------------------------------------------------

                figure = createSensorFusionFigure(
                    sensorResult,
                    bevResult,
                    semanticResult,
                    visualizationRangeM
                )

                figure.suptitle(
                    (
                        f"NavFusion Semantic BEV | "
                        f"{scene['name']} | "
                        f"Sample {sampleIndex:04d}"
                    ),
                    fontsize=16,
                    y=0.995
                )

                # ------------------------------------------------
                # PHOTO MODE
                # ------------------------------------------------

                if outputMode == "photo":

                    saveSemanticBevPhoto(
                        figure,
                        scene,
                        sampleIndex
                    )

                    # Only photo mode opens the interactive window.
                    plt.show()

                # ------------------------------------------------
                # VIDEO MODE
                # ------------------------------------------------

                else:

                    rgbFrame = figureToRgbFrame(
                        figure
                    )

                    (
                        frameHeight,
                        frameWidth,
                        _
                    ) = rgbFrame.shape

                    currentFrameSize = (
                        frameHeight,
                        frameWidth
                    )

                    # Start FFmpeg only after the first figure has
                    # been rendered because that determines the
                    # exact video dimensions.
                    if videoProcess is None:

                        videoFrameSize = (
                            currentFrameSize
                        )

                        videoProcess = createVideoWriter(
                            videoPath,
                            frameWidth,
                            frameHeight
                        )

                    if videoFrameSize is None:
                        raise RuntimeError(
                            "Video frame size was not initialized."
                        )

                    writeRgbFrameToVideo(
                        videoProcess,
                        rgbFrame,
                        videoFrameSize
                    )

                    print(
                        f"Semantic video frame written: "
                        f"{processedSampleCount + 1}"
                    )

                # ------------------------------------------------
                # Release Matplotlib memory
                # ------------------------------------------------

                plt.close(
                    figure
                )

                # ------------------------------------------------
                # Move to next nuScenes sample
                # ------------------------------------------------

                sampleToken = sample[
                    "next"
                ]

                sampleIndex += 1

                processedSampleCount += 1

            # --------------------------------------------------
            # Photo mode needs only the first sample
            # --------------------------------------------------

            if shouldStopProcessing(
                processedSampleCount
            ):
                break

    finally:

        # ------------------------------------------------------
        # Finalize video
        # ------------------------------------------------------
        #
        # Photo mode never starts videoProcess.
        #
        # Video mode closes FFmpeg cleanly even if processing
        # raises an exception after some frames have been written.
        if (
            outputMode == "video"
            and videoProcess is not None
        ):

            closeVideoWriter(
                videoProcess,
                videoPath
            )

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"Total samples processed: "
        f"{processedSampleCount}"
    )

    print("=" * 70)

    if outputMode == "video":

        print(
            f"Semantic BEV video:\n"
            f"{videoPath}"
        )

        print("=" * 70)


if __name__ == "__main__":
    main()