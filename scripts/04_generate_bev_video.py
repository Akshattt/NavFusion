from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np

from navfusion.dataset.nuscenes_loader import NuScenesLoader


DATASET_ROOT = Path(
    r"C:\Users\Mettle\Documents\PythonProjects\NavFusion\Datasets\nuScenes"
)

LIDAR_CHANNEL = "LIDAR_TOP"


# --------------------------------------------------------------
# BEV CONFIGURATION
# --------------------------------------------------------------

BEV_X_MIN_M = -20.0
BEV_X_MAX_M = 50.0

BEV_Y_MIN_M = -25.0
BEV_Y_MAX_M = 25.0

BEV_RESOLUTION_M = 0.2


# --------------------------------------------------------------
# OBSTACLE CONFIGURATION
# --------------------------------------------------------------

# Minimum height above the estimated local ground plane
# required for a LiDAR return to be considered an
# obstacle candidate.
MIN_OBSTACLE_HEIGHT_M = 0.15


# --------------------------------------------------------------
# CAR SELF-RETURN REGION
# --------------------------------------------------------------

# Approximate region occupied by the sensor vehicle.
#
# Our single-frame diagnostics showed thousands of returns
# extremely close to the LiDAR origin, including the densest
# BEV cell at approximately:
#
# x ~= 0 m
# y ~= -0.3 m
# z ~= 0 m
#
# These are not useful external navigation obstacles.
CAR_X_MIN_M = -2.5
CAR_X_MAX_M = 2.5

CAR_Y_MIN_M = -1.2
CAR_Y_MAX_M = 1.2


# --------------------------------------------------------------
# VIDEO CONFIGURATION
# --------------------------------------------------------------

# nuScenes keyframe samples are approximately 2 Hz.
#
# Therefore 2 FPS gives roughly real-time playback of the
# keyframe sequence.
VIDEO_FPS = 2

VIDEO_DPI = 120


def getLidarRecordAndPath(loader, sample):
    """Return the LIDAR_TOP sample_data record and physical file path."""

    lidarSampleDataToken = sample["data"][LIDAR_CHANNEL]

    lidarRecord = loader.nusc.get(
        "sample_data",
        lidarSampleDataToken
    )

    lidarPath = (
        loader.dataroot
        / lidarRecord["filename"]
    )

    if not lidarPath.is_file():
        raise FileNotFoundError(
            f"LiDAR file not found: {lidarPath}"
        )

    return lidarRecord, lidarPath


def calculateBevGridSize():
    """Calculate the BEV raster height and width in grid cells."""

    bevHeight = int(
        (BEV_X_MAX_M - BEV_X_MIN_M)
        / BEV_RESOLUTION_M
    )

    bevWidth = int(
        (BEV_Y_MAX_M - BEV_Y_MIN_M)
        / BEV_RESOLUTION_M
    )

    return bevHeight, bevWidth


def filterPointsToBev(points):
    """Keep LiDAR points inside the selected BEV physical region."""

    x = points[0, :]
    y = points[1, :]
    z = points[2, :]
    intensity = points[3, :]

    bevMask = (
        (x >= BEV_X_MIN_M)
        & (x < BEV_X_MAX_M)
        & (y >= BEV_Y_MIN_M)
        & (y < BEV_Y_MAX_M)
    )

    xBev = x[bevMask]
    yBev = y[bevMask]
    zBev = z[bevMask]
    intensityBev = intensity[bevMask]

    return (
        xBev,
        yBev,
        zBev,
        intensityBev
    )


def estimateGroundPlane(
    xBev,
    yBev,
    zBev
):
    """Estimate a simple road plane from low-z LiDAR returns."""

    # Use the lower part of the z distribution as candidate
    # ground points.
    groundLowerZ = np.percentile(
        zBev,
        5
    )

    groundUpperZ = np.percentile(
        zBev,
        35
    )

    groundCandidateMask = (
        (zBev >= groundLowerZ)
        & (zBev <= groundUpperZ)
    )

    xGround = xBev[groundCandidateMask]
    yGround = yBev[groundCandidateMask]
    zGround = zBev[groundCandidateMask]

    # Fit:
    #
    #     z = a*x + b*y + c
    #
    # by solving:
    #
    #     A * coefficients ~= z
    #
    # where each row of A is:
    #
    #     [x_i, y_i, 1]
    groundMatrix = np.column_stack(
        (
            xGround,
            yGround,
            np.ones_like(xGround)
        )
    )

    groundCoefficients, _, _, _ = np.linalg.lstsq(
        groundMatrix,
        zGround,
        rcond=None
    )

    groundA = groundCoefficients[0]
    groundB = groundCoefficients[1]
    groundC = groundCoefficients[2]

    return (
        groundA,
        groundB,
        groundC
    )


def calculateHeightAboveGround(
    xBev,
    yBev,
    zBev,
    groundA,
    groundB,
    groundC
):
    """Calculate each LiDAR return's height above the fitted ground."""

    estimatedGroundZ = (
        groundA * xBev
        + groundB * yBev
        + groundC
    )

    heightAboveGround = (
        zBev - estimatedGroundZ
    )

    return heightAboveGround


def createObstacleMask(
    xBev,
    yBev,
    heightAboveGround
):
    """Select obstacle candidates while removing car self-returns."""

    # Identify points inside the approximate car footprint.
    carFootprintMask = (
        (xBev >= CAR_X_MIN_M)
        & (xBev <= CAR_X_MAX_M)
        & (yBev >= CAR_Y_MIN_M)
        & (yBev <= CAR_Y_MAX_M)
    )

    # An obstacle candidate must:
    #
    # 1. Be sufficiently high above the estimated ground.
    # 2. Be outside the car's own footprint.
    obstacleMask = (
        (heightAboveGround > MIN_OBSTACLE_HEIGHT_M)
        & (~carFootprintMask)
    )

    return (
        obstacleMask,
        carFootprintMask
    )


def convertMetersToBevGrid(
    xBev,
    yBev,
    bevHeight,
    bevWidth
):
    """Convert physical x/y metres into BEV image rows and columns."""

    xCell = (
        (xBev - BEV_X_MIN_M)
        / BEV_RESOLUTION_M
    ).astype(int)

    yCell = (
        (yBev - BEV_Y_MIN_M)
        / BEV_RESOLUTION_M
    ).astype(int)

    # Image row 0 is at the top.
    #
    # Reverse x so forward (+x) appears upward.
    bevRow = (
        bevHeight
        - 1
        - xCell
    )

    # Reverse y so left (+y) appears on the left side
    # of the displayed BEV.
    bevColumn = (
        bevWidth
        - 1
        - yCell
    )

    return (
        bevRow,
        bevColumn
    )


def createDensityGrid(
    rows,
    columns,
    bevHeight,
    bevWidth
):
    """Rasterize point row/column locations into a density grid."""

    densityGrid = np.zeros(
        (bevHeight, bevWidth),
        dtype=np.float32
    )

    np.add.at(
        densityGrid,
        (rows, columns),
        1
    )

    return densityGrid


def createDisplayDensity(
    densityGrid
):
    """Log-compress and normalize a density grid for visualization."""

    densityDisplay = np.log1p(
        densityGrid
    )

    if densityDisplay.max() > 0:
        densityDisplay = (
            densityDisplay
            / densityDisplay.max()
        )

    return densityDisplay


def calculateCarGridPosition(
    bevHeight,
    bevWidth
):
    """Calculate the BEV row/column corresponding to x=0, y=0."""

    carXCell = int(
        (0.0 - BEV_X_MIN_M)
        / BEV_RESOLUTION_M
    )

    carYCell = int(
        (0.0 - BEV_Y_MIN_M)
        / BEV_RESOLUTION_M
    )

    carRow = (
        bevHeight
        - 1
        - carXCell
    )

    carColumn = (
        bevWidth
        - 1
        - carYCell
    )

    return (
        carRow,
        carColumn
    )


def processSample(
    loader,
    sample,
    bevHeight,
    bevWidth
):
    """Process one nuScenes sample into all-LiDAR and obstacle BEVs."""

    lidarRecord, lidarPath = getLidarRecordAndPath(
        loader,
        sample
    )

    lidarPointCloud = LidarPointCloud.from_file(
        str(lidarPath)
    )

    points = lidarPointCloud.points

    (
        xBev,
        yBev,
        zBev,
        intensityBev
    ) = filterPointsToBev(
        points
    )

    (
        groundA,
        groundB,
        groundC
    ) = estimateGroundPlane(
        xBev,
        yBev,
        zBev
    )

    heightAboveGround = calculateHeightAboveGround(
        xBev,
        yBev,
        zBev,
        groundA,
        groundB,
        groundC
    )

    (
        obstacleMask,
        carFootprintMask
    ) = createObstacleMask(
        xBev,
        yBev,
        heightAboveGround
    )

    (
        bevRow,
        bevColumn
    ) = convertMetersToBevGrid(
        xBev,
        yBev,
        bevHeight,
        bevWidth
    )

    # --------------------------------------------------------------
    # ALL-LIDAR BEV
    # --------------------------------------------------------------

    bevDensity = createDensityGrid(
        bevRow,
        bevColumn,
        bevHeight,
        bevWidth
    )

    bevDensityDisplay = createDisplayDensity(
        bevDensity
    )

    # --------------------------------------------------------------
    # OBSTACLE-ONLY BEV
    # --------------------------------------------------------------

    obstacleRow = bevRow[obstacleMask]
    obstacleColumn = bevColumn[obstacleMask]

    obstacleDensity = createDensityGrid(
        obstacleRow,
        obstacleColumn,
        bevHeight,
        bevWidth
    )

    obstacleDensityDisplay = createDisplayDensity(
        obstacleDensity
    )

    diagnostics = {
        "total_points": points.shape[1],
        "bev_points": xBev.shape[0],
        "obstacle_points": obstacleMask.sum(),
        "car_points": carFootprintMask.sum(),
        "ground_a": groundA,
        "ground_b": groundB,
        "ground_c": groundC,
        "timestamp": lidarRecord["timestamp"]
    }

    return (
        bevDensityDisplay,
        obstacleDensityDisplay,
        diagnostics
    )


def drawFrame(
    axes,
    bevDensityDisplay,
    obstacleDensityDisplay,
    carRow,
    carColumn,
    sceneName,
    frameNumber,
    totalFrames,
    diagnostics
):
    """Draw one side-by-side video frame."""

    axes[0].clear()
    axes[1].clear()

    # --------------------------------------------------------------
    # LEFT: ALL LIDAR RETURNS
    # --------------------------------------------------------------

    axes[0].imshow(
        bevDensityDisplay
    )

    axes[0].set_title(
        "All LiDAR Returns"
    )

    axes[0].set_xlabel(
        "Left / Right"
    )

    axes[0].set_ylabel(
        "Forward / Backward"
    )

    axes[0].scatter(
        carColumn,
        carRow,
        marker="x",
        s=80,
        label="Car"
    )

    axes[0].legend(
        loc="upper right"
    )

    # --------------------------------------------------------------
    # RIGHT: OBSTACLE CANDIDATES
    # --------------------------------------------------------------

    axes[1].imshow(
        obstacleDensityDisplay
    )

    axes[1].set_title(
        "Ground-Filtered Obstacles"
    )

    axes[1].set_xlabel(
        "Left / Right"
    )

    axes[1].set_ylabel(
        "Forward / Backward"
    )

    axes[1].scatter(
        carColumn,
        carRow,
        marker="x",
        s=80,
        label="Car"
    )

    axes[1].legend(
        loc="upper right"
    )

    # --------------------------------------------------------------
    # FRAME TITLE
    # --------------------------------------------------------------

    frameText = (
        f"{sceneName} | "
        f"Frame {frameNumber}/{totalFrames} | "
        f"BEV points: {diagnostics['bev_points']} | "
        f"Obstacle points: {diagnostics['obstacle_points']}"
    )

    axes[0].figure.suptitle(
        frameText
    )


def main():
    """Generate a BEV ground-filtering video for the first scene."""

    loader = NuScenesLoader(
        dataroot=DATASET_ROOT,
        version="v1.0-mini",
        verbose=True
    )

    # --------------------------------------------------------------
    # SELECT FIRST SCENE
    # --------------------------------------------------------------

    scene = loader.nusc.scene[0]

    sceneName = scene["name"]

    totalFrames = scene["nbr_samples"]

    print("\nNavFusion BEV Video")
    print("===================")

    print(
        f"Scene        : "
        f"{sceneName}"
    )

    print(
        f"Samples      : "
        f"{totalFrames}"
    )

    print(
        f"Video FPS    : "
        f"{VIDEO_FPS}"
    )

    # --------------------------------------------------------------
    # CALCULATE CONSTANT GRID SIZE
    # --------------------------------------------------------------

    (
        bevHeight,
        bevWidth
    ) = calculateBevGridSize()

    (
        carRow,
        carColumn
    ) = calculateCarGridPosition(
        bevHeight,
        bevWidth
    )

    # --------------------------------------------------------------
    # OUTPUT PATH
    # --------------------------------------------------------------

    projectRoot = Path(__file__).resolve().parents[2]

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
        / "scene_0_lidar_ground_filter_video.mp4"
    )

    # --------------------------------------------------------------
    # CREATE MATPLOTLIB VIDEO FIGURE
    # --------------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 8)
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    # FFMpegWriter sends each Matplotlib frame to the installed
    # ffmpeg executable and encodes the final MP4.
    writer = FFMpegWriter(
        fps=VIDEO_FPS,
        metadata={
            "title": "NavFusion LiDAR Ground Filtering",
            "artist": "NavFusion"
        },
        bitrate=4000
    )

    # --------------------------------------------------------------
    # WALK THROUGH EVERY SAMPLE IN THE SCENE
    # --------------------------------------------------------------

    sampleToken = scene["first_sample_token"]

    frameNumber = 1

    with writer.saving(
        figure,
        str(outputPath),
        dpi=VIDEO_DPI
    ):

        while sampleToken:

            sample = loader.nusc.get(
                "sample",
                sampleToken
            )

            (
                bevDensityDisplay,
                obstacleDensityDisplay,
                diagnostics
            ) = processSample(
                loader,
                sample,
                bevHeight,
                bevWidth
            )

            drawFrame(
                axes,
                bevDensityDisplay,
                obstacleDensityDisplay,
                carRow,
                carColumn,
                sceneName,
                frameNumber,
                totalFrames,
                diagnostics
            )

            # Capture the current Matplotlib figure and send it
            # to ffmpeg as the next video frame.
            writer.grab_frame()

            print(
                f"Frame "
                f"{frameNumber:02d}/{totalFrames:02d} | "
                f"BEV points: "
                f"{diagnostics['bev_points']:5d} | "
                f"Obstacle points: "
                f"{diagnostics['obstacle_points']:5d} | "
                f"Car-region points: "
                f"{diagnostics['car_points']:5d}"
            )

            # nuScenes sample["next"] contains the token of the
            # next keyframe sample.
            #
            # The final sample contains an empty string, ending
            # the while loop.
            sampleToken = sample["next"]

            frameNumber += 1

    plt.close(
        figure
    )

    print("\nVideo generation complete.")

    print(
        f"Saved to:\n"
        f"{outputPath}"
    )


if __name__ == "__main__":
    main()