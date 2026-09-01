from pathlib import Path

from nuscenes.utils.data_classes import LidarPointCloud

from navfusion.dataset.nuscenes_loader import NuScenesLoader

import numpy as np

from matplotlib import pyplot as plt


DATASET_ROOT = Path(
    r"C:\Users\Mettle\Documents\PythonProjects\Navfusion\Datasets\nuScenes"
)

LIDAR_CHANNEL = "LIDAR_TOP"


def getLidarRecordandPath(loader, sample):
    """To get the LiDAR top metadata record and physical file path."""

    lidarSameDataToken = sample["data"][LIDAR_CHANNEL]

    lidarRecord = loader.nusc.get(
        "sample_data",
        lidarSameDataToken
    )

    lidarPath = loader.dataroot / lidarRecord["filename"]

    if not lidarPath.is_file():
        raise FileNotFoundError(
            f"File not found : {lidarPath}"
        )

    return lidarRecord, lidarPath


def main():
    """Load the first nuScenes LiDAR scan for BEV generation."""

    loader = NuScenesLoader(
        dataroot=DATASET_ROOT,
        version="v1.0-mini",
        verbose=True
    )

    # --------------------------------------------------------------
    # LOAD FIRST SCENE AND SAMPLE
    # --------------------------------------------------------------

    # Select the first scene.
    scene = loader.nusc.scene[0]

    # Get the first sample belonging to that scene.
    firstSampleToken = scene["first_sample_token"]

    sample = loader.nusc.get(
        "sample",
        firstSampleToken
    )

    # Get the LIDAR_TOP metadata and physical .bin file.
    lidarRecord, lidarPath = getLidarRecordandPath(
        loader,
        sample
    )

    # Load the binary LiDAR scan as a LidarPointCloud object.
    lidarPointCloud = LidarPointCloud.from_file(
        str(lidarPath)
    )

    # lidarPointCloud.points has shape (4, N):
    #
    # row 0 -> x
    # row 1 -> y
    # row 2 -> z
    # row 3 -> intensity
    points = lidarPointCloud.points

    x = points[0, :]
    y = points[1, :]
    z = points[2, :]
    intensity = points[3, :]

    # --------------------------------------------------------------
    # BEV CONFIGURATION
    # --------------------------------------------------------------

    # X represents forward/backward distance relative to the LiDAR.
    BEV_X_MIN_M = -20.0
    BEV_X_MAX_M = 50.0

    # Y represents left/right distance relative to the LiDAR.
    BEV_Y_MIN_M = -25.0
    BEV_Y_MAX_M = 25.0

    # Each BEV grid cell represents:
    #
    # 0.2 m x 0.2 m
    BEV_RESOLUTION_M = 0.2

    # --------------------------------------------------------------
    # CAR-FOOTPRINT REGION
    # --------------------------------------------------------------

    # These values define an approximate rectangular region around
    # the car.
    #
    # We are NOT using this region to remove points yet.
    #
    # For now, it is only used for diagnostics so we can measure how
    # many LiDAR points occur near/inside this region.
    CAR_X_MIN_M = -2.5
    CAR_X_MAX_M = 2.5

    CAR_Y_MIN_M = -1.2
    CAR_Y_MAX_M = 1.2

    # Radius used to inspect all LiDAR points close to the car.
    NEAR_CAR_RADIUS_M = 3.0

    # --------------------------------------------------------------
    # OLD ABSOLUTE-Z OBSTACLE THRESHOLD
    # --------------------------------------------------------------

    # We originally used:
    #
    # OBSTACLE_Z_MIN_M = -1.5
    #
    # This meant:
    #
    #     z > -1.5 m -> obstacle candidate
    #
    # We are no longer using this as the main obstacle rule because
    # z is measured relative to the LiDAR sensor rather than relative
    # to the road surface.
    #
    # A low object such as a rock could still damage the car while
    # having:
    #
    #     z < -1.5 m
    #
    # Therefore this old threshold remains commented for development
    # history.
    #
    # OBSTACLE_Z_MIN_M = -1.5

    # --------------------------------------------------------------
    # FILTER POINTS TO THE BEV REGION
    # --------------------------------------------------------------

    # Keep only LiDAR returns inside our selected rectangular region.
    #
    # Half-open upper bounds are used:
    #
    #     x >= minimum
    #     x < maximum
    #
    # This prevents a point exactly on the maximum boundary from
    # producing an invalid grid index.
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

    # --------------------------------------------------------------
    # ESTIMATE THE GROUND PLANE
    # --------------------------------------------------------------

    # Use the lower part of the LiDAR z-distribution as candidate
    # road/ground points.
    #
    # Candidate range:
    #
    #     5th percentile <= z <= 35th percentile
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

    # --------------------------------------------------------------
    # GROUND-PLANE MATHEMATICAL MODEL
    # --------------------------------------------------------------

    # Model the road as:
    #
    #     z = a*x + b*y + c
    #
    # For each ground candidate:
    #
    #     z_i = a*x_i + b*y_i + c
    #
    # We can write this as:
    #
    #     [x_i, y_i, 1] [a]   [z_i]
    #                   [b] =
    #                   [c]
    #
    # groundMatrix therefore contains one row:
    #
    #     [x_i, y_i, 1]
    #
    # for every candidate ground point.
    groundMatrix = np.column_stack(
        (
            xGround,
            yGround,
            np.ones_like(xGround)
        )
    )

    # Solve for:
    #
    #     a
    #     b
    #     c
    #
    # using least squares.
    #
    # np.linalg.lstsq() chooses the plane coefficients that minimize
    # the total squared vertical fitting error.
    groundCoefficients, _, _, _ = np.linalg.lstsq(
        groundMatrix,
        zGround,
        rcond=None
    )

    groundA = groundCoefficients[0]
    groundB = groundCoefficients[1]
    groundC = groundCoefficients[2]

    # --------------------------------------------------------------
    # ESTIMATE LOCAL GROUND HEIGHT
    # --------------------------------------------------------------

    # Estimate the road height underneath every BEV LiDAR return:
    #
    #     estimatedGroundZ =
    #         a*x + b*y + c
    estimatedGroundZ = (
        groundA * xBev
        + groundB * yBev
        + groundC
    )

    # Calculate how far each LiDAR return lies above its estimated
    # local ground surface:
    #
    #     heightAboveGround =
    #         zPoint - zGround
    #
    # Example:
    #
    #     estimated ground = -1.80 m
    #     rock top         = -1.60 m
    #
    # Therefore:
    #
    #     height = -1.60 - (-1.80)
    #            = 0.20 m
    heightAboveGround = (
        zBev - estimatedGroundZ
    )

    # --------------------------------------------------------------
    # OLD ABSOLUTE-Z OBSTACLE FILTER
    # --------------------------------------------------------------

    # This was our first obstacle-filtering method:
    #
    # obstacleMask = (
    #     zBev > OBSTACLE_Z_MIN_M
    # )
    #
    # xObstacle = xBev[obstacleMask]
    # yObstacle = yBev[obstacleMask]
    # zObstacle = zBev[obstacleMask]
    #
    # It remains commented so the development history is preserved.

    # --------------------------------------------------------------
    # GROUND-RELATIVE OBSTACLE FILTER
    # --------------------------------------------------------------

    # Any return more than 15 cm above the estimated local road
    # surface is currently treated as an obstacle candidate.
    #
    # This is an MVP threshold and can later be improved.
    MIN_OBSTACLE_HEIGHT_M = 0.15

    obstacleMask = (
        heightAboveGround > MIN_OBSTACLE_HEIGHT_M
    )

    # --------------------------------------------------------------
    # CAR-FOOTPRINT DIAGNOSTIC MASK
    # --------------------------------------------------------------

    # This identifies points whose x/y position falls inside the
    # approximate car rectangle.
    #
    # IMPORTANT:
    #
    # We are calculating this mask for diagnostics only.
    # We are NOT removing these points yet.
    carFootprintMask = (
        (xBev >= CAR_X_MIN_M)
        & (xBev <= CAR_X_MAX_M)
        & (yBev >= CAR_Y_MIN_M)
        & (yBev <= CAR_Y_MAX_M)
    )

    # --------------------------------------------------------------
    # EXPERIMENTAL CAR SELF-RETURN FILTER
    # --------------------------------------------------------------

    # We previously proposed this:
    #
    # obstacleMask = (
    #     (heightAboveGround > MIN_OBSTACLE_HEIGHT_M)
    #     & (~carFootprintMask)
    # )
    #
    # The ~ operator means Boolean NOT.
    #
    # Therefore:
    #
    #     ~carFootprintMask
    #
    # keeps only points outside the approximate car footprint.
    #
    # HOWEVER:
    #
    # We have not yet proven that the dense points around the origin
    # are actually returns from the car itself.
    #
    # Therefore this version remains COMMENTED OUT until the
    # diagnostics below tell us what those points actually are.

    # --------------------------------------------------------------
    # CREATE OBSTACLE POINT ARRAYS
    # --------------------------------------------------------------

    xObstacle = xBev[obstacleMask]
    yObstacle = yBev[obstacleMask]
    zObstacle = zBev[obstacleMask]

    # Keep the ground-relative height for each obstacle candidate.
    heightObstacle = heightAboveGround[obstacleMask]

    # --------------------------------------------------------------
    # CALCULATE BEV GRID DIMENSIONS
    # --------------------------------------------------------------

    # X dimension:
    #
    #     50 - (-20)
    #     = 70 m
    #
    # At:
    #
    #     0.2 m/cell
    #
    # we obtain:
    #
    #     70 / 0.2
    #     = 350 cells
    bevHeight = int(
        (BEV_X_MAX_M - BEV_X_MIN_M)
        / BEV_RESOLUTION_M
    )

    # Y dimension:
    #
    #     25 - (-25)
    #     = 50 m
    #
    # Therefore:
    #
    #     50 / 0.2
    #     = 250 cells
    bevWidth = int(
        (BEV_Y_MAX_M - BEV_Y_MIN_M)
        / BEV_RESOLUTION_M
    )

    # --------------------------------------------------------------
    # CONVERT PHYSICAL METRES TO GRID CELLS
    # --------------------------------------------------------------

    # Convert x from metres into a discrete grid-cell index:
    #
    #     xCell =
    #         (x - X_MIN) / resolution
    xCell = (
        (xBev - BEV_X_MIN_M)
        / BEV_RESOLUTION_M
    ).astype(int)

    # Convert y from metres into a discrete grid-cell index:
    #
    #     yCell =
    #         (y - Y_MIN) / resolution
    yCell = (
        (yBev - BEV_Y_MIN_M)
        / BEV_RESOLUTION_M
    ).astype(int)

    # --------------------------------------------------------------
    # CONVERT GRID CELLS TO IMAGE ROW/COLUMN
    # --------------------------------------------------------------

    # Image row 0 is located at the TOP.
    #
    # We want:
    #
    #     +x / forward
    #
    # to appear toward the TOP of the image.
    bevRow = bevHeight - 1 - xCell

    # We want:
    #
    #     +y / left of car
    #
    # to appear on the LEFT side of the image.
    bevColumn = bevWidth - 1 - yCell

    # --------------------------------------------------------------
    # CREATE OBSTACLE GRID LOCATIONS
    # --------------------------------------------------------------

    # Keep only the rows/columns corresponding to obstacle candidates.
    obstacleRow = bevRow[obstacleMask]
    obstacleColumn = bevColumn[obstacleMask]

    # --------------------------------------------------------------
    # CREATE OBSTACLE DENSITY GRID
    # --------------------------------------------------------------

    obstacleDensity = np.zeros(
        (bevHeight, bevWidth),
        dtype=np.float32
    )

    # Add one count for every obstacle return entering a BEV cell.
    np.add.at(
        obstacleDensity,
        (obstacleRow, obstacleColumn),
        1
    )

    # Log-compress obstacle density:
    #
    #     log(1 + density)
    obstacleDensityDisplay = np.log1p(
        obstacleDensity
    )

    # Normalize display values to:
    #
    #     0 <= value <= 1
    if obstacleDensityDisplay.max() > 0:
        obstacleDensityDisplay = (
            obstacleDensityDisplay
            / obstacleDensityDisplay.max()
        )

    # --------------------------------------------------------------
    # OLD DUPLICATE OBSTACLE MASK
    # --------------------------------------------------------------

    # This duplicate previously appeared later in the file:
    #
    # obstacleMask = (
    #     heightAboveGround > MIN_OBSTACLE_HEIGHT_M
    # )
    #
    # It is preserved as a comment because the active obstacleMask
    # was already calculated above.

    # --------------------------------------------------------------
    # CREATE ALL-LIDAR DENSITY GRID
    # --------------------------------------------------------------

    bevDensity = np.zeros(
        (bevHeight, bevWidth),
        dtype=np.float32
    )

    # Add every BEV LiDAR point into its corresponding grid cell.
    np.add.at(
        bevDensity,
        (bevRow, bevColumn),
        1
    )

    # --------------------------------------------------------------
    # DIAGNOSE THE DENSEST BEV CELL
    # --------------------------------------------------------------

    # np.argmax() finds the flattened array position containing the
    # largest density value.
    #
    # np.unravel_index() converts that flattened position back into:
    #
    #     row
    #     column
    maxDensityRow, maxDensityColumn = np.unravel_index(
        np.argmax(bevDensity),
        bevDensity.shape
    )

    # Identify every LiDAR point that was mapped into exactly this
    # row and column.
    maxDensityCellMask = (
        (bevRow == maxDensityRow)
        & (bevColumn == maxDensityColumn)
    )

    xMaxDensity = xBev[maxDensityCellMask]
    yMaxDensity = yBev[maxDensityCellMask]
    zMaxDensity = zBev[maxDensityCellMask]

    # --------------------------------------------------------------
    # CONVERT DENSEST IMAGE CELL BACK TO PHYSICAL GRID CELL
    # --------------------------------------------------------------

    # Earlier we used:
    #
    #     bevRow = bevHeight - 1 - xCell
    #
    # Therefore reverse it:
    #
    #     xCell = bevHeight - 1 - bevRow
    maxDensityXCell = (
        bevHeight - 1 - maxDensityRow
    )

    # Likewise:
    #
    #     bevColumn = bevWidth - 1 - yCell
    #
    # therefore:
    #
    #     yCell = bevWidth - 1 - bevColumn
    maxDensityYCell = (
        bevWidth - 1 - maxDensityColumn
    )

    # --------------------------------------------------------------
    # DETERMINE PHYSICAL X/Y BOUNDS OF THE DENSEST CELL
    # --------------------------------------------------------------

    # Lower physical x boundary:
    #
    #     X_MIN + cellIndex * resolution
    maxDensityXMin = (
        BEV_X_MIN_M
        + maxDensityXCell * BEV_RESOLUTION_M
    )

    maxDensityXMax = (
        maxDensityXMin
        + BEV_RESOLUTION_M
    )

    maxDensityYMin = (
        BEV_Y_MIN_M
        + maxDensityYCell * BEV_RESOLUTION_M
    )

    maxDensityYMax = (
        maxDensityYMin
        + BEV_RESOLUTION_M
    )

    # --------------------------------------------------------------
    # INSPECT ALL POINTS CLOSE TO THE CAR
    # --------------------------------------------------------------

    # Compute the horizontal Euclidean distance of every point from
    # the LiDAR/car origin:
    #
    #     distance = sqrt(x^2 + y^2)
    distanceFromCar = np.sqrt(
        xBev ** 2
        + yBev ** 2
    )

    # Keep points whose horizontal distance is at most 3 metres.
    nearCarMask = (
        distanceFromCar <= NEAR_CAR_RADIUS_M
    )

    # --------------------------------------------------------------
    # CREATE DISPLAY VERSION OF ALL-LIDAR DENSITY
    # --------------------------------------------------------------

    bevDensityDisplay = np.log1p(
        bevDensity
    )

    if bevDensityDisplay.max() > 0:
        bevDensityDisplay = (
            bevDensityDisplay
            / bevDensityDisplay.max()
        )

    # --------------------------------------------------------------
    # DIAGNOSTICS: BEV DENSITY
    # --------------------------------------------------------------

    print("\nBEV density:")

    print(
        f"Grid shape       : "
        f"{bevDensity.shape}"
    )

    print(
        f"Occupied cells   : "
        f"{(bevDensity > 0).sum()}"
    )

    print(
        f"Maximum density  : "
        f"{bevDensity.max():.0f} points/cell"
    )

    # This should equal the number of points surviving bevMask.
    print(
        f"Total point count: "
        f"{bevDensity.sum():.0f}"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: DENSEST BEV CELL
    # --------------------------------------------------------------

    print("\nDensest BEV cell:")

    print(
        f"Grid location     : "
        f"row {maxDensityRow}, "
        f"column {maxDensityColumn}"
    )

    print(
        f"Number of points  : "
        f"{maxDensityCellMask.sum()}"
    )

    print(
        f"x cell range      : "
        f"{maxDensityXMin:.2f} "
        f"to {maxDensityXMax:.2f} m"
    )

    print(
        f"y cell range      : "
        f"{maxDensityYMin:.2f} "
        f"to {maxDensityYMax:.2f} m"
    )

    print(
        f"Actual x range    : "
        f"{xMaxDensity.min():.4f} "
        f"to {xMaxDensity.max():.4f} m"
    )

    print(
        f"Actual y range    : "
        f"{yMaxDensity.min():.4f} "
        f"to {yMaxDensity.max():.4f} m"
    )

    print(
        f"Actual z range    : "
        f"{zMaxDensity.min():.4f} "
        f"to {zMaxDensity.max():.4f} m"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: NEAR-CAR LIDAR
    # --------------------------------------------------------------

    print("\nNear-car LiDAR inspection:")

    print(
        f"Radius inspected  : "
        f"{NEAR_CAR_RADIUS_M:.1f} m"
    )

    print(
        f"Points found      : "
        f"{nearCarMask.sum()}"
    )

    print(
        f"x range           : "
        f"{xBev[nearCarMask].min():.2f} "
        f"to {xBev[nearCarMask].max():.2f} m"
    )

    print(
        f"y range           : "
        f"{yBev[nearCarMask].min():.2f} "
        f"to {yBev[nearCarMask].max():.2f} m"
    )

    print(
        f"z range           : "
        f"{zBev[nearCarMask].min():.2f} "
        f"to {zBev[nearCarMask].max():.2f} m"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: CAR-FOOTPRINT REGION
    # --------------------------------------------------------------

    print("\nCar-footprint inspection:")

    print(
        f"x region          : "
        f"{CAR_X_MIN_M:.1f} "
        f"to {CAR_X_MAX_M:.1f} m"
    )

    print(
        f"y region          : "
        f"{CAR_Y_MIN_M:.1f} "
        f"to {CAR_Y_MAX_M:.1f} m"
    )

    print(
        f"Points inside     : "
        f"{carFootprintMask.sum()}"
    )

    print(
        "Filtering applied  : No"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: HEIGHT DISTRIBUTION
    # --------------------------------------------------------------

    print("\nBEV height distribution:")

    print(
        f"1st percentile  : "
        f"{np.percentile(zBev, 1):.2f} m"
    )

    print(
        f"5th percentile  : "
        f"{np.percentile(zBev, 5):.2f} m"
    )

    print(
        f"10th percentile : "
        f"{np.percentile(zBev, 10):.2f} m"
    )

    print(
        f"25th percentile : "
        f"{np.percentile(zBev, 25):.2f} m"
    )

    print(
        f"50th percentile : "
        f"{np.percentile(zBev, 50):.2f} m"
    )

    print(
        f"75th percentile : "
        f"{np.percentile(zBev, 75):.2f} m"
    )

    print(
        f"90th percentile : "
        f"{np.percentile(zBev, 90):.2f} m"
    )

    print(
        f"95th percentile : "
        f"{np.percentile(zBev, 95):.2f} m"
    )

    print(
        f"99th percentile : "
        f"{np.percentile(zBev, 99):.2f} m"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: BEV REGION
    # --------------------------------------------------------------

    print("\nBEV region:")

    print(
        f"x range        : "
        f"{BEV_X_MIN_M:.1f} "
        f"to {BEV_X_MAX_M:.1f} m"
    )

    print(
        f"y range        : "
        f"{BEV_Y_MIN_M:.1f} "
        f"to {BEV_Y_MAX_M:.1f} m"
    )

    print(
        f"Points in BEV  : "
        f"{bevMask.sum()}"
    )

    print("\nFiltered BEV coordinate ranges:")

    print(
        f"xBev : "
        f"{xBev.min():.2f} "
        f"to {xBev.max():.2f} m"
    )

    print(
        f"yBev : "
        f"{yBev.min():.2f} "
        f"to {yBev.max():.2f} m"
    )

    print(
        f"zBev : "
        f"{zBev.min():.2f} "
        f"to {zBev.max():.2f} m"
    )

    print(
        f"intensityBev : "
        f"{intensityBev.min():.2f} "
        f"to {intensityBev.max():.2f}"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: GRID
    # --------------------------------------------------------------

    print("\nBEV grid:")

    print(
        f"Resolution      : "
        f"{BEV_RESOLUTION_M:.2f} m/cell"
    )

    print(
        f"Grid height     : "
        f"{bevHeight} cells"
    )

    print(
        f"Grid width      : "
        f"{bevWidth} cells"
    )

    print(
        f"Row index range : "
        f"{bevRow.min()} "
        f"to {bevRow.max()}"
    )

    print(
        f"Col index range : "
        f"{bevColumn.min()} "
        f"to {bevColumn.max()}"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: ORIGINAL LIDAR INPUT
    # --------------------------------------------------------------

    print("\nNavFusion BEV - LiDAR Input")
    print("============================")

    print(
        f"Scene            : "
        f"{scene['name']}"
    )

    print(
        f"Sample token     : "
        f"{sample['token']}"
    )

    print(
        f"LiDAR file       : "
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
        f"Timestamp        : "
        f"{lidarRecord['timestamp']}"
    )

    print("\nCoordinate ranges:")

    print(
        f"x : "
        f"{x.min():.2f} "
        f"to {x.max():.2f} m"
    )

    print(
        f"y : "
        f"{y.min():.2f} "
        f"to {y.max():.2f} m"
    )

    print(
        f"z : "
        f"{z.min():.2f} "
        f"to {z.max():.2f} m"
    )

    print(
        f"intensity : "
        f"{intensity.min():.2f} "
        f"to {intensity.max():.2f}"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: GROUND PLANE
    # --------------------------------------------------------------

    print("\nGround-plane estimation:")

    print(
        f"Ground candidate z range : "
        f"{groundLowerZ:.2f} "
        f"to {groundUpperZ:.2f} m"
    )

    print(
        f"Ground candidate points  : "
        f"{zGround.shape[0]}"
    )

    print(
        f"Ground plane             : "
        f"z = {groundA:.6f}x "
        f"+ {groundB:.6f}y "
        f"+ {groundC:.6f}"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: HEIGHT ABOVE GROUND
    # --------------------------------------------------------------

    print("\nHeight above estimated ground:")

    print(
        f"Minimum height : "
        f"{heightAboveGround.min():.2f} m"
    )

    print(
        f"Median height  : "
        f"{np.median(heightAboveGround):.2f} m"
    )

    print(
        f"Maximum height : "
        f"{heightAboveGround.max():.2f} m"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: OBSTACLE CANDIDATES
    # --------------------------------------------------------------

    print("\nObstacle candidates:")

    print(
        f"Minimum obstacle height : "
        f"{MIN_OBSTACLE_HEIGHT_M:.2f} m"
    )

    print(
        f"Candidate points        : "
        f"{xObstacle.shape[0]}"
    )

    # --------------------------------------------------------------
    # DIAGNOSTICS: OBSTACLE BEV
    # --------------------------------------------------------------

    print("\nObstacle BEV:")

    print(
        f"Grid shape       : "
        f"{obstacleDensity.shape}"
    )

    print(
        f"Occupied cells   : "
        f"{(obstacleDensity > 0).sum()}"
    )

    # This should equal:
    #
    #     xObstacle.shape[0]
    #
    # because every obstacle candidate is rasterized exactly once.
    print(
        f"Obstacle returns : "
        f"{obstacleDensity.sum():.0f}"
    )

    print(
        f"Maximum density  : "
        f"{obstacleDensity.max():.0f} points/cell"
    )

    # --------------------------------------------------------------
    # CALCULATE CAR POSITION IN THE BEV GRID
    # --------------------------------------------------------------

    # The car/LiDAR coordinate-system origin is:
    #
    #     x = 0
    #     y = 0
    carXCell = int(
        (0.0 - BEV_X_MIN_M)
        / BEV_RESOLUTION_M
    )

    carYCell = int(
        (0.0 - BEV_Y_MIN_M)
        / BEV_RESOLUTION_M
    )

    carRow = bevHeight - 1 - carXCell
    carColumn = bevWidth - 1 - carYCell

    # --------------------------------------------------------------
    # CREATE SIDE-BY-SIDE VISUALIZATION
    # --------------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 10)
    )

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
        s=100,
        label="Car"
    )

    axes[0].legend()

    # --------------------------------------------------------------
    # RIGHT: GROUND-FILTERED OBSTACLE CANDIDATES
    # --------------------------------------------------------------

    # NOTE:
    #
    # This currently contains points selected ONLY by:
    #
    #     heightAboveGround > 0.15 m
    #
    # The experimental car-footprint removal is NOT being applied.
    axes[1].imshow(
        obstacleDensityDisplay
    )

    axes[1].set_title(
        "Ground-Filtered Obstacle Candidates"
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
        s=100,
        label="Car"
    )

    axes[1].legend()

    figure.suptitle(
        "NavFusion - LiDAR Ground Filtering"
    )

    figure.tight_layout()

    # --------------------------------------------------------------
    # OLD SINGLE-PLOT VISUALIZATION
    # --------------------------------------------------------------

    # This was the original single density-BEV visualization.
    #
    # It remains commented for development history because our new
    # figure contains:
    #
    #     axes[0]
    #     axes[1]
    #
    # rather than one variable named axis.
    #
    # axis.imshow(
    #     bevDensityDisplay
    # )
    #
    # axis.set_title(
    #     "NavFusion - Lidar Density BEV"
    # )
    #
    # axis.set_xlabel(
    #     "Left / Right"
    # )
    #
    # axis.set_ylabel(
    #     "forward / backward"
    # )
    #
    # axis.scatter(
    #     carColumn,
    #     carRow,
    #     marker="x",
    #     s=100,
    #     label="Car"
    # )
    #
    # axis.legend()
    #
    # figure.tight_layout()

    # --------------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------------

    # __file__:
    #
    #     ...\Code\scripts\03_generate_bev.py
    #
    # parents[2]:
    #
    #     ...\NavFusion
    projectRoot = Path(__file__).resolve().parents[2]

    outputDirectory = projectRoot / "Outputs"

    outputDirectory.mkdir(
        parents=True,
        exist_ok=True
    )

    outputPath = (
        outputDirectory
        / "scene_0_lidar_ground_filter_comparison.png"
    )

    figure.savefig(
        outputPath,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"\nBEV saved to:\n"
        f"{outputPath}"
    )

    # plt.show() is blocking.
    #
    # CMD becomes available again after the figure window is closed.
    plt.show()


if __name__ == "__main__":
    main()