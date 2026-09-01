import numpy as np


def filterPointsToBev(
    lidarPointCloud,
    bevConfig
):
    """
    Keep LiDAR points inside the configured metric BEV region.

    LiDAR sensor coordinates:

        +x = forward
        +y = left
        +z = up
    """

    points = lidarPointCloud.points

    x = points[0, :]
    y = points[1, :]
    z = points[2, :]
    intensity = points[3, :]

    bevMask = (
        (x >= bevConfig["xMinM"])
        & (x < bevConfig["xMaxM"])
        & (y >= bevConfig["yMinM"])
        & (y < bevConfig["yMaxM"])
    )

    return {
        "bevMask": bevMask,
        "x": x[bevMask],
        "y": y[bevMask],
        "z": z[bevMask],
        "intensity": intensity[bevMask]
    }


def selectGroundCandidates(
    z,
    lowerPercentile,
    upperPercentile
):
    """
    Select low-height points as possible road/ground points.

    This is only candidate selection. RANSAC decides which
    candidate points actually belong to the dominant ground plane.
    """

    groundLowerZ = np.percentile(
        z,
        lowerPercentile
    )

    groundUpperZ = np.percentile(
        z,
        upperPercentile
    )

    groundCandidateMask = (
        (z >= groundLowerZ)
        & (z <= groundUpperZ)
    )

    return (
        groundCandidateMask,
        groundLowerZ,
        groundUpperZ
    )


def calculatePlaneDistances(
    x,
    y,
    z,
    planeCoefficients
):
    """
    Calculate perpendicular point-to-plane distance.

    Ground model:

        z = a*x + b*y + c

    Rearranged:

        a*x + b*y - z + c = 0

    Distance:

                  |a*x + b*y - z + c|
        d = --------------------------------
               sqrt(a^2 + b^2 + 1)
    """

    a, b, c = planeCoefficients

    numerator = np.abs(
        (a * x)
        + (b * y)
        - z
        + c
    )

    denominator = np.sqrt(
        (a ** 2)
        + (b ** 2)
        + 1.0
    )

    distances = (
        numerator
        / denominator
    )

    return distances


def fitPlaneLeastSquares(
    x,
    y,
    z
):
    """
    Fit:

        z = a*x + b*y + c

    using least squares.
    """

    groundMatrix = np.column_stack(
        (
            x,
            y,
            np.ones_like(x)
        )
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        groundMatrix,
        z,
        rcond=None
    )

    return coefficients


def estimateGroundPlaneRansac(
    x,
    y,
    z,
    bevConfig
):
    """
    Estimate the dominant ground plane using RANSAC.

    Process:

        1. Select plausible low-z ground points.
        2. Randomly choose 3 points.
        3. Fit a plane through the 3 points.
        4. Measure candidate distances to that plane.
        5. Count RANSAC inliers.
        6. Keep the best plane.
        7. Refit using least squares over all best inliers.
    """

    (
        groundCandidateMask,
        groundLowerZ,
        groundUpperZ
    ) = selectGroundCandidates(
        z,
        bevConfig["groundLowerPercentile"],
        bevConfig["groundUpperPercentile"]
    )

    xGround = x[
        groundCandidateMask
    ]

    yGround = y[
        groundCandidateMask
    ]

    zGround = z[
        groundCandidateMask
    ]

    groundCandidateCount = xGround.size

    if groundCandidateCount < 3:
        raise ValueError(
            "Not enough ground candidates to estimate a plane."
        )

    randomGenerator = np.random.default_rng(
        bevConfig["ransacRandomSeed"]
    )

    bestInlierMask = None
    bestInlierCount = 0
    bestMeanDistance = np.inf

    for _ in range(
        bevConfig["ransacIterations"]
    ):

        sampleIndices = randomGenerator.choice(
            groundCandidateCount,
            size=3,
            replace=False
        )

        sampleX = xGround[
            sampleIndices
        ]

        sampleY = yGround[
            sampleIndices
        ]

        sampleZ = zGround[
            sampleIndices
        ]

        sampleMatrix = np.column_stack(
            (
                sampleX,
                sampleY,
                np.ones(3)
            )
        )

        # Three collinear points do not give a reliable plane.
        if np.linalg.matrix_rank(
            sampleMatrix
        ) < 3:
            continue

        planeCoefficients = np.linalg.solve(
            sampleMatrix,
            sampleZ
        )

        distances = calculatePlaneDistances(
            xGround,
            yGround,
            zGround,
            planeCoefficients
        )

        inlierMask = (
            distances
            <= bevConfig["ransacDistanceThresholdM"]
        )

        inlierCount = int(
            np.sum(
                inlierMask
            )
        )

        if inlierCount == 0:
            continue

        meanInlierDistance = float(
            np.mean(
                distances[
                    inlierMask
                ]
            )
        )

        if (
            inlierCount > bestInlierCount
            or (
                inlierCount == bestInlierCount
                and meanInlierDistance < bestMeanDistance
            )
        ):
            bestInlierMask = inlierMask
            bestInlierCount = inlierCount
            bestMeanDistance = meanInlierDistance

    if bestInlierMask is None:
        raise RuntimeError(
            "RANSAC could not find a valid ground plane."
        )

    # Refine the best RANSAC hypothesis using all its inliers.
    finalPlaneCoefficients = fitPlaneLeastSquares(
        xGround[
            bestInlierMask
        ],
        yGround[
            bestInlierMask
        ],
        zGround[
            bestInlierMask
        ]
    )

    return {
        "planeCoefficients": finalPlaneCoefficients,

        "groundCandidateMask": groundCandidateMask,

        "groundLowerZ": groundLowerZ,

        "groundUpperZ": groundUpperZ,

        "groundCandidateCount": int(
            groundCandidateCount
        ),

        "ransacInlierCount": int(
            bestInlierCount
        ),

        "ransacMeanInlierDistanceM": float(
            bestMeanDistance
        )
    }


def calculateHeightAboveGround(
    x,
    y,
    z,
    planeCoefficients
):
    """
    Calculate each LiDAR point's height above the fitted ground.

    Ground:

        groundZ = a*x + b*y + c

    Height:

        heightAboveGround = z - groundZ
    """

    a, b, c = planeCoefficients

    estimatedGroundZ = (
        (a * x)
        + (b * y)
        + c
    )

    heightAboveGround = (
        z
        - estimatedGroundZ
    )

    return (
        estimatedGroundZ,
        heightAboveGround
    )


def createSelfReturnMask(
    x,
    y,
    z,
    bevConfig
):
    """
    Identify known near-sensor LiDAR returns produced by the
    car / LiDAR mounting structure.

    The region is defined in the raw LIDAR_TOP coordinate frame.

    A point is inside the self-return region only when all three
    coordinate conditions are satisfied simultaneously:

        xMin <= x <= xMax
        yMin <= y <= yMax
        zMin <= z <= zMax

    The original LiDAR data is not deleted.

    The mask is used only to prevent known car returns from
    becoming external obstacles.
    """

    selfReturnMask = (
        (x >= bevConfig["selfReturnXMinM"])
        & (x <= bevConfig["selfReturnXMaxM"])
        & (y >= bevConfig["selfReturnYMinM"])
        & (y <= bevConfig["selfReturnYMaxM"])
        & (z >= bevConfig["selfReturnZMinM"])
        & (z <= bevConfig["selfReturnZMaxM"])
    )

    return selfReturnMask


def calculateBevGridSize(
    bevConfig
):
    """
    Calculate the number of rows and columns in the BEV grid.

    Example:

        x range = 50 - (-20)
                = 70 m

        resolution = 0.20 m/cell

        rows = 70 / 0.20
             = 350
    """

    bevHeight = int(
        (
            bevConfig["xMaxM"]
            - bevConfig["xMinM"]
        )
        / bevConfig["resolutionM"]
    )

    bevWidth = int(
        (
            bevConfig["yMaxM"]
            - bevConfig["yMinM"]
        )
        / bevConfig["resolutionM"]
    )

    return bevHeight, bevWidth


def convertMetersToBevGrid(
    x,
    y,
    bevConfig
):
    """
    Convert metric LiDAR x/y positions into BEV row/column indices.
    """

    bevHeight, bevWidth = calculateBevGridSize(
        bevConfig
    )

    xCell = np.floor(
        (
            x
            - bevConfig["xMinM"]
        )
        / bevConfig["resolutionM"]
    ).astype(
        np.int32
    )

    yCell = np.floor(
        (
            y
            - bevConfig["yMinM"]
        )
        / bevConfig["resolutionM"]
    ).astype(
        np.int32
    )

    # Image row 0 is at the top.
    # Invert x so +x forward appears toward the top.
    bevRow = (
        bevHeight
        - 1
        - xCell
    )

    # +y is left in the LiDAR frame.
    # Smaller image columns are left, so invert y.
    bevColumn = (
        bevWidth
        - 1
        - yCell
    )

    return (
        bevRow,
        bevColumn,
        bevHeight,
        bevWidth
    )


def createDensityGrid(
    bevRow,
    bevColumn,
    bevHeight,
    bevWidth
):
    """
    Count LiDAR returns in each BEV grid cell.
    """

    densityGrid = np.zeros(
        (
            bevHeight,
            bevWidth
        ),
        dtype=np.float32
    )

    np.add.at(
        densityGrid,
        (
            bevRow,
            bevColumn
        ),
        1
    )

    return densityGrid


def createDisplayDensity(
    densityGrid
):
    """
    Log-compress and normalize raw LiDAR density for visualization.
    """

    displayDensity = np.log1p(
        densityGrid
    )

    maximumDensity = np.max(
        displayDensity
    )

    if maximumDensity > 0:
        displayDensity = (
            displayDensity
            / maximumDensity
        )

    return displayDensity


def calculateCarGridPosition(
    bevConfig
):
    """
    Find the BEV cell corresponding to LiDAR origin (0, 0).
    """

    carX = np.array(
        [0.0]
    )

    carY = np.array(
        [0.0]
    )

    (
        carRow,
        carColumn,
        _,
        _
    ) = convertMetersToBevGrid(
        carX,
        carY,
        bevConfig
    )

    return (
        int(
            carRow[0]
        ),
        int(
            carColumn[0]
        )
    )


def processLidarBev(
    frameData,
    bevConfig
):
    """
    Process the exact LiDAR point cloud supplied by run_navfusion.py.

    Flow:

        LiDAR
          ↓
        BEV spatial crop
          ↓
        RANSAC ground estimation
          ↓
        height above ground
          ↓
        obstacle candidates
          ↓
        self-return mask
          ↓
        clean external obstacles
          ↓
        BEV density rasters
    """

    lidarPointCloud = frameData[
        "lidarPointCloud"
    ]

    # ----------------------------------------------------------
    # Crop to the configured BEV region
    # ----------------------------------------------------------

    filteredPoints = filterPointsToBev(
        lidarPointCloud,
        bevConfig
    )

    x = filteredPoints[
        "x"
    ]

    y = filteredPoints[
        "y"
    ]

    z = filteredPoints[
        "z"
    ]

    intensity = filteredPoints[
        "intensity"
    ]

    # ----------------------------------------------------------
    # Estimate the road/ground plane with RANSAC
    # ----------------------------------------------------------

    groundResult = estimateGroundPlaneRansac(
        x,
        y,
        z,
        bevConfig
    )

    # ----------------------------------------------------------
    # Calculate height above estimated ground
    # ----------------------------------------------------------

    (
        estimatedGroundZ,
        heightAboveGround
    ) = calculateHeightAboveGround(
        x,
        y,
        z,
        groundResult[
            "planeCoefficients"
        ]
    )

    # ----------------------------------------------------------
    # Initial obstacle candidates
    # ----------------------------------------------------------

    # Any return more than the configured height above the
    # estimated ground becomes an initial obstacle candidate.
    obstacleCandidateMask = (
        heightAboveGround
        > bevConfig[
            "minimumObstacleHeightM"
        ]
    )

    # ----------------------------------------------------------
    # Detect known car / sensor self returns
    # ----------------------------------------------------------

    selfReturnMask = createSelfReturnMask(
        x,
        y,
        z,
        bevConfig
    )

    # ----------------------------------------------------------
    # Clean external obstacle mask
    # ----------------------------------------------------------

    # ~ means Boolean NOT.
    #
    # Therefore:
    #
    # obstacle candidate
    # AND
    # NOT self return
    #
    # becomes a clean external obstacle.
    obstacleMask = (
        obstacleCandidateMask
        & (~selfReturnMask)
    )

    # ----------------------------------------------------------
    # Convert every BEV-region point into grid coordinates
    # ----------------------------------------------------------

    (
        bevRow,
        bevColumn,
        bevHeight,
        bevWidth
    ) = convertMetersToBevGrid(
        x,
        y,
        bevConfig
    )

    # ----------------------------------------------------------
    # Raw LiDAR density
    # ----------------------------------------------------------

    bevDensity = createDensityGrid(
        bevRow,
        bevColumn,
        bevHeight,
        bevWidth
    )

    # ----------------------------------------------------------
    # RANSAC obstacles BEFORE self-return filtering
    # ----------------------------------------------------------

    obstacleCandidateRow = bevRow[
        obstacleCandidateMask
    ]

    obstacleCandidateColumn = bevColumn[
        obstacleCandidateMask
    ]

    obstacleCandidateDensity = createDensityGrid(
        obstacleCandidateRow,
        obstacleCandidateColumn,
        bevHeight,
        bevWidth
    )

    # ----------------------------------------------------------
    # RANSAC obstacles AFTER self-return filtering
    # ----------------------------------------------------------

    obstacleRow = bevRow[
        obstacleMask
    ]

    obstacleColumn = bevColumn[
        obstacleMask
    ]

    obstacleDensity = createDensityGrid(
        obstacleRow,
        obstacleColumn,
        bevHeight,
        bevWidth
    )

    # ----------------------------------------------------------
    # Display-friendly density maps
    # ----------------------------------------------------------

    bevDisplayDensity = createDisplayDensity(
        bevDensity
    )

    obstacleCandidateDisplayDensity = createDisplayDensity(
        obstacleCandidateDensity
    )

    obstacleDisplayDensity = createDisplayDensity(
        obstacleDensity
    )

    # ----------------------------------------------------------
    # Car position in the BEV grid
    # ----------------------------------------------------------

    carRow, carColumn = calculateCarGridPosition(
        bevConfig
    )

    # ----------------------------------------------------------
    # Return every intermediate result
    # ----------------------------------------------------------

    return {
        "x": x,

        "y": y,

        "z": z,

        "intensity": intensity,

        "estimatedGroundZ": estimatedGroundZ,

        "heightAboveGround": heightAboveGround,

        "obstacleCandidateMask": obstacleCandidateMask,

        "selfReturnMask": selfReturnMask,

        "obstacleMask": obstacleMask,

        "bevRow": bevRow,

        "bevColumn": bevColumn,

        "bevHeight": bevHeight,

        "bevWidth": bevWidth,

        "bevDensity": bevDensity,

        "obstacleCandidateDensity": obstacleCandidateDensity,

        "obstacleDensity": obstacleDensity,

        "bevDisplayDensity": bevDisplayDensity,

        "obstacleCandidateDisplayDensity": (
            obstacleCandidateDisplayDensity
        ),

        "obstacleDisplayDensity": obstacleDisplayDensity,

        "carRow": carRow,

        "carColumn": carColumn,

        **groundResult
    }