import numpy as np
from pyquaternion import Quaternion


def transformLidarToCar(
    lidarPointCloud,
    lidarCalibration
):
    """
    Transform the complete LIDAR_TOP point cloud from the LiDAR
    sensor coordinate frame into the car coordinate frame.

    nuScenes calibrated_sensor provides:

        rotation
            sensor -> car rotation

        translation
            sensor origin expressed in the car frame

    For every LiDAR point:

        pCar = R * pLidar + t

    where:

        pLidar shape = (3, N)
        R shape      = (3, 3)
        t shape      = (3, 1)
        pCar shape   = (3, N)

    The original LiDAR point cloud is not modified.
    """

    points = lidarPointCloud.points

    # Original xyz coordinates in the LIDAR_TOP sensor frame.
    lidarPoints = points[
        0:3,
        :
    ]

    # The fourth LiDAR row contains intensity.
    intensity = points[
        3,
        :
    ]

    # nuScenes stores rotation as a quaternion:
    #
    # [w, x, y, z]
    #
    # Quaternion(...).rotation_matrix converts it into a
    # 3 x 3 rotation matrix.
    rotationMatrix = Quaternion(
        lidarCalibration[
            "rotation"
        ]
    ).rotation_matrix

    # Convert:
    #
    # [tx, ty, tz]
    #
    # into:
    #
    # [[tx],
    #  [ty],
    #  [tz]]
    #
    # The (3, 1) shape lets NumPy add the same translation to
    # every one of the N LiDAR points.
    translationVector = np.array(
        lidarCalibration[
            "translation"
        ],
        dtype=np.float64
    ).reshape(
        3,
        1
    )

    # ----------------------------------------------------------
    # LiDAR sensor frame -> car frame
    # ----------------------------------------------------------
    #
    # Matrix multiplication:
    #
    # (3 x 3) @ (3 x N)
    #          =
    #        (3 x N)
    #
    # Then NumPy broadcasts the (3 x 1) translation across all
    # N point columns.
    carPoints = (
        rotationMatrix
        @ lidarPoints
        + translationVector
    )

    return {
        "lidarPoints": lidarPoints,

        "carPoints": carPoints,

        "intensity": intensity,

        "rotationMatrix": rotationMatrix,

        "translationVector": translationVector
    }


def filterPointsToBev(
    lidarPointCloud,
    lidarCalibration,
    bevConfig
):
    """
    Transform LiDAR points into the car frame and retain points
    inside the configured car-frame BEV region.

    Car coordinate convention:

        +x = forward
        +y = left
        +z = up

    The original sensor-frame coordinates are retained alongside
    the car-frame coordinates so the verified self-return mask can
    continue operating in the raw LIDAR_TOP frame.
    """

    transformedPoints = transformLidarToCar(
        lidarPointCloud,
        lidarCalibration
    )

    lidarPoints = transformedPoints[
        "lidarPoints"
    ]

    carPoints = transformedPoints[
        "carPoints"
    ]

    intensity = transformedPoints[
        "intensity"
    ]

    # ----------------------------------------------------------
    # Original LIDAR_TOP sensor coordinates
    # ----------------------------------------------------------

    lidarX = lidarPoints[
        0,
        :
    ]

    lidarY = lidarPoints[
        1,
        :
    ]

    lidarZ = lidarPoints[
        2,
        :
    ]

    # ----------------------------------------------------------
    # Car-frame coordinates
    # ----------------------------------------------------------

    carX = carPoints[
        0,
        :
    ]

    carY = carPoints[
        1,
        :
    ]

    carZ = carPoints[
        2,
        :
    ]

    # ----------------------------------------------------------
    # BEV crop in the CAR FRAME
    # ----------------------------------------------------------
    #
    # Use half-open upper bounds:
    #
    # xMin <= x < xMax
    # yMin <= y < yMax
    #
    # This prevents a point exactly at xMax/yMax from generating
    # an out-of-range grid index.
    bevMask = (
        (carX >= bevConfig["xMinM"])
        & (carX < bevConfig["xMaxM"])
        & (carY >= bevConfig["yMinM"])
        & (carY < bevConfig["yMaxM"])
    )

    return {
        "bevMask": bevMask,

        # Car-frame coordinates used for RANSAC and BEV.
        "x": carX[
            bevMask
        ],

        "y": carY[
            bevMask
        ],

        "z": carZ[
            bevMask
        ],

        # Original sensor coordinates remain aligned with x/y/z.
        "lidarX": lidarX[
            bevMask
        ],

        "lidarY": lidarY[
            bevMask
        ],

        "lidarZ": lidarZ[
            bevMask
        ],

        "intensity": intensity[
            bevMask
        ],

        "lidarToCarRotationMatrix": transformedPoints[
            "rotationMatrix"
        ],

        "lidarToCarTranslationVector": transformedPoints[
            "translationVector"
        ]
    }


def selectGroundCandidates(
    z,
    lowerPercentile,
    upperPercentile
):
    """
    Select low-height points as possible road/ground points.

    z is now expressed in the car coordinate frame.

    This step only produces plausible ground candidates.
    RANSAC decides which candidates actually fit the dominant
    road plane.
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
    Estimate the dominant ground plane in the car frame using
    RANSAC.

    Process:

        1. Select plausible low-z ground points.
        2. Randomly choose 3 points.
        3. Fit a plane through those 3 points.
        4. Measure candidate distances to the plane.
        5. Count points inside the RANSAC distance threshold.
        6. Keep the best hypothesis.
        7. Refit that plane using least squares over its inliers.
    """

    (
        groundCandidateMask,
        groundLowerZ,
        groundUpperZ
    ) = selectGroundCandidates(
        z,
        bevConfig[
            "groundLowerPercentile"
        ],
        bevConfig[
            "groundUpperPercentile"
        ]
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
        bevConfig[
            "ransacRandomSeed"
        ]
    )

    bestInlierMask = None

    bestInlierCount = 0

    bestMeanDistance = np.inf

    for _ in range(
        bevConfig[
            "ransacIterations"
        ]
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

        # Three linearly dependent points cannot uniquely
        # determine the plane coefficients.
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
            <= bevConfig[
                "ransacDistanceThresholdM"
            ]
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

        # First maximize inlier count.
        #
        # If two models contain the same number of inliers,
        # choose the one with the smaller mean inlier error.
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

    # RANSAC selects the correct subset of ground points.
    #
    # Least squares then computes a more accurate final plane
    # using all inliers belonging to the best hypothesis.
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
    Calculate each car-frame LiDAR point's height above the fitted
    ground plane.

    Ground:

        estimatedGroundZ = a*x + b*y + c

    Height:

        heightAboveGround = z - estimatedGroundZ
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
    lidarX,
    lidarY,
    lidarZ,
    bevConfig
):
    """
    Identify known near-sensor returns using the ORIGINAL
    LIDAR_TOP sensor coordinate frame.

    This is intentional.

    The self-return bounds were experimentally verified in the
    raw LiDAR frame, so converting these points into the car frame
    before applying those bounds would change the meaning of the
    calibrated filtering region.

    A point is a self return when:

        xMin <= lidarX <= xMax
        yMin <= lidarY <= yMax
        zMin <= lidarZ <= zMax
    """

    selfReturnMask = (
        (
            lidarX
            >= bevConfig[
                "selfReturnXMinM"
            ]
        )
        & (
            lidarX
            <= bevConfig[
                "selfReturnXMaxM"
            ]
        )
        & (
            lidarY
            >= bevConfig[
                "selfReturnYMinM"
            ]
        )
        & (
            lidarY
            <= bevConfig[
                "selfReturnYMaxM"
            ]
        )
        & (
            lidarZ
            >= bevConfig[
                "selfReturnZMinM"
            ]
        )
        & (
            lidarZ
            <= bevConfig[
                "selfReturnZMaxM"
            ]
        )
    )

    return selfReturnMask


def calculateBevGridSize(
    bevConfig
):
    """
    Calculate BEV rows and columns.

    Current configuration:

        x:
            -20 m to +50 m
            = 70 m

        y:
            -25 m to +25 m
            = 50 m

        resolution:
            0.20 m/cell

    Therefore:

        bevHeight = 70 / 0.20
                  = 350 rows

        bevWidth = 50 / 0.20
                 = 250 columns
    """

    bevHeight = int(
        (
            bevConfig[
                "xMaxM"
            ]
            - bevConfig[
                "xMinM"
            ]
        )
        / bevConfig[
            "resolutionM"
        ]
    )

    bevWidth = int(
        (
            bevConfig[
                "yMaxM"
            ]
            - bevConfig[
                "yMinM"
            ]
        )
        / bevConfig[
            "resolutionM"
        ]
    )

    return (
        bevHeight,
        bevWidth
    )


def convertMetersToBevGrid(
    x,
    y,
    bevConfig
):
    """
    Convert CAR-FRAME metric x/y coordinates into BEV
    row/column indices.

    First:

        xCell = floor(
            (x - xMin)
            / resolution
        )

        yCell = floor(
            (y - yMin)
            / resolution
        )

    Then convert physical grid coordinates to image coordinates.
    """

    bevHeight, bevWidth = calculateBevGridSize(
        bevConfig
    )

    xCell = np.floor(
        (
            x
            - bevConfig[
                "xMinM"
            ]
        )
        / bevConfig[
            "resolutionM"
        ]
    ).astype(
        np.int32
    )

    yCell = np.floor(
        (
            y
            - bevConfig[
                "yMinM"
            ]
        )
        / bevConfig[
            "resolutionM"
        ]
    ).astype(
        np.int32
    )

    # Image row zero is at the top.
    #
    # Therefore invert x so +x / forward appears toward the
    # top of the BEV.
    bevRow = (
        bevHeight
        - 1
        - xCell
    )

    # In the car frame:
    #
    # +y = left.
    #
    # Smaller image columns appear on the left, so invert y too.
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
    Count points falling into each BEV cell.
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
    Create a log-compressed normalized density image for
    visualization.

    This does not change the raw density used in the
    multi-channel BEV.
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
    Convert the actual car-frame origin:

        xCar = 0
        yCar = 0

    into its BEV row/column.

    Now that the BEV itself is in the car frame, this is truly the
    car origin rather than the LIDAR_TOP sensor origin.
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


def createMaxHeightGrid(
    bevRow,
    bevColumn,
    heightAboveGround,
    bevHeight,
    bevWidth
):
    """
    Store the maximum ground-relative obstacle height observed in
    each BEV cell.
    """

    maxHeightGrid = np.zeros(
        (
            bevHeight,
            bevWidth
        ),
        dtype=np.float32
    )

    np.maximum.at(
        maxHeightGrid,
        (
            bevRow,
            bevColumn
        ),
        heightAboveGround
    )

    return maxHeightGrid


def createMeanIntensityGrid(
    bevRow,
    bevColumn,
    intensity,
    bevHeight,
    bevWidth
):
    """
    Calculate mean LiDAR intensity in every occupied BEV cell.

    For each cell:

                    sum of intensities
        mean = -----------------------------
               number of LiDAR points
    """

    intensitySumGrid = np.zeros(
        (
            bevHeight,
            bevWidth
        ),
        dtype=np.float32
    )

    intensityCountGrid = np.zeros(
        (
            bevHeight,
            bevWidth
        ),
        dtype=np.float32
    )

    np.add.at(
        intensitySumGrid,
        (
            bevRow,
            bevColumn
        ),
        intensity
    )

    np.add.at(
        intensityCountGrid,
        (
            bevRow,
            bevColumn
        ),
        1
    )

    meanIntensityGrid = np.zeros(
        (
            bevHeight,
            bevWidth
        ),
        dtype=np.float32
    )

    occupiedIntensityMask = (
        intensityCountGrid > 0
    )

    meanIntensityGrid[
        occupiedIntensityMask
    ] = (
        intensitySumGrid[
            occupiedIntensityMask
        ]
        / intensityCountGrid[
            occupiedIntensityMask
        ]
    )

    return meanIntensityGrid


def createOccupancyGrid(
    densityGrid
):
    """
    Create binary external-obstacle occupancy.

    Each cell is:

        0 = no clean obstacle points
        1 = one or more clean obstacle points
    """

    occupancyGrid = (
        densityGrid > 0
    ).astype(
        np.float32
    )

    return occupancyGrid


def createMultiChannelBev(
    densityGrid,
    maxHeightGrid,
    meanIntensityGrid,
    occupancyGrid
):
    """
    Stack the four spatially aligned LiDAR BEV channels.

    Channel 0:
        clean obstacle density

    Channel 1:
        maximum height above ground

    Channel 2:
        mean LiDAR intensity

    Channel 3:
        binary occupancy

    Output:

        (4, bevHeight, bevWidth)

    Current configuration:

        (4, 350, 250)
    """

    multiChannelBev = np.stack(
        (
            densityGrid,
            maxHeightGrid,
            meanIntensityGrid,
            occupancyGrid
        ),
        axis=0
    )

    return multiChannelBev


def processLidarBev(
    frameData,
    bevConfig
):
    """
    Generate a car-frame multi-channel LiDAR BEV.

    Processing flow:

        raw LIDAR_TOP
               ↓
        LiDAR -> car calibration
               ↓
        car-frame point cloud
               ↓
        car-frame BEV crop
               ↓
        RANSAC ground estimation
               ↓
        height above ground
               ↓
        obstacle candidates
               ↓
        raw-sensor-frame self-return mask
               ↓
        clean external obstacles
               ↓
        car-frame metric rasterization
               ↓
        density / height / intensity / occupancy
               ↓
        multi-channel BEV
    """

    lidarPointCloud = frameData[
        "lidarPointCloud"
    ]

    lidarCalibration = frameData[
        "lidarCalibration"
    ]

    # ----------------------------------------------------------
    # Transform LiDAR -> car and crop in the car frame
    # ----------------------------------------------------------

    filteredPoints = filterPointsToBev(
        lidarPointCloud,
        lidarCalibration,
        bevConfig
    )

    # ----------------------------------------------------------
    # Car-frame point coordinates
    # ----------------------------------------------------------

    x = filteredPoints[
        "x"
    ]

    y = filteredPoints[
        "y"
    ]

    z = filteredPoints[
        "z"
    ]

    # ----------------------------------------------------------
    # Matching original LIDAR_TOP coordinates
    # ----------------------------------------------------------
    #
    # These arrays have exactly the same length/order as x/y/z
    # because the SAME bevMask was applied to both coordinate
    # representations.
    lidarX = filteredPoints[
        "lidarX"
    ]

    lidarY = filteredPoints[
        "lidarY"
    ]

    lidarZ = filteredPoints[
        "lidarZ"
    ]

    intensity = filteredPoints[
        "intensity"
    ]

    # ----------------------------------------------------------
    # Estimate ground in the CAR FRAME
    # ----------------------------------------------------------

    groundResult = estimateGroundPlaneRansac(
        x,
        y,
        z,
        bevConfig
    )

    # ----------------------------------------------------------
    # Calculate car-frame height above ground
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

    obstacleCandidateMask = (
        heightAboveGround
        > bevConfig[
            "minimumObstacleHeightM"
        ]
    )

    # ----------------------------------------------------------
    # Self-return filtering in ORIGINAL SENSOR FRAME
    # ----------------------------------------------------------
    #
    # Do not use car-frame x/y/z here.
    #
    # The current self-return bounds were validated using
    # LIDAR_TOP coordinates.
    selfReturnMask = createSelfReturnMask(
        lidarX,
        lidarY,
        lidarZ,
        bevConfig
    )

    # ----------------------------------------------------------
    # Clean external obstacle mask
    # ----------------------------------------------------------

    obstacleMask = (
        obstacleCandidateMask
        & (~selfReturnMask)
    )

    # ----------------------------------------------------------
    # Convert CAR-FRAME x/y into the metric BEV grid
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
    # Raw car-frame LiDAR density
    # ----------------------------------------------------------

    bevDensity = createDensityGrid(
        bevRow,
        bevColumn,
        bevHeight,
        bevWidth
    )

    # ----------------------------------------------------------
    # Obstacles before self filtering
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
    # Clean obstacle locations
    # ----------------------------------------------------------

    obstacleRow = bevRow[
        obstacleMask
    ]

    obstacleColumn = bevColumn[
        obstacleMask
    ]

    obstacleHeightAboveGround = heightAboveGround[
        obstacleMask
    ]

    obstacleIntensity = intensity[
        obstacleMask
    ]

    # ----------------------------------------------------------
    # Channel 0: obstacle density
    # ----------------------------------------------------------

    obstacleDensity = createDensityGrid(
        obstacleRow,
        obstacleColumn,
        bevHeight,
        bevWidth
    )

    # ----------------------------------------------------------
    # Channel 1: maximum height above ground
    # ----------------------------------------------------------

    maxHeightGrid = createMaxHeightGrid(
        obstacleRow,
        obstacleColumn,
        obstacleHeightAboveGround,
        bevHeight,
        bevWidth
    )

    # ----------------------------------------------------------
    # Channel 2: mean intensity
    # ----------------------------------------------------------

    meanIntensityGrid = createMeanIntensityGrid(
        obstacleRow,
        obstacleColumn,
        obstacleIntensity,
        bevHeight,
        bevWidth
    )

    # ----------------------------------------------------------
    # Channel 3: binary occupancy
    # ----------------------------------------------------------

    occupancyGrid = createOccupancyGrid(
        obstacleDensity
    )

    # ----------------------------------------------------------
    # Four-channel car-frame BEV tensor
    # ----------------------------------------------------------

    multiChannelBev = createMultiChannelBev(
        obstacleDensity,
        maxHeightGrid,
        meanIntensityGrid,
        occupancyGrid
    )

    # ----------------------------------------------------------
    # Display-only density maps
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
    # Actual car origin in car-frame BEV
    # ----------------------------------------------------------

    carRow, carColumn = calculateCarGridPosition(
        bevConfig
    )

    # ----------------------------------------------------------
    # LiDAR sensor position in the CAR FRAME
    # ----------------------------------------------------------
    #
    # The calibrated_sensor translation is the location of the
    # LIDAR_TOP origin relative to the car origin.
    lidarSensorPositionCarM = (
        filteredPoints[
            "lidarToCarTranslationVector"
        ]
        .reshape(
            3
        )
    )

    # ----------------------------------------------------------
    # Return results
    # ----------------------------------------------------------

    return {
        # These are now CAR-FRAME coordinates.
        "x": x,

        "y": y,

        "z": z,

        # Matching raw LIDAR_TOP coordinates.
        "lidarX": lidarX,

        "lidarY": lidarY,

        "lidarZ": lidarZ,

        "intensity": intensity,

        "lidarToCarRotationMatrix": filteredPoints[
            "lidarToCarRotationMatrix"
        ],

        "lidarToCarTranslationVector": filteredPoints[
            "lidarToCarTranslationVector"
        ],

        "lidarSensorPositionCarM": lidarSensorPositionCarM,

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

        "maxHeightGrid": maxHeightGrid,

        "meanIntensityGrid": meanIntensityGrid,

        "occupancyGrid": occupancyGrid,

        "multiChannelBev": multiChannelBev,

        "bevDisplayDensity": bevDisplayDensity,

        "obstacleCandidateDisplayDensity": (
            obstacleCandidateDisplayDensity
        ),

        "obstacleDisplayDensity": obstacleDisplayDensity,

        "carRow": carRow,

        "carColumn": carColumn,

        **groundResult
    }