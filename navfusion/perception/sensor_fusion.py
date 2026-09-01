import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def lidarToCar(
    pointCloud,
    lidarCalibration
):
    """
    Transform the complete LiDAR point cloud from the raw
    LIDAR_TOP sensor frame into the car frame.

    Transformation:

        pCar = R * pLidar + t

    where:

        R = LiDAR-to-car rotation
        t = LiDAR position in the car frame
    """

    lidarRotation = Quaternion(
        lidarCalibration[
            "rotation"
        ]
    ).rotation_matrix

    lidarTranslation = np.array(
        lidarCalibration[
            "translation"
        ]
    )

    # rotate() and translate() modify the point cloud in place.
    pointCloud.rotate(
        lidarRotation
    )

    pointCloud.translate(
        lidarTranslation
    )

    return pointCloud


def transformLidarPointsToCar(
    lidarPoints,
    lidarCalibration
):
    """
    Transform selected raw LIDAR_TOP points into the car frame.

    Unlike lidarToCar(), this function operates directly on a
    NumPy array and does not modify the supplied source array.

    Input:

        lidarPoints
            shape = (3, N) or (4, N)

    Transformation:

        pCar = R * pLidar + t

    Shapes:

        R          = (3, 3)
        pLidar     = (3, N)
        translation= (3, 1)
        pCar       = (3, N)
    """

    rotationMatrix = Quaternion(
        lidarCalibration[
            "rotation"
        ]
    ).rotation_matrix

    translationVector = np.array(
        lidarCalibration[
            "translation"
        ],
        dtype=np.float64
    ).reshape(
        3,
        1
    )

    # Keep only xyz.
    #
    # If the supplied array contains LiDAR intensity as row 4,
    # that row is not part of the geometric transformation.
    lidarXyz = lidarPoints[
        0:3,
        :
    ]

    # Matrix multiplication rotates every point:
    #
    # (3 x 3) @ (3 x N)
    #          =
    #        (3 x N)
    #
    # The (3 x 1) translation is then broadcast over all N
    # columns.
    carPoints = (
        rotationMatrix
        @ lidarXyz
        + translationVector
    )

    return carPoints


def carToGlobal(
    pointCloud,
    carPose
):
    """
    Transform car-frame points into the global/world frame.
    """

    carRotation = Quaternion(
        carPose[
            "rotation"
        ]
    ).rotation_matrix

    carTranslation = np.array(
        carPose[
            "translation"
        ]
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
    """
    Transform global-frame points into a car coordinate frame.
    """

    carTranslation = np.array(
        carPose[
            "translation"
        ]
    )

    carRotation = Quaternion(
        carPose[
            "rotation"
        ]
    ).rotation_matrix

    # Forward transformation:
    #
    # pGlobal = R * pCar + t
    #
    # Inverse:
    #
    # pCar = R^T * (pGlobal - t)

    # Undo translation first.
    pointCloud.translate(
        -carTranslation
    )

    # Rotation matrices are orthonormal:
    #
    # R^-1 = R^T
    pointCloud.rotate(
        carRotation.T
    )

    return pointCloud


def carToCamera(
    pointCloud,
    cameraCalibration
):
    """
    Transform car-frame points into the camera coordinate frame.

    nuScenes calibrated_sensor stores:

        camera -> car

    but here we need:

        car -> camera

    so the inverse transformation is applied.
    """

    cameraTranslation = np.array(
        cameraCalibration[
            "translation"
        ]
    )

    cameraRotation = Quaternion(
        cameraCalibration[
            "rotation"
        ]
    ).rotation_matrix

    # Undo camera translation.
    pointCloud.translate(
        -cameraTranslation
    )

    # Undo camera rotation.
    pointCloud.rotate(
        cameraRotation.T
    )

    return pointCloud


def cameraToImage(
    pointCloud,
    cameraCalibration
):
    """
    Project camera-frame 3D points onto the 2D image.

    In the camera frame:

        z = optical-axis depth
    """

    depths = pointCloud.points[
        2,
        :
    ]

    cameraIntrinsic = np.array(
        cameraCalibration[
            "camera_intrinsic"
        ]
    )

    points2d = view_points(
        pointCloud.points[
            :3,
            :
        ],
        cameraIntrinsic,
        normalize=True
    )

    return (
        points2d,
        depths
    )


def projectLidarToCamera(
    frameData
):
    """
    Project the supplied LiDAR cloud into the supplied camera.

    No dataset lookup occurs inside this function.

    Transformation chain:

        LIDAR_TOP
            ↓
        car at LiDAR timestamp
            ↓
        global
            ↓
        car at camera timestamp
            ↓
        camera
            ↓
        image

    The original LiDAR point ordering is preserved throughout
    these transformations.

    Therefore index i continues to refer to the same physical
    LiDAR return in:

        original LiDAR points
        projected 2D points
        camera depths
    """

    # rotate() and translate() modify a cloud in place.
    #
    # Work on a copy so frameData["lidarPointCloud"] remains
    # unchanged.
    pointCloud = LidarPointCloud(
        frameData[
            "lidarPointCloud"
        ].points.copy()
    )

    # ----------------------------------------------------------
    # LIDAR_TOP -> car at LiDAR timestamp
    # ----------------------------------------------------------

    pointCloud = lidarToCar(
        pointCloud,
        frameData[
            "lidarCalibration"
        ]
    )

    # ----------------------------------------------------------
    # Car at LiDAR timestamp -> global
    # ----------------------------------------------------------

    pointCloud = carToGlobal(
        pointCloud,
        frameData[
            "lidarCarPose"
        ]
    )

    # ----------------------------------------------------------
    # Global -> car at camera timestamp
    # ----------------------------------------------------------

    pointCloud = globalToCar(
        pointCloud,
        frameData[
            "cameraCarPose"
        ]
    )

    # ----------------------------------------------------------
    # Car at camera timestamp -> camera
    # ----------------------------------------------------------

    pointCloud = carToCamera(
        pointCloud,
        frameData[
            "cameraCalibration"
        ]
    )

    # ----------------------------------------------------------
    # Camera -> image
    # ----------------------------------------------------------

    points2d, depths = cameraToImage(
        pointCloud,
        frameData[
            "cameraCalibration"
        ]
    )

    return (
        points2d,
        depths
    )


def runVehicleDetection(
    yoloModel,
    cameraImage,
    confidenceThreshold,
    vehicleClasses
):
    """
    Run YOLO on the exact camera image supplied by the runner.

    The YOLO model is loaded once by run_navfusion.py and passed
    into this function.
    """

    imageArray = np.array(
        cameraImage
    )

    results = yoloModel.predict(
        source=imageArray,
        conf=confidenceThreshold,
        verbose=False
    )

    result = results[
        0
    ]

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        classId = int(
            box.cls[
                0
            ].item()
        )

        className = result.names[
            classId
        ]

        # Keep only vehicle categories selected by the runner.
        if className not in vehicleClasses:
            continue

        confidence = float(
            box.conf[
                0
            ].item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[
                0
            ]
            .cpu()
            .numpy()
        )

        detections.append(
            {
                "className": className,

                "confidence": confidence,

                "box": np.array(
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    dtype=np.float32
                )
            }
        )

    return detections


def shrinkBoundingBox(
    box,
    shrinkFactor
):
    """
    Shrink a YOLO bounding box inward.

    Example:

        shrinkFactor = 0.10

    means remove 10% of box width/height from each corresponding
    side.

    This reduces contamination from background LiDAR points close
    to the edges of the YOLO detection.
    """

    x1, y1, x2, y2 = box

    width = (
        x2
        - x1
    )

    height = (
        y2
        - y1
    )

    xShrink = (
        width
        * shrinkFactor
    )

    yShrink = (
        height
        * shrinkFactor
    )

    return np.array(
        [
            x1 + xShrink,
            y1 + yShrink,
            x2 - xShrink,
            y2 - yShrink
        ],
        dtype=np.float32
    )


def filterDepthsWithMad(
    depths,
    madScale
):
    """
    Return a Boolean mask selecting robust depth inliers.

    Median Absolute Deviation:

        medianDepth =
            median(depths)

        deviation_i =
            |depth_i - medianDepth|

        MAD =
            median(deviation_i)

    The median/MAD pair is resistant to a small number of
    background LiDAR returns.
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
        depths
        - medianDepth
    )

    mad = np.median(
        absoluteDeviation
    )

    # If every point has almost exactly the same depth, there is
    # no meaningful spread to reject.
    if mad < 1e-6:

        return np.ones(
            depths.shape,
            dtype=bool
        )

    # For approximately Gaussian data:
    #
    # robustSigma ≈ 1.4826 * MAD
    robustSigma = (
        1.4826
        * mad
    )

    return (
        absoluteDeviation
        <= madScale * robustSigma
    )


def associateLidarWithVehicles(
    detections,
    points2d,
    depths,
    fusionMask,
    originalLidarPoints,
    lidarCalibration,
    boundingBoxShrinkFactor,
    minimumLidarPoints,
    madScale
):
    """
    Associate projected LiDAR returns with each YOLO vehicle.

    Index correspondence is preserved:

        originalLidarPoints[:, i]
                ↕
        points2d[:, i]
                ↕
        depths[i]

    All three refer to the same physical LiDAR return.

    After association:

        1. Keep points projected inside the shrunken YOLO box.
        2. Filter depth outliers with MAD.
        3. Preserve median raw LIDAR_TOP coordinates.
        4. Transform the SAME clean points into the car frame.
        5. Preserve every clean car-frame point for later semantic
           BEV rasterization.
    """

    u = points2d[
        0,
        :
    ]

    v = points2d[
        1,
        :
    ]

    fusedObjects = []

    for detection in detections:

        # ------------------------------------------------------
        # Shrink YOLO bounding box
        # ------------------------------------------------------

        shrunkBox = shrinkBoundingBox(
            detection[
                "box"
            ],
            boundingBoxShrinkFactor
        )

        x1, y1, x2, y2 = shrunkBox

        # ------------------------------------------------------
        # Find projected LiDAR points inside this box
        # ------------------------------------------------------

        objectMask = (
            fusionMask
            & (u >= x1)
            & (u <= x2)
            & (v >= y1)
            & (v <= y2)
        )

        # np.flatnonzero converts the Boolean mask into the
        # original LiDAR-array indices.
        objectIndices = np.flatnonzero(
            objectMask
        )

        if objectIndices.size < minimumLidarPoints:
            continue

        # ------------------------------------------------------
        # Extract camera depths for candidate object points
        # ------------------------------------------------------

        objectDepths = depths[
            objectIndices
        ]

        # ------------------------------------------------------
        # Reject background / foreground depth outliers
        # ------------------------------------------------------

        depthInlierMask = filterDepthsWithMad(
            objectDepths,
            madScale
        )

        cleanIndices = objectIndices[
            depthInlierMask
        ]

        if cleanIndices.size < minimumLidarPoints:
            continue

        cleanDepths = depths[
            cleanIndices
        ]

        # ------------------------------------------------------
        # Camera optical-axis object depth
        # ------------------------------------------------------

        # This is NOT Euclidean range.
        #
        # It is median z depth in the camera coordinate frame.
        distanceM = float(
            np.median(
                cleanDepths
            )
        )

        # ------------------------------------------------------
        # Raw LIDAR_TOP object points
        # ------------------------------------------------------

        # cleanIndices still refer to the original LiDAR array.
        #
        # Therefore these are the exact physical LiDAR returns
        # associated with the object after MAD filtering.
        objectLidarPoints = originalLidarPoints[
            :,
            cleanIndices
        ]

        # ------------------------------------------------------
        # Median raw LIDAR_TOP object position
        # ------------------------------------------------------

        lidarXM = float(
            np.median(
                objectLidarPoints[
                    0,
                    :
                ]
            )
        )

        lidarYM = float(
            np.median(
                objectLidarPoints[
                    1,
                    :
                ]
            )
        )

        lidarZM = float(
            np.median(
                objectLidarPoints[
                    2,
                    :
                ]
            )
        )

        # ------------------------------------------------------
        # Transform SAME clean object points into CAR FRAME
        # ------------------------------------------------------

        objectCarPoints = transformLidarPointsToCar(
            objectLidarPoints,
            lidarCalibration
        )

        # ------------------------------------------------------
        # Median car-frame object position
        # ------------------------------------------------------

        carXM = float(
            np.median(
                objectCarPoints[
                    0,
                    :
                ]
            )
        )

        carYM = float(
            np.median(
                objectCarPoints[
                    1,
                    :
                ]
            )
        )

        carZM = float(
            np.median(
                objectCarPoints[
                    2,
                    :
                ]
            )
        )

        # ------------------------------------------------------
        # Save fused semantic object
        # ------------------------------------------------------

        fusedObjects.append(
            {
                "className": detection[
                    "className"
                ],

                "confidence": detection[
                    "confidence"
                ],

                "box": detection[
                    "box"
                ],

                "shrunkBox": shrunkBox,

                # Camera optical-axis depth.
                "distanceM": distanceM,

                # ----------------------------------------------
                # Median object position in raw LIDAR_TOP frame
                # ----------------------------------------------

                "lidarXM": lidarXM,

                "lidarYM": lidarYM,

                "lidarZM": lidarZM,

                # ----------------------------------------------
                # Median object position in CAR FRAME
                # ----------------------------------------------

                "carXM": carXM,

                "carYM": carYM,

                "carZM": carZM,

                # ----------------------------------------------
                # Complete clean car-frame object point cloud
                # ----------------------------------------------
                #
                # Shape:
                #
                #     (3, cleanLidarCount)
                #
                # These points will later allow the semantic class
                # to be rasterized onto all corresponding BEV
                # cells instead of representing an object using
                # only one median point.
                "cleanCarPoints": objectCarPoints,

                # ----------------------------------------------
                # Association diagnostics
                # ----------------------------------------------

                "rawLidarCount": int(
                    objectIndices.size
                ),

                "cleanLidarCount": int(
                    cleanIndices.size
                ),

                # Keep original LiDAR-array indices because the
                # visualizer uses them to draw the associated
                # points over the camera image.
                "cleanIndices": cleanIndices
            }
        )

    return fusedObjects


def processSensorFusion(
    frameData,
    yoloModel,
    fusionConfig
):
    """
    Process ONE already-loaded frame.

    frameData is supplied completely by run_navfusion.py.

    This function cannot:

        choose a scene,
        choose a sample,
        choose a camera channel,
        choose a LiDAR channel,
        load another sensor file,
        query another calibration record.

    It only processes the exact data supplied by the runner.
    """

    # ----------------------------------------------------------
    # Get already-loaded source data
    # ----------------------------------------------------------

    cameraImage = frameData[
        "cameraImage"
    ]

    lidarPointCloud = frameData[
        "lidarPointCloud"
    ]

    lidarCalibration = frameData[
        "lidarCalibration"
    ]

    # ----------------------------------------------------------
    # Preserve original raw LIDAR_TOP data
    # ----------------------------------------------------------

    # Later transformations work on copies.
    #
    # This array remains the raw sensor-frame source of truth:
    #
    # shape = (4, N)
    originalLidarPoints = (
        lidarPointCloud
        .points
        .copy()
    )

    # ----------------------------------------------------------
    # Project LiDAR into the camera
    # ----------------------------------------------------------

    points2d, depths = projectLidarToCamera(
        frameData
    )

    imageWidth, imageHeight = (
        cameraImage.size
    )

    # ----------------------------------------------------------
    # Determine which projected LiDAR returns are visible
    # ----------------------------------------------------------

    # depths > 1.0:
    #
    # keep points at least 1 m in front of the camera.
    #
    # Remaining conditions ensure the projection lies inside the
    # physical camera image.
    fusionMask = (
        (depths > 1.0)
        & (points2d[0, :] >= 0)
        & (points2d[0, :] < imageWidth)
        & (points2d[1, :] >= 0)
        & (points2d[1, :] < imageHeight)
    )

    # ----------------------------------------------------------
    # Run YOLO vehicle detection
    # ----------------------------------------------------------

    detections = runVehicleDetection(
        yoloModel,
        cameraImage,
        fusionConfig[
            "yoloConfidence"
        ],
        fusionConfig[
            "vehicleClasses"
        ]
    )

    # ----------------------------------------------------------
    # Associate LiDAR returns with YOLO vehicles
    # ----------------------------------------------------------

    fusedObjects = associateLidarWithVehicles(
        detections,
        points2d,
        depths,
        fusionMask,
        originalLidarPoints,

        # Required for:
        #
        # raw LIDAR_TOP
        #       ↓
        # rotation + translation
        #       ↓
        # car-frame semantic object coordinates
        lidarCalibration,

        fusionConfig[
            "boundingBoxShrinkFactor"
        ],
        fusionConfig[
            "minimumLidarPoints"
        ],
        fusionConfig[
            "madScale"
        ]
    )

    # ----------------------------------------------------------
    # Return synchronized sensor-fusion result
    # ----------------------------------------------------------

    return {
        **frameData,

        "originalLidarPoints": originalLidarPoints,

        "points2d": points2d,

        "depths": depths,

        "fusionMask": fusionMask,

        "detections": detections,

        "fusedObjects": fusedObjects
    }