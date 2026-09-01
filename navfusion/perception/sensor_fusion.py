import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def lidarToCar(pointCloud, lidarCalibration):
    """Transform LiDAR sensor-frame points into the car frame."""
    lidarRotation = Quaternion(
        lidarCalibration["rotation"]
    ).rotation_matrix

    lidarTranslation = np.array(
        lidarCalibration["translation"]
    )

    pointCloud.rotate(lidarRotation)
    pointCloud.translate(lidarTranslation)

    return pointCloud


def carToGlobal(pointCloud, carPose):
    """Transform car-frame points into the global frame."""
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
    """Transform global-frame points into a car coordinate frame."""
    carTranslation = np.array(
        carPose["translation"]
    )

    carRotation = Quaternion(
        carPose["rotation"]
    ).rotation_matrix

    # Undo translation.
    pointCloud.translate(
        -carTranslation
    )

    # For a rotation matrix:
    #
    # R^-1 = R^T
    #
    # so the transpose reverses the car rotation.
    pointCloud.rotate(
        carRotation.T
    )

    return pointCloud


def carToCamera(pointCloud, cameraCalibration):
    """Transform car-frame points into the camera coordinate frame."""
    cameraTranslation = np.array(
        cameraCalibration["translation"]
    )

    cameraRotation = Quaternion(
        cameraCalibration["rotation"]
    ).rotation_matrix

    # nuScenes calibration describes camera -> car.
    # We require car -> camera, so use the inverse transformation.
    pointCloud.translate(
        -cameraTranslation
    )

    pointCloud.rotate(
        cameraRotation.T
    )

    return pointCloud


def cameraToImage(pointCloud, cameraCalibration):
    """Project camera-frame 3D points onto the 2D image."""
    # In the camera frame, z is optical depth.
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


def projectLidarToCamera(frameData):
    """
    Project the supplied LiDAR cloud into the supplied camera.

    This function does not access nuScenes itself.

    Transformation:
        LiDAR
          ->
        car at LiDAR time
          ->
        global
          ->
        car at camera time
          ->
        camera
          ->
        image
    """
    # rotate() and translate() change the cloud in place,
    # so work with a copy.
    pointCloud = LidarPointCloud(
        frameData["lidarPointCloud"].points.copy()
    )

    pointCloud = lidarToCar(
        pointCloud,
        frameData["lidarCalibration"]
    )

    pointCloud = carToGlobal(
        pointCloud,
        frameData["lidarCarPose"]
    )

    pointCloud = globalToCar(
        pointCloud,
        frameData["cameraCarPose"]
    )

    pointCloud = carToCamera(
        pointCloud,
        frameData["cameraCalibration"]
    )

    points2d, depths = cameraToImage(
        pointCloud,
        frameData["cameraCalibration"]
    )

    return points2d, depths


def runVehicleDetection(
    yoloModel,
    cameraImage,
    confidenceThreshold,
    vehicleClasses
):
    """
    Run YOLO on the exact image supplied by run_navfusion.py.

    The model is not loaded here. The runner loads it once and
    supplies the already-loaded model.
    """
    imageArray = np.array(
        cameraImage
    )

    results = yoloModel.predict(
        source=imageArray,
        conf=confidenceThreshold,
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

        if className not in vehicleClasses:
            continue

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        detections.append(
            {
                "className": className,
                "confidence": confidence,
                "box": np.array(
                    [x1, y1, x2, y2],
                    dtype=np.float32
                )
            }
        )

    return detections


def shrinkBoundingBox(box, shrinkFactor):
    """
    Shrink the bounding box inward.

    If shrinkFactor = 0.10, 10% of the width/height is removed
    from each corresponding side.
    """
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    xShrink = width * shrinkFactor
    yShrink = height * shrinkFactor

    return np.array(
        [
            x1 + xShrink,
            y1 + yShrink,
            x2 - xShrink,
            y2 - yShrink
        ],
        dtype=np.float32
    )


def filterDepthsWithMad(depths, madScale):
    """
    Return a Boolean mask selecting robust depth inliers.

    Median Absolute Deviation:

        medianDepth = median(depths)

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
        depths - medianDepth
    )

    mad = np.median(
        absoluteDeviation
    )

    if mad < 1e-6:
        return np.ones(
            depths.shape,
            dtype=bool
        )

    # For approximately Gaussian data:
    #
    # sigma_robust ≈ 1.4826 * MAD
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
    boundingBoxShrinkFactor,
    minimumLidarPoints,
    madScale
):
    """
    Associate projected LiDAR points with each YOLO vehicle.

    The array index preserves point correspondence:

        originalLidarPoints[:, i]
              ↕
        points2d[:, i]
              ↕
        depths[i]

    These all represent the same original LiDAR return.
    """
    u = points2d[0, :]
    v = points2d[1, :]

    fusedObjects = []

    for detection in detections:
        shrunkBox = shrinkBoundingBox(
            detection["box"],
            boundingBoxShrinkFactor
        )

        x1, y1, x2, y2 = shrunkBox

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

        if objectIndices.size < minimumLidarPoints:
            continue

        objectDepths = depths[
            objectIndices
        ]

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

        # Camera optical-axis depth.
        distanceM = float(
            np.median(
                cleanDepths
            )
        )

        # Preserve the original LIDAR_TOP xyz for now.
        # We are not changing this behavior during the architecture
        # migration.
        objectLidarPoints = originalLidarPoints[
            :3,
            cleanIndices
        ]

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

        fusedObjects.append(
            {
                "className": detection["className"],
                "confidence": detection["confidence"],
                "box": detection["box"],
                "shrunkBox": shrunkBox,
                "distanceM": distanceM,
                "lidarXM": lidarXM,
                "lidarYM": lidarYM,
                "lidarZM": lidarZM,
                "rawLidarCount": int(
                    objectIndices.size
                ),
                "cleanLidarCount": int(
                    cleanIndices.size
                ),
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
        - choose a scene,
        - choose a sample,
        - choose a camera channel,
        - choose a LiDAR channel,
        - load another sensor file,
        - query another calibration record.

    It only processes the data it was given.
    """
    cameraImage = frameData[
        "cameraImage"
    ]

    lidarPointCloud = frameData[
        "lidarPointCloud"
    ]

    originalLidarPoints = (
        lidarPointCloud.points.copy()
    )

    points2d, depths = projectLidarToCamera(
        frameData
    )

    imageWidth, imageHeight = (
        cameraImage.size
    )

    fusionMask = (
        (depths > 1.0)
        & (points2d[0, :] >= 0)
        & (points2d[0, :] < imageWidth)
        & (points2d[1, :] >= 0)
        & (points2d[1, :] < imageHeight)
    )

    detections = runVehicleDetection(
        yoloModel,
        cameraImage,
        fusionConfig["yoloConfidence"],
        fusionConfig["vehicleClasses"]
    )

    fusedObjects = associateLidarWithVehicles(
        detections,
        points2d,
        depths,
        fusionMask,
        originalLidarPoints,
        fusionConfig["boundingBoxShrinkFactor"],
        fusionConfig["minimumLidarPoints"],
        fusionConfig["madScale"]
    )

    return {
        **frameData,
        "originalLidarPoints": originalLidarPoints,
        "points2d": points2d,
        "depths": depths,
        "fusionMask": fusionMask,
        "detections": detections,
        "fusedObjects": fusedObjects
    }