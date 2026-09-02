import numpy as np

from pyquaternion import Quaternion

from navfusion.mapping.lidar_bev import (
    calculateBevGridSize
)


def createBevGroundGrid(
    bevConfig,
    planeCoefficients
):
    """
    Create one 3D ground point at the center of every BEV cell.

    The BEV uses the same orientation as lidar_bev.py:

        +x = forward
        +y = left
        +z = up

    Image orientation:

        row 0
            =
        far forward

        column 0
            =
        left side

    The z coordinate comes from the RANSAC ground plane:

        z = a*x + b*y + c

    Therefore the camera IPM follows the ground plane already
    estimated from the LiDAR rather than assuming a perfectly
    horizontal road.
    """

    bevHeight, bevWidth = calculateBevGridSize(
        bevConfig
    )

    resolutionM = bevConfig[
        "resolutionM"
    ]

    # ----------------------------------------------------------
    # Create image row/column coordinates
    # ----------------------------------------------------------

    rowIndices = np.arange(
        bevHeight
    )

    columnIndices = np.arange(
        bevWidth
    )

    bevRows, bevColumns = np.meshgrid(
        rowIndices,
        columnIndices,
        indexing="ij"
    )

    # ----------------------------------------------------------
    # Convert image indices back into physical grid cells
    # ----------------------------------------------------------
    #
    # Forward BEV conversion was:
    #
    # row =
    #     bevHeight - 1 - xCell
    #
    # Therefore inverse:
    #
    # xCell =
    #     bevHeight - 1 - row
    xCells = (
        bevHeight
        - 1
        - bevRows
    )

    # Forward:
    #
    # column =
    #     bevWidth - 1 - yCell
    #
    # Therefore inverse:
    #
    # yCell =
    #     bevWidth - 1 - column
    yCells = (
        bevWidth
        - 1
        - bevColumns
    )

    # ----------------------------------------------------------
    # Metric location of each CELL CENTER
    # ----------------------------------------------------------
    #
    # +0.5 moves from the cell boundary to its center.
    x = (
        bevConfig[
            "xMinM"
        ]
        + (
            xCells
            + 0.5
        )
        * resolutionM
    )

    y = (
        bevConfig[
            "yMinM"
        ]
        + (
            yCells
            + 0.5
        )
        * resolutionM
    )

    # ----------------------------------------------------------
    # Ground z from RANSAC plane
    # ----------------------------------------------------------

    a, b, c = planeCoefficients

    z = (
        a * x
        + b * y
        + c
    )

    # ----------------------------------------------------------
    # Flatten into:
    #
    #     3 x N
    #
    # where:
    #
    # N = 350 * 250 = 87,500 cells
    # ----------------------------------------------------------

    groundPoints = np.stack(
        (
            x,
            y,
            z
        ),
        axis=0
    ).reshape(
        3,
        -1
    )

    return {
        "groundPoints": groundPoints,

        "xGrid": x,

        "yGrid": y,

        "zGrid": z,

        "bevHeight": bevHeight,

        "bevWidth": bevWidth
    }


def lidarTimeCarToGlobal(
    carPoints,
    lidarCarPose
):
    """
    Transform points from the car frame at the LiDAR timestamp
    into the nuScenes global frame.

    Formula:

        pGlobal = R * pCar + t
    """

    rotationMatrix = Quaternion(
        lidarCarPose[
            "rotation"
        ]
    ).rotation_matrix

    translationVector = np.array(
        lidarCarPose[
            "translation"
        ],
        dtype=np.float64
    ).reshape(
        3,
        1
    )

    globalPoints = (
        rotationMatrix
        @ carPoints
        + translationVector
    )

    return globalPoints


def globalToCameraTimeCar(
    globalPoints,
    cameraCarPose
):
    """
    Transform global points into the car frame at the camera
    timestamp.

    Forward car -> global:

        pGlobal = R * pCar + t

    Therefore inverse:

        pCar = R^T * (pGlobal - t)
    """

    rotationMatrix = Quaternion(
        cameraCarPose[
            "rotation"
        ]
    ).rotation_matrix

    translationVector = np.array(
        cameraCarPose[
            "translation"
        ],
        dtype=np.float64
    ).reshape(
        3,
        1
    )

    cameraTimeCarPoints = (
        rotationMatrix.T
        @ (
            globalPoints
            - translationVector
        )
    )

    return cameraTimeCarPoints


def cameraTimeCarToCamera(
    carPoints,
    cameraCalibration
):
    """
    Transform points from the car frame into the camera frame.

    nuScenes calibrated_sensor stores:

        camera -> car

    We need:

        car -> camera

    Therefore:

        pCamera = R^T * (pCar - t)
    """

    rotationMatrix = Quaternion(
        cameraCalibration[
            "rotation"
        ]
    ).rotation_matrix

    translationVector = np.array(
        cameraCalibration[
            "translation"
        ],
        dtype=np.float64
    ).reshape(
        3,
        1
    )

    cameraPoints = (
        rotationMatrix.T
        @ (
            carPoints
            - translationVector
        )
    )

    return cameraPoints


def projectCameraPointsToImage(
    cameraPoints,
    cameraCalibration,
    minimumCameraDepthM
):
    """
    Project 3D camera-frame points onto the camera image.

    Camera pinhole projection:

                fx * X
        u = ------------- + cx
                  Z

                fy * Y
        v = ------------- + cy
                  Z

    Matrix form:

        q = K * pCamera

        u = qx / qz
        v = qy / qz

    Only points with positive camera depth are projectable.
    """

    cameraIntrinsic = np.array(
        cameraCalibration[
            "camera_intrinsic"
        ],
        dtype=np.float64
    )

    depths = cameraPoints[
        2,
        :
    ]

    projected = (
        cameraIntrinsic
        @ cameraPoints
    )

    u = np.full(
        depths.shape,
        np.nan,
        dtype=np.float64
    )

    v = np.full(
        depths.shape,
        np.nan,
        dtype=np.float64
    )

    positiveDepthMask = (
        depths
        > minimumCameraDepthM
    )

    u[
        positiveDepthMask
    ] = (
        projected[
            0,
            positiveDepthMask
        ]
        / projected[
            2,
            positiveDepthMask
        ]
    )

    v[
        positiveDepthMask
    ] = (
        projected[
            1,
            positiveDepthMask
        ]
        / projected[
            2,
            positiveDepthMask
        ]
    )

    return (
        u,
        v,
        depths,
        positiveDepthMask
    )


def sampleCameraImageToBev(
    cameraImage,
    u,
    v,
    positiveDepthMask,
    bevHeight,
    bevWidth
):
    """
    Sample the camera image at the projected location of every
    BEV ground cell.

    This first implementation uses nearest-neighbor sampling.

    Invalid or camera-invisible BEV cells remain black.
    """

    imageArray = np.array(
        cameraImage
    )

    imageHeight, imageWidth, _ = imageArray.shape

    # ----------------------------------------------------------
    # Find points physically inside the image
    # ----------------------------------------------------------

    imageMask = (
        positiveDepthMask
        & np.isfinite(
            u
        )
        & np.isfinite(
            v
        )
        & (u >= 0)
        & (u < imageWidth)
        & (v >= 0)
        & (v < imageHeight)
    )

    validIndices = np.flatnonzero(
        imageMask
    )

    # ----------------------------------------------------------
    # Allocate flat BEV RGB image
    # ----------------------------------------------------------

    totalCells = (
        bevHeight
        * bevWidth
    )

    bevRgbFlat = np.zeros(
        (
            totalCells,
            3
        ),
        dtype=np.uint8
    )

    # ----------------------------------------------------------
    # Convert floating image coordinates to nearest pixels
    # ----------------------------------------------------------

    uPixels = np.rint(
        u[
            validIndices
        ]
    ).astype(
        np.int32
    )

    vPixels = np.rint(
        v[
            validIndices
        ]
    ).astype(
        np.int32
    )

    # Rounding could move a coordinate such as 1599.7 to 1600,
    # so clip after rounding.
    uPixels = np.clip(
        uPixels,
        0,
        imageWidth - 1
    )

    vPixels = np.clip(
        vPixels,
        0,
        imageHeight - 1
    )

    bevRgbFlat[
        validIndices
    ] = imageArray[
        vPixels,
        uPixels
    ]

    # ----------------------------------------------------------
    # Restore BEV image geometry
    # ----------------------------------------------------------

    cameraIpmRgb = bevRgbFlat.reshape(
        bevHeight,
        bevWidth,
        3
    )

    cameraIpmValidMask = imageMask.reshape(
        bevHeight,
        bevWidth
    )

    return (
        cameraIpmRgb,
        cameraIpmValidMask
    )


def processCameraIpm(
    frameData,
    bevResult,
    bevConfig,
    ipmConfig
):
    """
    Generate CAM_FRONT inverse-perspective mapping aligned to the
    existing LiDAR BEV.

    IMPORTANT:

    The target grid is the car frame at the LIDAR timestamp,
    because that is the coordinate frame already used by
    processLidarBev().

    Transformation:

        BEV ground cell
            ↓
        car frame at LiDAR timestamp
            ↓
        global
            ↓
        car frame at camera timestamp
            ↓
        camera
            ↓
        image pixel
            ↓
        RGB value copied into BEV cell

    This preserves temporal alignment between the asynchronous
    camera and LiDAR measurements.
    """

    # ----------------------------------------------------------
    # Create ground points in EXACT existing BEV geometry
    # ----------------------------------------------------------

    groundGrid = createBevGroundGrid(
        bevConfig,
        bevResult[
            "planeCoefficients"
        ]
    )

    groundPoints = groundGrid[
        "groundPoints"
    ]

    # ----------------------------------------------------------
    # LiDAR-time car -> global
    # ----------------------------------------------------------

    globalPoints = lidarTimeCarToGlobal(
        groundPoints,
        frameData[
            "lidarCarPose"
        ]
    )

    # ----------------------------------------------------------
    # Global -> camera-time car
    # ----------------------------------------------------------

    cameraTimeCarPoints = globalToCameraTimeCar(
        globalPoints,
        frameData[
            "cameraCarPose"
        ]
    )

    # ----------------------------------------------------------
    # Camera-time car -> camera
    # ----------------------------------------------------------

    cameraPoints = cameraTimeCarToCamera(
        cameraTimeCarPoints,
        frameData[
            "cameraCalibration"
        ]
    )

    # ----------------------------------------------------------
    # Camera -> image
    # ----------------------------------------------------------

    (
        u,
        v,
        depths,
        positiveDepthMask
    ) = projectCameraPointsToImage(
        cameraPoints,
        frameData[
            "cameraCalibration"
        ],
        ipmConfig[
            "minimumCameraDepthM"
        ]
    )

    # ----------------------------------------------------------
    # Image -> BEV RGB
    # ----------------------------------------------------------

    (
        cameraIpmRgb,
        cameraIpmValidMask
    ) = sampleCameraImageToBev(
        frameData[
            "cameraImage"
        ],
        u,
        v,
        positiveDepthMask,
        groundGrid[
            "bevHeight"
        ],
        groundGrid[
            "bevWidth"
        ]
    )

    validCellCount = int(
        np.sum(
            cameraIpmValidMask
        )
    )

    totalCellCount = int(
        cameraIpmValidMask.size
    )

    coveragePercent = (
        100.0
        * validCellCount
        / totalCellCount
    )

    return {
        "cameraIpmRgb": cameraIpmRgb,

        "cameraIpmValidMask": cameraIpmValidMask,

        "u": u,

        "v": v,

        "depths": depths,

        "cameraPoints": cameraPoints,

        "xGrid": groundGrid[
            "xGrid"
        ],

        "yGrid": groundGrid[
            "yGrid"
        ],

        "zGrid": groundGrid[
            "zGrid"
        ],

        "bevHeight": groundGrid[
            "bevHeight"
        ],

        "bevWidth": groundGrid[
            "bevWidth"
        ],

        "validCellCount": validCellCount,

        "totalCellCount": totalCellCount,

        "coveragePercent": coveragePercent
    }