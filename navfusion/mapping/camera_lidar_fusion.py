import numpy as np


# ==============================================================
# Semantic display colors
# ==============================================================

# RGB display values only.
#
# These colors do NOT change the numeric semantic IDs stored in
# the semantic BEV.
#
# Semantic IDs:
#
# 0 = no semantic vehicle label
# 1 = car
# 2 = truck
# 3 = bus
# 4 = motorcycle
semanticClassColors = {
    1: np.array(
        [
            31,
            119,
            180
        ],
        dtype=np.float32
    ),

    2: np.array(
        [
            255,
            127,
            14
        ],
        dtype=np.float32
    ),

    3: np.array(
        [
            44,
            160,
            44
        ],
        dtype=np.float32
    ),

    4: np.array(
        [
            214,
            39,
            40
        ],
        dtype=np.float32
    )
}


# ==============================================================
# Blend one RGB color into selected BEV cells
# ==============================================================

def blendColorIntoMask(
    image,
    mask,
    color,
    alpha
):
    """
    Alpha-blend one RGB color into selected image cells.

    Parameters:

        image
            RGB image with shape:

                (height, width, 3)

        mask
            Boolean grid with shape:

                (height, width)

        color
            RGB overlay color:

                [R, G, B]

        alpha
            Overlay strength between 0.0 and 1.0.

    Mathematical operation:

        output =
            (1 - alpha) * background
            +
            alpha * overlay

    Example:

        alpha = 0.70

        fusedPixel =
            0.30 * cameraPixel
            +
            0.70 * obstacleColor
    """

    # ----------------------------------------------------------
    # Nothing to do if this mask contains no selected cells
    # ----------------------------------------------------------

    if not np.any(
        mask
    ):
        return

    # ----------------------------------------------------------
    # Extract selected RGB pixels
    # ----------------------------------------------------------

    backgroundPixels = image[
        mask
    ].astype(
        np.float32
    )

    # ----------------------------------------------------------
    # Alpha blending
    # ----------------------------------------------------------

    blendedPixels = (
        (
            1.0
            - alpha
        )
        * backgroundPixels
        +
        alpha
        * color
    )

    # ----------------------------------------------------------
    # Convert back to standard uint8 RGB
    # ----------------------------------------------------------

    image[
        mask
    ] = np.clip(
        blendedPixels,
        0,
        255
    ).astype(
        np.uint8
    )


# ==============================================================
# Validate common BEV geometry
# ==============================================================

def validateFusionShapes(
    ipmResult,
    bevResult,
    semanticResult
):
    """
    Confirm that Camera IPM, LiDAR occupancy and semantic BEV all
    use exactly the same BEV row/column geometry.

    Expected:

        Camera IPM:
            (350, 250, 3)

        LiDAR occupancy:
            (350, 250)

        Semantic grid:
            (350, 250)
    """

    cameraIpmRgb = ipmResult[
        "cameraIpmRgb"
    ]

    occupancyGrid = bevResult[
        "occupancyGrid"
    ]

    semanticGrid = semanticResult[
        "semanticGrid"
    ]

    cameraShape = cameraIpmRgb.shape[
        0:2
    ]

    occupancyShape = occupancyGrid.shape

    semanticShape = semanticGrid.shape

    if cameraShape != occupancyShape:
        raise ValueError(
            f"Camera IPM shape {cameraShape} does not match "
            f"LiDAR occupancy shape {occupancyShape}."
        )

    if cameraShape != semanticShape:
        raise ValueError(
            f"Camera IPM shape {cameraShape} does not match "
            f"semantic BEV shape {semanticShape}."
        )


# ==============================================================
# Create fused Camera + LiDAR BEV
# ==============================================================

def processCameraLidarFusion(
    ipmResult,
    bevResult,
    semanticResult,
    fusionConfig
):
    """
    Fuse Camera IPM, LiDAR obstacle occupancy and vehicle semantic
    labels into one RGB car-frame BEV visualization.

    Data responsibilities:

        Camera IPM
            ↓
        visual road / lane / texture background

        LiDAR occupancy
            ↓
        physical obstacle geometry

        Semantic BEV
            ↓
        vehicle-class identity

    IMPORTANT:

    This function creates a NEW RGB visualization.

    It does not modify:

        ipmResult["cameraIpmRgb"]

        bevResult["occupancyGrid"]

        semanticResult["semanticGrid"]

        semanticResult["semanticMultiChannelBev"]
    """

    # ----------------------------------------------------------
    # Confirm that all inputs use the same BEV grid
    # ----------------------------------------------------------

    validateFusionShapes(
        ipmResult,
        bevResult,
        semanticResult
    )

    # ----------------------------------------------------------
    # Get input representations
    # ----------------------------------------------------------

    cameraIpmRgb = ipmResult[
        "cameraIpmRgb"
    ]

    occupancyGrid = bevResult[
        "occupancyGrid"
    ]

    semanticGrid = semanticResult[
        "semanticGrid"
    ]

    # ----------------------------------------------------------
    # Start with the Camera IPM as the visual background
    # ----------------------------------------------------------
    #
    # copy() is important.
    #
    # Without copy(), the fusion overlays would directly modify
    # the original IPM array stored inside ipmResult.
    fusedBevRgb = cameraIpmRgb.copy()

    # ----------------------------------------------------------
    # LiDAR physical occupancy mask
    # ----------------------------------------------------------

    occupiedMask = (
        occupancyGrid
        > 0
    )

    # ----------------------------------------------------------
    # Semantic vehicle mask
    # ----------------------------------------------------------

    semanticMask = (
        semanticGrid
        > 0
    )

    # ----------------------------------------------------------
    # LiDAR obstacle without a known vehicle semantic class
    # ----------------------------------------------------------
    #
    # Example:
    #
    #     wall
    #     pole
    #     vegetation
    #     unidentified obstacle
    #
    # LiDAR says something is physically present, but our current
    # CAM_FRONT YOLO vehicle classes do not identify it.
    unlabeledObstacleMask = (
        occupiedMask
        & ~semanticMask
    )

    # ----------------------------------------------------------
    # Overlay LiDAR-only obstacles
    # ----------------------------------------------------------

    unlabeledObstacleColor = np.array(
        fusionConfig[
            "unlabeledObstacleColorRgb"
        ],
        dtype=np.float32
    )

    blendColorIntoMask(
        fusedBevRgb,
        unlabeledObstacleMask,
        unlabeledObstacleColor,
        fusionConfig[
            "unlabeledObstacleAlpha"
        ]
    )

    # ----------------------------------------------------------
    # Overlay semantic vehicle classes
    # ----------------------------------------------------------
    #
    # semanticGrid:
    #
    # 0 = no vehicle semantic label
    # 1 = car
    # 2 = truck
    # 3 = bus
    # 4 = motorcycle

    classCellCounts = {}

    for classId, classColor in semanticClassColors.items():

        classMask = (
            semanticGrid
            == classId
        )

        classCellCount = int(
            np.sum(
                classMask
            )
        )

        classCellCounts[
            classId
        ] = classCellCount

        # ------------------------------------------------------
        # Blend semantic class color over Camera IPM
        # ------------------------------------------------------
        #
        # Correct argument order:
        #
        #     image
        #     mask
        #     color
        #     alpha
        blendColorIntoMask(
            fusedBevRgb,
            classMask,
            classColor,
            fusionConfig[
                "semanticAlpha"
            ]
        )

    # ----------------------------------------------------------
    # Camera-visible region
    # ----------------------------------------------------------

    cameraVisibleMask = ipmResult[
        "cameraIpmValidMask"
    ]

    # ----------------------------------------------------------
    # LiDAR obstacles that also lie inside the CAM_FRONT IPM
    # ----------------------------------------------------------

    cameraVisibleObstacleMask = (
        occupiedMask
        & cameraVisibleMask
    )

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    occupiedCellCount = int(
        np.sum(
            occupiedMask
        )
    )

    semanticCellCount = int(
        np.sum(
            semanticMask
        )
    )

    unlabeledObstacleCellCount = int(
        np.sum(
            unlabeledObstacleMask
        )
    )

    cameraVisibleObstacleCellCount = int(
        np.sum(
            cameraVisibleObstacleMask
        )
    )

    # ----------------------------------------------------------
    # Return independent fusion result
    # ----------------------------------------------------------

    return {
        "fusedBevRgb": fusedBevRgb,

        "occupiedMask": occupiedMask,

        "semanticMask": semanticMask,

        "unlabeledObstacleMask": (
            unlabeledObstacleMask
        ),

        "cameraVisibleObstacleMask": (
            cameraVisibleObstacleMask
        ),

        "occupiedCellCount": (
            occupiedCellCount
        ),

        "semanticCellCount": (
            semanticCellCount
        ),

        "unlabeledObstacleCellCount": (
            unlabeledObstacleCellCount
        ),

        "cameraVisibleObstacleCellCount": (
            cameraVisibleObstacleCellCount
        ),

        "classCellCounts": (
            classCellCounts
        )
    }