import numpy as np

from navfusion.mapping.lidar_bev import (
    convertMetersToBevGrid
)


# ==============================================================
# Semantic class IDs
# ==============================================================

# Class ID 0 means that no semantic vehicle class has been
# assigned to that cell.
#
# Occupancy is stored separately in the LiDAR occupancy channel,
# so class 0 must NOT be interpreted as "free space".
semanticClassIds = {
    "unlabeled": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
    "motorcycle": 4
}


def pointIsInsideBev(
    x,
    y,
    bevConfig
):
    """
    Return True when one metric car-frame point lies inside the
    half-open BEV region.

    xMin <= x < xMax
    yMin <= y < yMax
    """

    return (
        x >= bevConfig["xMinM"]
        and x < bevConfig["xMaxM"]
        and y >= bevConfig["yMinM"]
        and y < bevConfig["yMaxM"]
    )


def createSemanticBev(
    sensorResult,
    bevResult,
    bevConfig
):
    """
    Rasterize YOLO + LiDAR fused objects onto the existing
    car-frame BEV.

    Input semantic geometry comes from:

        fusedObject["cleanCarPoints"]

    Those points have already:

        1. projected into a YOLO vehicle box,
        2. survived MAD depth filtering,
        3. been transformed from LIDAR_TOP into the car frame.

    We then:

        car-frame object points
                ↓
        apply same BEV metric bounds
                ↓
        same 0.20 m grid conversion
                ↓
        require LiDAR obstacle occupancy
                ↓
        write semantic class ID

    A semantic label is therefore only assigned where there is
    actual LiDAR obstacle support.

    If two YOLO objects attempt to label the same BEV cell, the
    object with the higher YOLO confidence wins.
    """

    bevHeight = bevResult[
        "bevHeight"
    ]

    bevWidth = bevResult[
        "bevWidth"
    ]

    occupancyGrid = bevResult[
        "occupancyGrid"
    ]

    # ----------------------------------------------------------
    # Semantic class grid
    # ----------------------------------------------------------
    #
    # 0 = unlabeled
    # 1 = car
    # 2 = truck
    # 3 = bus
    # 4 = motorcycle
    semanticGrid = np.zeros(
        (
            bevHeight,
            bevWidth
        ),
        dtype=np.int16
    )

    # Store the YOLO confidence associated with the semantic
    # assignment in each cell.
    semanticConfidenceGrid = np.zeros(
        (
            bevHeight,
            bevWidth
        ),
        dtype=np.float32
    )

    # Store which fused object generated each semantic cell.
    #
    # 0 means no semantic object.
    semanticObjectGrid = np.zeros(
        (
            bevHeight,
            bevWidth
        ),
        dtype=np.int32
    )

    # Count how many associated object LiDAR points fell into
    # each BEV cell.
    semanticPointCountGrid = np.zeros(
        (
            bevHeight,
            bevWidth
        ),
        dtype=np.int32
    )

    objectSummaries = []

    objectsContributing = 0

    # ----------------------------------------------------------
    # Rasterize every fused object
    # ----------------------------------------------------------

    for objectNumber, fusedObject in enumerate(
        sensorResult[
            "fusedObjects"
        ],
        start=1
    ):

        className = fusedObject[
            "className"
        ]

        if className not in semanticClassIds:
            continue

        classId = semanticClassIds[
            className
        ]

        objectConfidence = float(
            fusedObject[
                "confidence"
            ]
        )

        cleanCarPoints = np.asarray(
            fusedObject[
                "cleanCarPoints"
            ]
        )

        if (
            cleanCarPoints.ndim != 2
            or cleanCarPoints.shape[0] < 2
            or cleanCarPoints.shape[1] == 0
        ):
            continue

        objectX = cleanCarPoints[
            0,
            :
        ]

        objectY = cleanCarPoints[
            1,
            :
        ]

        # ------------------------------------------------------
        # Restrict semantic points to current BEV region
        # ------------------------------------------------------

        inBevMask = (
            (objectX >= bevConfig["xMinM"])
            & (objectX < bevConfig["xMaxM"])
            & (objectY >= bevConfig["yMinM"])
            & (objectY < bevConfig["yMaxM"])
        )

        objectXInBev = objectX[
            inBevMask
        ]

        objectYInBev = objectY[
            inBevMask
        ]

        pointsInsideBev = int(
            objectXInBev.size
        )

        if pointsInsideBev == 0:

            objectSummaries.append(
                {
                    "objectNumber": objectNumber,
                    "className": className,
                    "classId": classId,
                    "confidence": objectConfidence,
                    "pointsInsideBev": 0,
                    "occupiedSemanticPoints": 0,
                    "semanticCellCount": 0,
                    "centerRow": None,
                    "centerColumn": None
                }
            )

            continue

        # ------------------------------------------------------
        # Convert exactly like the LiDAR BEV
        # ------------------------------------------------------

        (
            objectRows,
            objectColumns,
            _,
            _
        ) = convertMetersToBevGrid(
            objectXInBev,
            objectYInBev,
            bevConfig
        )

        # ------------------------------------------------------
        # Require geometric obstacle support
        # ------------------------------------------------------
        #
        # YOLO alone does not create occupancy.
        #
        # Semantic labels are attached only to cells that the
        # LiDAR BEV already considers occupied.
        obstacleSupportedMask = (
            occupancyGrid[
                objectRows,
                objectColumns
            ]
            > 0
        )

        objectRows = objectRows[
            obstacleSupportedMask
        ]

        objectColumns = objectColumns[
            obstacleSupportedMask
        ]

        occupiedSemanticPoints = int(
            objectRows.size
        )

        if occupiedSemanticPoints == 0:

            objectSummaries.append(
                {
                    "objectNumber": objectNumber,
                    "className": className,
                    "classId": classId,
                    "confidence": objectConfidence,
                    "pointsInsideBev": pointsInsideBev,
                    "occupiedSemanticPoints": 0,
                    "semanticCellCount": 0,
                    "centerRow": None,
                    "centerColumn": None
                }
            )

            continue

        objectsContributing += 1

        # ------------------------------------------------------
        # Find unique BEV cells for this object
        # ------------------------------------------------------

        flatCellIndices = (
            objectRows
            * bevWidth
            + objectColumns
        )

        (
            uniqueFlatCellIndices,
            pointCounts
        ) = np.unique(
            flatCellIndices,
            return_counts=True
        )

        uniqueRows = (
            uniqueFlatCellIndices
            // bevWidth
        )

        uniqueColumns = (
            uniqueFlatCellIndices
            % bevWidth
        )

        # Count associated semantic LiDAR returns in each cell.
        np.add.at(
            semanticPointCountGrid,
            (
                uniqueRows,
                uniqueColumns
            ),
            pointCounts
        )

        # ------------------------------------------------------
        # Resolve overlapping semantic detections
        # ------------------------------------------------------
        #
        # Higher-confidence YOLO detection owns the cell.
        replaceMask = (
            objectConfidence
            >= semanticConfidenceGrid[
                uniqueRows,
                uniqueColumns
            ]
        )

        replaceRows = uniqueRows[
            replaceMask
        ]

        replaceColumns = uniqueColumns[
            replaceMask
        ]

        semanticGrid[
            replaceRows,
            replaceColumns
        ] = classId

        semanticConfidenceGrid[
            replaceRows,
            replaceColumns
        ] = objectConfidence

        semanticObjectGrid[
            replaceRows,
            replaceColumns
        ] = objectNumber

        # ------------------------------------------------------
        # Convert median object center into BEV coordinates
        # ------------------------------------------------------

        centerRow = None
        centerColumn = None

        carXM = float(
            fusedObject[
                "carXM"
            ]
        )

        carYM = float(
            fusedObject[
                "carYM"
            ]
        )

        if pointIsInsideBev(
            carXM,
            carYM,
            bevConfig
        ):

            (
                centerRows,
                centerColumns,
                _,
                _
            ) = convertMetersToBevGrid(
                np.array(
                    [carXM]
                ),
                np.array(
                    [carYM]
                ),
                bevConfig
            )

            centerRow = int(
                centerRows[
                    0
                ]
            )

            centerColumn = int(
                centerColumns[
                    0
                ]
            )

        objectSummaries.append(
            {
                "objectNumber": objectNumber,
                "className": className,
                "classId": classId,
                "confidence": objectConfidence,
                "pointsInsideBev": pointsInsideBev,
                "occupiedSemanticPoints": occupiedSemanticPoints,
                "semanticCellCount": int(
                    uniqueFlatCellIndices.size
                ),
                "centerRow": centerRow,
                "centerColumn": centerColumn
            }
        )

    # ----------------------------------------------------------
    # Count final cells belonging to each semantic class
    # ----------------------------------------------------------

    classCellCounts = {}

    for className, classId in semanticClassIds.items():

        if classId == 0:
            continue

        classCellCounts[
            className
        ] = int(
            np.sum(
                semanticGrid == classId
            )
        )

    semanticCellCount = int(
        np.sum(
            semanticGrid > 0
        )
    )

    # ----------------------------------------------------------
    # Add semantic channel to existing 4-channel BEV
    # ----------------------------------------------------------
    #
    # Existing:
    #
    # [D, H, I, O]
    #
    # New:
    #
    # [D, H, I, O, S]
    semanticMultiChannelBev = np.concatenate(
        (
            bevResult[
                "multiChannelBev"
            ],
            semanticGrid[
                np.newaxis,
                :,
                :
            ].astype(
                np.float32
            )
        ),
        axis=0
    )

    return {
        "semanticGrid": semanticGrid,

        "semanticConfidenceGrid": semanticConfidenceGrid,

        "semanticObjectGrid": semanticObjectGrid,

        "semanticPointCountGrid": semanticPointCountGrid,

        "semanticClassIds": semanticClassIds,

        "classCellCounts": classCellCounts,

        "semanticCellCount": semanticCellCount,

        "objectsContributing": objectsContributing,

        "objectSummaries": objectSummaries,

        "semanticMultiChannelBev": semanticMultiChannelBev
    }