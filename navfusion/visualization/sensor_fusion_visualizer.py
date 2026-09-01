import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def drawCameraImage(
    axis,
    sensorResult
):
    """
    Draw the original CAM_FRONT image.
    """

    axis.imshow(
        sensorResult[
            "cameraImage"
        ]
    )

    axis.set_title(
        "CAM_FRONT"
    )

    axis.axis(
        "off"
    )


def drawRawLidar(
    axis,
    sensorResult,
    visualizationRangeM
):
    """
    Draw raw LIDAR_TOP sensor-frame points.

    +x = forward
    +y = left
    """

    lidarPoints = sensorResult[
        "lidarPointCloud"
    ].points

    x = lidarPoints[
        0,
        :
    ]

    y = lidarPoints[
        1,
        :
    ]

    rangeMask = (
        (x >= -visualizationRangeM)
        & (x <= visualizationRangeM)
        & (y >= -visualizationRangeM)
        & (y <= visualizationRangeM)
    )

    # Horizontal axis = LiDAR y.
    # Vertical axis   = LiDAR x.
    axis.scatter(
        y[
            rangeMask
        ],
        x[
            rangeMask
        ],
        s=1
    )

    axis.scatter(
        0.0,
        0.0,
        marker="^",
        s=80,
        label="LIDAR_TOP"
    )

    # +y means left.
    #
    # Inverting the horizontal axis makes physical left appear
    # visually on the left side of the plot.
    axis.set_xlim(
        visualizationRangeM,
        -visualizationRangeM
    )

    axis.set_ylim(
        -visualizationRangeM,
        visualizationRangeM
    )

    axis.set_aspect(
        "equal"
    )

    axis.set_xlabel(
        "LiDAR y (m, +left)"
    )

    axis.set_ylabel(
        "LiDAR x (m, +forward)"
    )

    axis.set_title(
        "LIDAR_TOP - Raw Sensor-Frame XY"
    )

    axis.grid(
        True,
        alpha=0.25
    )

    axis.legend(
        loc="upper right"
    )


def drawProjectedLidar(
    axis,
    sensorResult
):
    """
    Draw every camera-visible projected LiDAR return.
    """

    cameraImage = sensorResult[
        "cameraImage"
    ]

    points2d = sensorResult[
        "points2d"
    ]

    depths = sensorResult[
        "depths"
    ]

    fusionMask = sensorResult[
        "fusionMask"
    ]

    axis.imshow(
        cameraImage
    )

    axis.scatter(
        points2d[
            0,
            fusionMask
        ],
        points2d[
            1,
            fusionMask
        ],
        c=depths[
            fusionMask
        ],
        s=3,
        cmap="viridis",
        alpha=0.80
    )

    axis.set_title(
        "CAM_FRONT + Projected LiDAR"
    )

    axis.axis(
        "off"
    )


def drawVehicleFusion(
    axis,
    sensorResult
):
    """
    Draw YOLO vehicle boxes and the LiDAR returns associated with
    each fused vehicle.
    """

    cameraImage = sensorResult[
        "cameraImage"
    ]

    points2d = sensorResult[
        "points2d"
    ]

    depths = sensorResult[
        "depths"
    ]

    fusedObjects = sensorResult[
        "fusedObjects"
    ]

    axis.imshow(
        cameraImage
    )

    for fusedObject in fusedObjects:

        x1, y1, x2, y2 = fusedObject[
            "box"
        ]

        width = (
            x2
            - x1
        )

        height = (
            y2
            - y1
        )

        rectangle = plt.Rectangle(
            (
                x1,
                y1
            ),
            width,
            height,
            fill=False,
            linewidth=2
        )

        axis.add_patch(
            rectangle
        )

        cleanIndices = fusedObject[
            "cleanIndices"
        ]

        axis.scatter(
            points2d[
                0,
                cleanIndices
            ],
            points2d[
                1,
                cleanIndices
            ],
            c=depths[
                cleanIndices
            ],
            s=8,
            cmap="plasma"
        )

        label = (
            f"{fusedObject['className']} "
            f"{fusedObject['confidence']:.2f}\n"
            f"{fusedObject['distanceM']:.1f} m"
        )

        axis.text(
            x1,
            max(
                0,
                y1 - 8
            ),
            label,
            fontsize=9,
            bbox={
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none"
            }
        )

    axis.set_title(
        "YOLO + Object-Associated LiDAR"
    )

    axis.axis(
        "off"
    )


def drawCarMarker(
    axis,
    bevResult
):
    """
    Draw the true car-frame origin on a BEV.
    """

    axis.scatter(
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


def drawLidarBev(
    axis,
    bevGrid,
    bevResult,
    title,
    cmap="viridis",
    colorbarLabel=None
):
    """
    Draw one car-frame LiDAR BEV channel.
    """

    image = axis.imshow(
        bevGrid,
        origin="upper",
        cmap=cmap,
        interpolation="nearest"
    )

    drawCarMarker(
        axis,
        bevResult
    )

    axis.set_title(
        title,
        pad=10
    )

    axis.set_xlabel(
        "BEV column (+y left)"
    )

    axis.set_ylabel(
        "BEV row (+x forward)"
    )

    axis.legend(
        loc="lower right"
    )

    if colorbarLabel is not None:

        colorbar = axis.figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04
        )

        colorbar.set_label(
            colorbarLabel
        )


def drawSemanticBev(
    axis,
    semanticResult,
    bevResult
):
    """
    Draw the semantic BEV.

    Semantic classes come from YOLO.

    Their physical locations come from the associated car-frame
    LiDAR points.

    Display:

        black       = no obstacle
        dark gray   = LiDAR obstacle without semantic label
        blue        = car
        orange      = truck
        green       = bus
        red         = motorcycle
    """

    semanticGrid = semanticResult[
        "semanticGrid"
    ]

    occupancyGrid = bevResult[
        "occupancyGrid"
    ]

    # ----------------------------------------------------------
    # Start with background
    # ----------------------------------------------------------

    semanticDisplayGrid = np.zeros(
        semanticGrid.shape,
        dtype=np.int16
    )

    # ----------------------------------------------------------
    # Show occupied but semantically unlabeled cells
    # ----------------------------------------------------------

    # Display value 5 means:
    #
    # LiDAR detects a physical obstacle,
    # but CAM_FRONT + YOLO has not assigned a vehicle class.
    unlabeledObstacleMask = (
        (occupancyGrid > 0)
        & (semanticGrid == 0)
    )

    semanticDisplayGrid[
        unlabeledObstacleMask
    ] = 5

    # ----------------------------------------------------------
    # Copy semantic classes
    # ----------------------------------------------------------

    semanticMask = (
        semanticGrid > 0
    )

    semanticDisplayGrid[
        semanticMask
    ] = semanticGrid[
        semanticMask
    ]

    # ----------------------------------------------------------
    # Visualization colors
    # ----------------------------------------------------------

    semanticCmap = ListedColormap(
        [
            "black",       # 0 background
            "tab:blue",    # 1 car
            "tab:orange",  # 2 truck
            "tab:green",   # 3 bus
            "tab:red",     # 4 motorcycle
            "dimgray"      # 5 unlabeled obstacle
        ]
    )

    semanticNorm = BoundaryNorm(
        [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            5.5
        ],
        semanticCmap.N
    )

    axis.imshow(
        semanticDisplayGrid,
        origin="upper",
        cmap=semanticCmap,
        norm=semanticNorm,
        interpolation="nearest"
    )

    # ----------------------------------------------------------
    # Make actual semantic cells clearly visible
    # ----------------------------------------------------------

    semanticRows, semanticColumns = np.nonzero(
        semanticGrid > 0
    )

    if semanticRows.size > 0:

        semanticValues = semanticGrid[
            semanticRows,
            semanticColumns
        ]

        axis.scatter(
            semanticColumns,
            semanticRows,
            c=semanticValues,
            cmap=semanticCmap,
            norm=semanticNorm,
            marker="s",
            s=20,
            linewidths=0
        )

    # ----------------------------------------------------------
    # Draw car-frame origin
    # ----------------------------------------------------------

    drawCarMarker(
        axis,
        bevResult
    )

    # ----------------------------------------------------------
    # Label semantic objects
    # ----------------------------------------------------------

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

        axis.text(
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

    # ----------------------------------------------------------
    # Semantic class legend
    # ----------------------------------------------------------

    legendHandles = [
        Patch(
            facecolor="tab:blue",
            label="Car"
        ),

        Patch(
            facecolor="tab:orange",
            label="Truck"
        ),

        Patch(
            facecolor="tab:green",
            label="Bus"
        ),

        Patch(
            facecolor="tab:red",
            label="Motorcycle"
        ),

        Patch(
            facecolor="dimgray",
            label="Unlabeled LiDAR obstacle"
        )
    ]

    # Place the legend outside the BEV so it does not hide cells.
    axis.legend(
        handles=legendHandles,
        loc="upper left",
        bbox_to_anchor=(
            1.02,
            1.0
        ),
        fontsize=7,
        borderaxespad=0
    )

    axis.set_title(
        "Semantic BEV - Vehicle Classes",
        pad=10
    )

    axis.set_xlabel(
        "BEV column (+y left)"
    )

    axis.set_ylabel(
        "BEV row (+x forward)"
    )


def createSensorFusionFigure(
    sensorResult,
    bevResult,
    semanticResult,
    visualizationRangeM
):
    """
    Create the complete NavFusion semantic-BEV visualization.

    Layout:

        Row 1:
            Camera
            Raw LiDAR
            Projected LiDAR

        Row 2:
            YOLO + associated LiDAR
            Density BEV
            Maximum-height BEV

        Row 3:
            Mean-intensity BEV
            Occupancy BEV
            Semantic BEV
    """

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(
            24,
            18
        )
    )

    # ----------------------------------------------------------
    # Row 1
    # ----------------------------------------------------------

    drawCameraImage(
        axes[
            0,
            0
        ],
        sensorResult
    )

    drawRawLidar(
        axes[
            0,
            1
        ],
        sensorResult,
        visualizationRangeM
    )

    drawProjectedLidar(
        axes[
            0,
            2
        ],
        sensorResult
    )

    # ----------------------------------------------------------
    # Row 2
    # ----------------------------------------------------------

    drawVehicleFusion(
        axes[
            1,
            0
        ],
        sensorResult
    )

    drawLidarBev(
        axes[
            1,
            1
        ],
        bevResult[
            "obstacleDisplayDensity"
        ],
        bevResult,
        "Car-Frame LiDAR BEV - Density",
        cmap="viridis"
    )

    drawLidarBev(
        axes[
            1,
            2
        ],
        bevResult[
            "maxHeightGrid"
        ],
        bevResult,
        "Car-Frame LiDAR BEV - Maximum Height",
        cmap="viridis",
        colorbarLabel="Height above ground (m)"
    )

    # ----------------------------------------------------------
    # Row 3
    # ----------------------------------------------------------

    drawLidarBev(
        axes[
            2,
            0
        ],
        bevResult[
            "meanIntensityGrid"
        ],
        bevResult,
        "Car-Frame LiDAR BEV - Mean Intensity",
        cmap="viridis",
        colorbarLabel="Mean intensity"
    )

    drawLidarBev(
        axes[
            2,
            1
        ],
        bevResult[
            "occupancyGrid"
        ],
        bevResult,
        "Car-Frame LiDAR BEV - Occupancy",
        cmap="gray"
    )

    # ----------------------------------------------------------
    # Actual semantic BEV panel
    # ----------------------------------------------------------

    drawSemanticBev(
        axes[
            2,
            2
        ],
        semanticResult,
        bevResult
    )

    # ----------------------------------------------------------
    # Figure spacing
    # ----------------------------------------------------------
    #
    # Right margin:
    #     reserved for semantic legend.
    #
    # Top margin:
    #     reserved for figure.suptitle() from run_navfusion.py.
    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            0.94,
            0.95
        ),
        h_pad=3.0,
        w_pad=3.0
    )

    return figure