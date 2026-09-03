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


def drawSurroundCameraIpm(
    axis,
    ipmResult,
    bevResult
):
    """
    Draw the composed six-camera surround IPM.

    The image is already expressed in the same car-frame BEV
    geometry as the LiDAR and semantic grids.
    """

    axis.imshow(
        ipmResult[
            "cameraIpmRgb"
        ],
        origin="upper",
        interpolation="nearest"
    )

    drawCarMarker(
        axis,
        bevResult
    )

    axis.set_title(
        "Six-Camera Surround IPM",
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


def drawSurroundCameraLidarFusion(
    axis,
    cameraLidarFusionResult,
    semanticResult,
    bevResult
):
    """
    Draw the final surround Camera-IPM + LiDAR + semantic BEV.

    Camera:
        road / lane / visual texture

    LiDAR:
        physical obstacle geometry

    Semantic BEV:
        supported vehicle-class identity
    """

    axis.imshow(
        cameraLidarFusionResult[
            "fusedBevRgb"
        ],
        origin="upper",
        interpolation="nearest"
    )

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

    axis.set_title(
        "Fused Surround IPM + LiDAR + Semantics",
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


def drawSurroundCameraSourceMap(
    axis,
    ipmResult,
    bevResult
):
    """
    Show which camera supplied each final surround-IPM BEV cell.

    sourceCameraIndex:

        -1 = no camera observation
         0 = first camera in cameraChannels
         1 = second camera
         ...

    The compositor chooses the camera with the largest
    optical-axis score for overlapping valid cells.
    """

    sourceCameraIndex = ipmResult[
        "sourceCameraIndex"
    ]

    cameraChannels = ipmResult[
        "cameraChannels"
    ]

    # Shift the integer grid by +1 so display value 0 can mean:
    #
    #     no camera
    #
    # while camera indices 0..5 become display values 1..6.
    sourceDisplayGrid = (
        sourceCameraIndex
        + 1
    )

    sourceColors = [
        "black",
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown"
    ]

    sourceCmap = ListedColormap(
        sourceColors
    )

    sourceNorm = BoundaryNorm(
        np.arange(
            -0.5,
            len(
                sourceColors
            )
            + 0.5,
            1.0
        ),
        sourceCmap.N
    )

    axis.imshow(
        sourceDisplayGrid,
        origin="upper",
        cmap=sourceCmap,
        norm=sourceNorm,
        interpolation="nearest"
    )

    drawCarMarker(
        axis,
        bevResult
    )

    legendHandles = [
        Patch(
            facecolor="black",
            label="No camera"
        )
    ]

    for cameraIndex, cameraChannel in enumerate(
        cameraChannels
    ):

        legendHandles.append(
            Patch(
                facecolor=sourceColors[
                    cameraIndex
                    + 1
                ],
                label=cameraChannel
            )
        )

    axis.legend(
        handles=legendHandles,
        loc="lower right",
        fontsize=5.5,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.35,
        framealpha=0.90
    )

    axis.set_title(
        "Surround IPM - Source Camera",
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
    ipmResult,
    cameraLidarFusionResult,
    visualizationRangeM
):
    """
    Create the complete NavFusion surround sensor-fusion
    visualization.

    The original nine diagnostic panels are preserved.

    Three surround-camera panels are added so the complete
    perception/mapping flow is visible in one figure.

    Layout:

        Row 1:
            CAM_FRONT
            Raw LiDAR
            CAM_FRONT + projected LiDAR

        Row 2:
            YOLO + associated LiDAR
            Six-camera surround IPM
            Fused surround IPM + LiDAR + semantics

        Row 3:
            Density BEV
            Maximum-height BEV
            Mean-intensity BEV

        Row 4:
            Occupancy BEV
            Semantic BEV
            Surround-IPM source-camera map
    """

    figure, axes = plt.subplots(
        4,
        3,
        figsize=(
            24,
            30
        )
    )

    # ----------------------------------------------------------
    # Row 1: front-camera / raw sensor diagnostics
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
    # Row 2: perception + surround camera BEV
    # ----------------------------------------------------------

    drawVehicleFusion(
        axes[
            1,
            0
        ],
        sensorResult
    )

    drawSurroundCameraIpm(
        axes[
            1,
            1
        ],
        ipmResult,
        bevResult
    )

    drawSurroundCameraLidarFusion(
        axes[
            1,
            2
        ],
        cameraLidarFusionResult,
        semanticResult,
        bevResult
    )

    # ----------------------------------------------------------
    # Row 3: geometric LiDAR BEV channels
    # ----------------------------------------------------------

    drawLidarBev(
        axes[
            2,
            0
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
            2,
            1
        ],
        bevResult[
            "maxHeightGrid"
        ],
        bevResult,
        "Car-Frame LiDAR BEV - Maximum Height",
        cmap="viridis",
        colorbarLabel="Height above ground (m)"
    )

    drawLidarBev(
        axes[
            2,
            2
        ],
        bevResult[
            "meanIntensityGrid"
        ],
        bevResult,
        "Car-Frame LiDAR BEV - Mean Intensity",
        cmap="viridis",
        colorbarLabel="Mean intensity"
    )

    # ----------------------------------------------------------
    # Row 4: occupancy / semantics / camera ownership
    # ----------------------------------------------------------

    drawLidarBev(
        axes[
            3,
            0
        ],
        bevResult[
            "occupancyGrid"
        ],
        bevResult,
        "Car-Frame LiDAR BEV - Occupancy",
        cmap="gray"
    )

    drawSemanticBev(
        axes[
            3,
            1
        ],
        semanticResult,
        bevResult
    )

    drawSurroundCameraSourceMap(
        axes[
            3,
            2
        ],
        ipmResult,
        bevResult
    )

    # ----------------------------------------------------------
    # Reduce repeated BEV axis labels
    # ----------------------------------------------------------
    #
    # All BEV panels use the same coordinate convention:
    #
    #     horizontal -> BEV column (+y left)
    #     vertical   -> BEV row (+x forward)
    #
    # Repeating these labels on every subplot made neighboring
    # rows visually collide.

    # Row 2:
    # Keep the vertical coordinate label only on the first panel.
    axes[
        1,
        1
    ].set_ylabel(
        ""
    )

    axes[
        1,
        2
    ].set_ylabel(
        ""
    )

    # Row 3:
    # The bottom row already carries the common horizontal BEV
    # coordinate label, so suppress it here.
    for columnIndex in range(
        3
    ):
        axes[
            2,
            columnIndex
        ].set_xlabel(
            ""
        )

    axes[
        2,
        1
    ].set_ylabel(
        ""
    )

    axes[
        2,
        2
    ].set_ylabel(
        ""
    )

    # Row 4:
    # Keep horizontal labels because this is the final row.
    # Keep the vertical label only on the first panel.
    axes[
        3,
        1
    ].set_ylabel(
        ""
    )

    axes[
        3,
        2
    ].set_ylabel(
        ""
    )

    # ----------------------------------------------------------
    # Remove internal x-axis labels that still collide with the
    # titles of the row below.
    # ----------------------------------------------------------
    #
    # Row 1 center:
    #
    # Raw LiDAR is the only top-row panel with a metric x label.
    # Its coordinate meaning is already clear from its title and
    # tick values, so suppress the redundant x-axis text.
    axes[
        0,
        1
    ].set_xlabel(
        ""
    )

    # Row 2:
    #
    # The two surround BEV panels sit directly above the geometric
    # BEV row. Suppress their x-axis text so those labels cannot
    # collide with the row-3 titles.
    axes[
        1,
        1
    ].set_xlabel(
        ""
    )

    axes[
        1,
        2
    ].set_xlabel(
        ""
    )

    # ----------------------------------------------------------
    # Figure spacing
    # ----------------------------------------------------------
    #
    # The BEV panels are tall because the metric grid is:
    #
    #     350 rows x 250 columns
    #
    # Extra vertical padding keeps subplot titles, tick labels,
    # legends and colorbars from colliding.
    figure.tight_layout(
        rect=(
            0.02,
            0.02,
            0.95,
            0.965
        ),
        h_pad=7.5,
        w_pad=3.5
    )

    return figure