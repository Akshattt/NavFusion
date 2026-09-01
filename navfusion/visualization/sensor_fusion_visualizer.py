import matplotlib.pyplot as plt


def drawVehicleFusion(
    axis,
    sensorResult
):
    """
    Draw YOLO bounding boxes and only the LiDAR points associated
    with each detected vehicle.
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

    # Do not draw every camera-visible LiDAR point in this panel.
    #
    # Only LiDAR points associated with a detected object are
    # displayed.
    for fusedObject in fusedObjects:

        x1, y1, x2, y2 = fusedObject[
            "box"
        ]

        width = (
            x2 - x1
        )

        height = (
            y2 - y1
        )

        # Draw the YOLO bounding box.
        rectangle = plt.Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            linewidth=2
        )

        axis.add_patch(
            rectangle
        )

        # cleanIndices contains LiDAR points that:
        #
        # 1. project inside the camera image,
        # 2. project inside the shrunk YOLO bounding box,
        # 3. remain after MAD depth-outlier filtering.
        cleanIndices = fusedObject[
            "cleanIndices"
        ]

        # Draw only LiDAR points associated with this object.
        #
        # alpha = 0.80 means 80% opaque,
        # therefore 20% transparent.
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
            s=12,
            cmap="viridis",
            alpha=0.80
        )

        # Example:
        #
        # truck 0.80
        # 10.07 m
        #
        # truck = YOLO predicted class
        # 0.80  = YOLO confidence
        # 10.07 = median filtered LiDAR/camera depth
        label = (
            f"{fusedObject['className']} "
            f"{fusedObject['confidence']:.2f}\n"
            f"{fusedObject['distanceM']:.2f} m"
        )

        axis.text(
            x1,
            max(
                15,
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
        "YOLO11 + LiDAR Vehicle Distance"
    )

    axis.axis(
        "off"
    )


def drawLidarBev(
    axis,
    bevGrid,
    carRow,
    carColumn,
    title,
    colorbarLabel=None
):
    """
    Draw one BEV channel.

    bevGrid may contain:

        density,
        maximum height,
        mean intensity,
        occupancy.

    All channels use the same BEV row/column coordinate system.
    """

    bevImage = axis.imshow(
        bevGrid,
        origin="upper"
    )

    # Mark the car/LiDAR origin.
    axis.scatter(
        carColumn,
        carRow,
        marker="x",
        s=80,
        label="Car"
    )

    axis.set_title(
        title
    )

    axis.set_xlabel(
        "BEV column"
    )

    axis.set_ylabel(
        "BEV row"
    )

    axis.legend()

    # Height and intensity have meaningful physical/numerical
    # values, so display their scales alongside the image.
    if colorbarLabel is not None:

        axis.figure.colorbar(
            bevImage,
            ax=axis,
            label=colorbarLabel
        )


def createSensorFusionFigure(
    sensorResult,
    bevResult,
    visualizationRangeM
):
    """
    Create the eight-panel NavFusion diagnostic visualization.

    Top row:

        1. Raw camera
        2. Raw LiDAR XY
        3. LiDAR projected onto camera
        4. YOLO + object-specific LiDAR

    Bottom row:

        5. Clean obstacle density
        6. Maximum obstacle height above ground
        7. Mean LiDAR intensity
        8. Binary occupancy

    The four bottom-row BEV channels are spatially aligned.
    """

    cameraImage = sensorResult[
        "cameraImage"
    ]

    lidarPointCloud = sensorResult[
        "lidarPointCloud"
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

    points = lidarPointCloud.points

    x = points[
        0,
        :
    ]

    y = points[
        1,
        :
    ]

    # Keep only the requested physical range for the raw XY
    # visualization.
    rangeMask = (
        (x > -visualizationRangeM)
        & (x < visualizationRangeM)
        & (y > -visualizationRangeM)
        & (y < visualizationRangeM)
    )

    xVisible = x[
        rangeMask
    ]

    yVisible = y[
        rangeMask
    ]

    # 2 rows x 4 columns = 8 diagnostic panels.
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(28, 14)
    )

    # Convert:
    #
    # axes[0, 0], axes[0, 1], ...
    #
    # into:
    #
    # axes[0], axes[1], ... axes[7]
    axes = axes.flatten()

    # ----------------------------------------------------------
    # Panel 1: Camera
    # ----------------------------------------------------------

    axes[0].imshow(
        cameraImage
    )

    axes[0].set_title(
        sensorResult[
            "cameraChannel"
        ]
    )

    axes[0].axis(
        "off"
    )

    # ----------------------------------------------------------
    # Panel 2: Raw LiDAR
    # ----------------------------------------------------------

    axes[1].scatter(
        yVisible,
        xVisible,
        s=0.5
    )

    axes[1].scatter(
        0,
        0,
        marker="x",
        s=80,
        label="LiDAR sensor"
    )

    axes[1].set_title(
        (
            f"{sensorResult['lidarChannel']} "
            f"- Raw Sensor-Frame XY"
        )
    )

    axes[1].set_xlabel(
        "y - left/right (m)"
    )

    axes[1].set_ylabel(
        "x - forward/backward (m)"
    )

    axes[1].set_xlim(
        -visualizationRangeM,
        visualizationRangeM
    )

    axes[1].set_ylim(
        -visualizationRangeM,
        visualizationRangeM
    )

    axes[1].set_aspect(
        "equal"
    )

    axes[1].grid(
        True
    )

    axes[1].legend()

    # ----------------------------------------------------------
    # Panel 3: Camera + all projected LiDAR
    # ----------------------------------------------------------

    axes[2].imshow(
        cameraImage
    )

    fusionScatter = axes[2].scatter(
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
        s=6,
        cmap="viridis",

        # 80% opaque = 20% transparent.
        alpha=0.80
    )

    axes[2].set_title(
        (
            f"{sensorResult['cameraChannel']} + "
            f"{sensorResult['lidarChannel']} Fusion"
        )
    )

    axes[2].axis(
        "off"
    )

    figure.colorbar(
        fusionScatter,
        ax=axes[2],
        label="Depth (m)"
    )

    # ----------------------------------------------------------
    # Panel 4: YOLO + object-specific LiDAR
    # ----------------------------------------------------------

    drawVehicleFusion(
        axes[3],
        sensorResult
    )

    # ----------------------------------------------------------
    # Panel 5: Clean obstacle density
    # ----------------------------------------------------------

    # obstacleDisplayDensity contains the RANSAC-ground-filtered
    # and self-return-filtered LiDAR obstacle density.
    #
    # It has already been log-scaled and normalized for display.
    drawLidarBev(
        axes[4],
        bevResult[
            "obstacleDisplayDensity"
        ],
        bevResult[
            "carRow"
        ],
        bevResult[
            "carColumn"
        ],
        "LiDAR BEV - Density"
    )

    # ----------------------------------------------------------
    # Panel 6: Maximum height above ground
    # ----------------------------------------------------------

    # For every occupied cell:
    #
    # maxHeightGrid[row, column]
    #
    # stores the highest clean obstacle return above the
    # RANSAC-estimated ground plane.
    drawLidarBev(
        axes[5],
        bevResult[
            "maxHeightGrid"
        ],
        bevResult[
            "carRow"
        ],
        bevResult[
            "carColumn"
        ],
        "LiDAR BEV - Maximum Height",
        colorbarLabel="Height above ground (m)"
    )

    # ----------------------------------------------------------
    # Panel 7: Mean LiDAR intensity
    # ----------------------------------------------------------

    # For every occupied cell:
    #
    # meanIntensityGrid[row, column]
    #
    # contains the average reflectivity/intensity of the clean
    # obstacle LiDAR returns in that cell.
    drawLidarBev(
        axes[6],
        bevResult[
            "meanIntensityGrid"
        ],
        bevResult[
            "carRow"
        ],
        bevResult[
            "carColumn"
        ],
        "LiDAR BEV - Mean Intensity",
        colorbarLabel="Mean LiDAR intensity"
    )

    # ----------------------------------------------------------
    # Panel 8: Binary occupancy
    # ----------------------------------------------------------

    # occupancyGrid contains:
    #
    # 0 = no clean obstacle return in this cell
    # 1 = at least one clean obstacle return
    drawLidarBev(
        axes[7],
        bevResult[
            "occupancyGrid"
        ],
        bevResult[
            "carRow"
        ],
        bevResult[
            "carColumn"
        ],
        "LiDAR BEV - Occupancy"
    )

    figure.tight_layout()

    return figure