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
    displayDensity,
    carRow,
    carColumn,
    title
):
    """
    Draw one LiDAR BEV density image.

    displayDensity has already been log-scaled and normalized
    inside lidar_bev.py.
    """

    axis.imshow(
        displayDensity,
        origin="upper"
    )

    # Mark the LiDAR/car origin.
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


def createSensorFusionFigure(
    sensorResult,
    bevResult,
    visualizationRangeM
):
    """
    Create the six-panel NavFusion diagnostic visualization.

    Panels:

        1. Raw camera
        2. Raw LiDAR XY
        3. LiDAR projected onto camera
        4. YOLO + object-specific LiDAR
        5. RANSAC obstacles before self-return filtering
        6. RANSAC obstacles after self-return filtering
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

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(22, 14)
    )

    # Convert the 2 x 3 array of axes into:
    #
    # axes[0]
    # axes[1]
    # ...
    # axes[5]
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
    # Panel 5: RANSAC obstacles before self-return filtering
    # ----------------------------------------------------------

    # This layer contains:
    #
    # height-above-ground obstacle candidates
    #
    # but still includes any points generated by the car /
    # LiDAR mounting structure.
    drawLidarBev(
        axes[4],
        bevResult[
            "obstacleCandidateDisplayDensity"
        ],
        bevResult[
            "carRow"
        ],
        bevResult[
            "carColumn"
        ],
        "RANSAC Obstacles Before Self Filter"
    )

    # ----------------------------------------------------------
    # Panel 6: RANSAC obstacles after self-return filtering
    # ----------------------------------------------------------

    # This layer contains:
    #
    # obstacleCandidateMask
    # AND
    # NOT selfReturnMask
    #
    # so known car/sensor returns are removed from the external
    # obstacle representation.
    drawLidarBev(
        axes[5],
        bevResult[
            "obstacleDisplayDensity"
        ],
        bevResult[
            "carRow"
        ],
        bevResult[
            "carColumn"
        ],
        "RANSAC Obstacles After Self Filter"
    )

    figure.tight_layout()

    return figure

