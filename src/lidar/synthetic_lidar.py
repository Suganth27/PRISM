import numpy as np


class SyntheticLiDAR:

    # Semantic class IDs
    GROUND = 0
    STATIC_OBSTACLE = 1
    DYNAMIC_OBJECT = 2

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def generate_ground(
        self,
        x_range=(-100, 100),
        y_range=(-100, 100),
        spacing=0.5
    ):
        """Generate flat road/ground points."""

        x = np.arange(
            x_range[0],
            x_range[1],
            spacing
        )

        y = np.arange(
            y_range[0],
            y_range[1],
            spacing
        )

        xx, yy = np.meshgrid(
            x,
            y
        )

        # Small LiDAR measurement noise
        zz = self.rng.normal(
            0,
            0.01,
            xx.shape
        )

        points = np.column_stack((
            xx.ravel(),
            yy.ravel(),
            zz.ravel()
        ))

        labels = np.full(
            len(points),
            self.GROUND,
            dtype=np.int32
        )

        return points, labels

    def generate_box(
        self,
        center,
        size,
        spacing=0.15
    ):
        """Generate points representing a box-shaped obstacle."""

        cx, cy, cz = center
        sx, sy, sz = size

        x = np.arange(
            cx - sx / 2,
            cx + sx / 2,
            spacing
        )

        y = np.arange(
            cy - sy / 2,
            cy + sy / 2,
            spacing
        )

        z = np.arange(
            cz,
            cz + sz,
            spacing
        )

        points = []

        # Bottom surface
        xx, yy = np.meshgrid(
            x,
            y
        )

        points.append(
            np.column_stack((
                xx.ravel(),
                yy.ravel(),
                np.full(
                    xx.size,
                    cz
                )
            ))
        )

        # Top surface
        points.append(
            np.column_stack((
                xx.ravel(),
                yy.ravel(),
                np.full(
                    xx.size,
                    cz + sz
                )
            ))
        )

        # Left / right surfaces
        yy, zz = np.meshgrid(
            y,
            z
        )

        points.append(
            np.column_stack((
                np.full(
                    yy.size,
                    cx - sx / 2
                ),
                yy.ravel(),
                zz.ravel()
            ))
        )

        points.append(
            np.column_stack((
                np.full(
                    yy.size,
                    cx + sx / 2
                ),
                yy.ravel(),
                zz.ravel()
            ))
        )

        # Front / back surfaces
        xx, zz = np.meshgrid(
            x,
            z
        )

        points.append(
            np.column_stack((
                xx.ravel(),
                np.full(
                    xx.size,
                    cy - sy / 2
                ),
                zz.ravel()
            ))
        )

        points.append(
            np.column_stack((
                xx.ravel(),
                np.full(
                    xx.size,
                    cy + sy / 2
                ),
                zz.ravel()
            ))
        )

        return np.vstack(points)

    def generate_frame(
        self,
        vehicle_x=8.0
    ):
        """
        Generate one synthetic LiDAR frame.

        Returns:
            points:
                Nx3 array containing X, Y, Z.

            labels:
                N array containing semantic class IDs.

        Semantic classes:
            0 -> Ground
            1 -> Static obstacle
            2 -> Dynamic object
        """

        # ----------------------------------------
        # Ground
        # ----------------------------------------

        ground, ground_labels = self.generate_ground()

        # ----------------------------------------
        # Static obstacle: wall
        # ----------------------------------------

        wall = self.generate_box(
            center=(12, 4, 1.5),
            size=(4, 0.5, 3)
        )

        wall_labels = np.full(
            len(wall),
            self.STATIC_OBSTACLE,
            dtype=np.int32
        )

        # ----------------------------------------
        # Dynamic object: vehicle
        # ----------------------------------------

        vehicle = self.generate_box(
            center=(vehicle_x, -3, 1.0),
            size=(4, 2, 2)
        )

        vehicle_labels = np.full(
            len(vehicle),
            self.DYNAMIC_OBJECT,
            dtype=np.int32
        )

        # ----------------------------------------
        # Combine everything
        # ----------------------------------------

        points = np.vstack([
            ground,
            wall,
            vehicle
        ])

        labels = np.concatenate([
            ground_labels,
            wall_labels,
            vehicle_labels
        ])

        return points, labels