import numpy as np


# Semantic class IDs
GROUND = 0
STATIC_OBSTACLE = 1
DYNAMIC_OBJECT = 2


class Adaptive2_5DMapper:

    def __init__(self):

        # Distance bands in metres.
        #
        # 0-10 m    -> 5 cm
        # 10-30 m   -> 10 cm
        # 30-60 m   -> 25 cm
        # 60-100 m  -> 50 cm

        self.distance_bands = [
            (10.0, 0.05),
            (30.0, 0.10),
            (60.0, 0.25),
            (100.0, 0.50),
        ]

    def get_resolution(self, x, y):
        """
        Select mapping resolution based on
        horizontal distance from the LiDAR.
        """

        distance = np.sqrt(
            x * x + y * y
        )

        for max_distance, resolution in self.distance_bands:

            if distance <= max_distance:
                return resolution

        return None

    def map_points(
        self,
        points,
        labels=None
    ):
        """
        Convert a 3D LiDAR point cloud into an
        adaptive-resolution 2.5D map.

        Parameters
        ----------
        points : numpy.ndarray
            Nx3 array containing x, y, z.

        labels : numpy.ndarray or None
            N semantic labels corresponding to
            the input points.

            0 = Ground
            1 = Static obstacle
            2 = Dynamic object

        Each cell stores:

        - spatial bounds
        - resolution
        - height_min
        - height_max
        - height_mean
        - point_count
        - class_counts
        - dominant_class
        """

        if labels is not None:

            if len(labels) != len(points):

                raise ValueError(
                    "Number of labels must match "
                    "number of LiDAR points."
                )

        cells = {}

        for index, point in enumerate(points):

            x, y, z = point

            resolution = self.get_resolution(
                x,
                y
            )

            # Ignore points outside the
            # 100 metre mapping range.
            if resolution is None:
                continue

            # Determine grid cell.
            cell_x = int(
                np.floor(
                    x / resolution
                )
            )

            cell_y = int(
                np.floor(
                    y / resolution
                )
            )

            # Resolution is part of the key.
            key = (
                round(resolution, 3),
                cell_x,
                cell_y
            )

            # Get semantic label.
            if labels is not None:

                semantic_class = int(
                    labels[index]
                )

            else:

                semantic_class = GROUND

            # ----------------------------------------
            # Create new cell
            # ----------------------------------------

            if key not in cells:

                x_min = (
                    cell_x *
                    resolution
                )

                x_max = (
                    x_min +
                    resolution
                )

                y_min = (
                    cell_y *
                    resolution
                )

                y_max = (
                    y_min +
                    resolution
                )

                cells[key] = {

                    "resolution": resolution,

                    "x_min": x_min,
                    "x_max": x_max,

                    "y_min": y_min,
                    "y_max": y_max,

                    "height_min": z,
                    "height_max": z,
                    "height_sum": z,

                    "point_count": 1,

                    "class_counts": {
                        GROUND: 0,
                        STATIC_OBSTACLE: 0,
                        DYNAMIC_OBJECT: 0
                    }
                }

            # ----------------------------------------
            # Update existing cell
            # ----------------------------------------

            else:

                cell = cells[key]

                cell["height_min"] = min(
                    cell["height_min"],
                    z
                )

                cell["height_max"] = max(
                    cell["height_max"],
                    z
                )

                cell["height_sum"] += z

                cell["point_count"] += 1

            # Record semantic class.
            if semantic_class in (
                GROUND,
                STATIC_OBSTACLE,
                DYNAMIC_OBJECT
            ):

                cells[key]["class_counts"][
                    semantic_class
                ] += 1

        # ----------------------------------------
        # Finalize cells
        # ----------------------------------------

        for cell in cells.values():

            # Mean elevation
            cell["height_mean"] = (
                cell["height_sum"] /
                cell["point_count"]
            )

            del cell["height_sum"]

            # Determine dominant semantic class
            class_counts = cell["class_counts"]

            cell["dominant_class"] = max(
                class_counts,
                key=class_counts.get
            )

        return cells