import numpy as np


class Adaptive2_5DMapper:

    def __init__(self):

        # Distance bands in metres.
        #
        # 0-10 m   -> 5 cm
        # 10-30 m  -> 10 cm
        # 30-60 m  -> 25 cm
        # 60-100 m -> 50 cm
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

        # Outside mapping range
        return None

    def map_points(self, points, labels=None):
        """
        Convert a 3D LiDAR point cloud into an
        adaptive-resolution 2.5D semantic map.

        Each cell stores:

        Spatial information:
            - resolution
            - x_min
            - x_max
            - y_min
            - y_max

        Height information:
            - height_min
            - height_max
            - height_mean

        Point information:
            - point_count

        Semantic information:
            - class_counts
            - dominant_class
        """

        cells = {}

        # If labels are not supplied, treat every
        # point as an unknown semantic class.
        if labels is None:

            labels = np.full(
                len(points),
                -1,
                dtype=np.int32
            )

        if len(points) != len(labels):

            raise ValueError(
                "Number of points and labels "
                "must be the same."
            )

        for point, label in zip(
            points,
            labels
        ):

            x, y, z = point

            resolution = self.get_resolution(
                x,
                y
            )

            # Ignore points outside
            # the mapping range.
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

            if key not in cells:

                # Spatial boundaries
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
                        int(label): 1
                    }
                }

            else:

                cell = cells[key]

                # ----------------------------
                # Height statistics
                # ----------------------------

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

                # ----------------------------
                # Semantic statistics
                # ----------------------------

                class_id = int(label)

                cell["class_counts"][
                    class_id
                ] = (
                    cell["class_counts"].get(
                        class_id,
                        0
                    ) + 1
                )

        # ----------------------------------------
        # Finalize cells
        # ----------------------------------------

        for cell in cells.values():

            # Mean height
            cell["height_mean"] = (
                cell["height_sum"] /
                cell["point_count"]
            )

            del cell["height_sum"]

            # Dominant semantic class
            cell["dominant_class"] = max(
                cell["class_counts"],
                key=cell["class_counts"].get
            )

        return cells