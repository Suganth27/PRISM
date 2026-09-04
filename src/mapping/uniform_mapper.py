import numpy as np


class Uniform2DMapper:
    """
    Converts a 3D LiDAR point cloud into a
    uniform-resolution 2.5D semantic grid.
    """

    def __init__(self, resolution=0.5):

        self.resolution = resolution

    def map_points(self, points, labels=None):
        """
        Convert points into a uniform 2.5D map.

        Each cell stores:

            - height_min
            - height_max
            - height_mean
            - point_count
            - class_counts
            - dominant_class
        """

        cells = {}

        # Unknown semantic class if labels
        # are not supplied.
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

            cell_x = int(
                np.floor(
                    x / self.resolution
                )
            )

            cell_y = int(
                np.floor(
                    y / self.resolution
                )
            )

            key = (
                cell_x,
                cell_y
            )

            if key not in cells:

                cells[key] = {

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

                # Height statistics
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

                # Semantic statistics
                class_id = int(label)

                cell["class_counts"][
                    class_id
                ] = (
                    cell["class_counts"].get(
                        class_id,
                        0
                    ) + 1
                )

        # Calculate final statistics
        for cell in cells.values():

            cell["height_mean"] = (
                cell["height_sum"] /
                cell["point_count"]
            )

            del cell["height_sum"]

            cell["dominant_class"] = max(
                cell["class_counts"],
                key=cell["class_counts"].get
            )

        return cells