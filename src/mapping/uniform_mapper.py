import numpy as np


class Uniform2DMapper:
    """
    Converts a 3D LiDAR point cloud into a uniform-resolution
    2.5D elevation grid.
    """

    def __init__(self, resolution=0.5):
        self.resolution = resolution

    def map_points(self, points):
        """
        Parameters
        ----------
        points : np.ndarray
            Point cloud with shape (N, 3)
            columns = x, y, z

        Returns
        -------
        dict
            Dictionary containing the 2.5D cells.
        """

        cells = {}

        for x, y, z in points:

            cell_x = int(np.floor(x / self.resolution))
            cell_y = int(np.floor(y / self.resolution))

            key = (cell_x, cell_y)

            if key not in cells:
                cells[key] = {
                    "height_min": z,
                    "height_max": z,
                    "height_sum": z,
                    "point_count": 1
                }

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

        # Calculate mean height
        for cell in cells.values():
            cell["height_mean"] = (
                cell["height_sum"] /
                cell["point_count"]
            )

            del cell["height_sum"]

        return cells