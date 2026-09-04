import numpy as np


class Adaptive2_5DMapper:

    def __init__(self):

        # Distance limits in metres
        self.distance_bands = [
            (10.0, 0.05),   # 0-10m   -> 5 cm
            (30.0, 0.10),   # 10-30m  -> 10 cm
            (60.0, 0.25),   # 30-60m  -> 25 cm
            (100.0, 0.50),  # 60-100m -> 50 cm
        ]

    def get_resolution(self, x, y):

        distance = np.sqrt(x * x + y * y)

        for max_distance, resolution in self.distance_bands:

            if distance <= max_distance:
                return resolution

        return None

    def map_points(self, points):

        cells = {}

        for x, y, z in points:

            resolution = self.get_resolution(x, y)

            # Ignore points outside mapping range
            if resolution is None:
                continue

            cell_x = int(np.floor(x / resolution))
            cell_y = int(np.floor(y / resolution))

            # Resolution is part of the key.
            #
            # This prevents a 5cm cell and a 10cm cell
            # from accidentally being treated as the same cell.
            key = (
                round(resolution, 3),
                cell_x,
                cell_y
            )

            if key not in cells:

                cells[key] = {
                    "resolution": resolution,
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

        # Calculate mean elevation
        for cell in cells.values():

            cell["height_mean"] = (
                cell["height_sum"] /
                cell["point_count"]
            )

            del cell["height_sum"]

        return cells