import numpy as np


class SyntheticLiDAR:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def generate_ground(self, x_range=(-20, 20), y_range=(-20, 20),
                        spacing=0.25):
        """Generate flat road/ground points."""

        x = np.arange(x_range[0], x_range[1], spacing)
        y = np.arange(y_range[0], y_range[1], spacing)

        xx, yy = np.meshgrid(x, y)

        # Small measurement noise
        zz = self.rng.normal(0, 0.01, xx.shape)

        points = np.column_stack((
            xx.ravel(),
            yy.ravel(),
            zz.ravel()
        ))

        return points

    def generate_box(self, center, size, spacing=0.15):
        """Generate points representing a box-shaped obstacle."""

        cx, cy, cz = center
        sx, sy, sz = size

        x = np.arange(cx - sx / 2, cx + sx / 2, spacing)
        y = np.arange(cy - sy / 2, cy + sy / 2, spacing)
        z = np.arange(cz, cz + sz, spacing)

        points = []

        # Bottom/top surfaces
        xx, yy = np.meshgrid(x, y)

        points.append(
            np.column_stack((
                xx.ravel(),
                yy.ravel(),
                np.full(xx.size, cz)
            ))
        )

        points.append(
            np.column_stack((
                xx.ravel(),
                yy.ravel(),
                np.full(xx.size, cz + sz)
            ))
        )

        # Side surfaces
        yy, zz = np.meshgrid(y, z)

        points.append(
            np.column_stack((
                np.full(yy.size, cx - sx / 2),
                yy.ravel(),
                zz.ravel()
            ))
        )

        points.append(
            np.column_stack((
                np.full(yy.size, cx + sx / 2),
                yy.ravel(),
                zz.ravel()
            ))
        )

        xx, zz = np.meshgrid(x, z)

        points.append(
            np.column_stack((
                xx.ravel(),
                np.full(xx.size, cy - sy / 2),
                zz.ravel()
            ))
        )

        points.append(
            np.column_stack((
                xx.ravel(),
                np.full(xx.size, cy + sy / 2),
                zz.ravel()
            ))
        )

        return np.vstack(points)

    def generate_frame(self, vehicle_x=8.0):
        """Generate one synthetic LiDAR frame."""

        ground = self.generate_ground()

        # Static obstacle / wall
        wall = self.generate_box(
            center=(12, 4, 1.5),
            size=(4, 0.5, 3),
        )

        # Dynamic vehicle
        vehicle = self.generate_box(
            center=(vehicle_x, -3, 1.0),
            size=(4, 2, 2),
        )

        points = np.vstack([
            ground,
            wall,
            vehicle
        ])

        return points


if __name__ == "__main__":
    lidar = SyntheticLiDAR()

    points = lidar.generate_frame()

    print("Synthetic LiDAR frame generated")
    print(f"Number of points: {len(points)}")
    print(f"Point cloud shape: {points.shape}")
    print()
    print("First 5 points:")
    print(points[:5])