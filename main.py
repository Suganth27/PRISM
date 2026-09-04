import open3d as o3d

from src.lidar.synthetic_lidar import SyntheticLiDAR
from src.mapping.uniform_mapper import Uniform2DMapper
from src.mapping.adaptive_mapper import Adaptive2_5DMapper
from src.visualization.map_viewer import show_adaptive_map


def main():

    # --------------------------------------------------
    # 1. Generate synthetic LiDAR
    # --------------------------------------------------

    lidar = SyntheticLiDAR()

    points = lidar.generate_frame()

    print("=" * 60)
    print("PRISM - LiDAR Perception Prototype")
    print("=" * 60)

    print(f"LiDAR points generated : {len(points)}")
    print(f"Point cloud shape      : {points.shape}")

    # --------------------------------------------------
    # 2. Create uniform 2.5D map
    # --------------------------------------------------

    resolution = 0.5

    mapper = Uniform2DMapper(
        resolution=resolution
    )

    cells = mapper.map_points(points)

    print()
    print("2.5D MAP")
    print("-" * 40)

    print(f"Grid resolution        : {resolution} m")
    print(f"Active cells           : {len(cells)}")

    # --------------------------------------------------
    # 3. Create adaptive 2.5D map
    # --------------------------------------------------

    adaptive_mapper = Adaptive2_5DMapper()

    adaptive_cells = adaptive_mapper.map_points(points)

    print()
    print("ADAPTIVE 2.5D MAP")
    print("-" * 40)

    print(f"Adaptive cells         : {len(adaptive_cells)}")

    # Count cells at each resolution
    resolution_counts = {}

    for cell in adaptive_cells.values():

        cell_resolution = cell["resolution"]

        resolution_counts[cell_resolution] = (
            resolution_counts.get(cell_resolution, 0) + 1
        )

    for cell_resolution, count in sorted(
        resolution_counts.items()
    ):

        print(
            f"{cell_resolution * 100:.0f} cm cells : "
            f"{count}"
        )

    # --------------------------------------------------
    # 4. Show some uniform-map cell information
    # --------------------------------------------------

    print()
    print("Sample cells:")

    for i, (key, cell) in enumerate(cells.items()):

        print(
            f"Cell {key}: "
            f"height={cell['height_mean']:.2f}m, "
            f"points={cell['point_count']}"
        )

        if i >= 4:
            break

    # --------------------------------------------------
    # 5. Show raw 3D LiDAR
    # --------------------------------------------------

    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(points)

    print()
    print("Opening raw 3D LiDAR visualization...")

    o3d.visualization.draw_geometries(
        [cloud],
        window_name="PRISM - Raw 3D LiDAR"
    )

    # --------------------------------------------------
    # 6. Show uniform 2.5D elevation map
    # --------------------------------------------------

    print("Opening adaptive 2.5D map...")

    show_adaptive_map(
        adaptive_cells
    )


if __name__ == "__main__":
    main()