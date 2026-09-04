import open3d as o3d

from src.lidar.synthetic_lidar import SyntheticLiDAR
from src.mapping.uniform_mapper import Uniform2DMapper
from src.mapping.adaptive_mapper import Adaptive2_5DMapper
from src.visualization.map_viewer import show_maps_together


def main():

    # ----------------------------------------
    # Generate synthetic LiDAR frame
    # ----------------------------------------

    lidar = SyntheticLiDAR()

    points, labels = lidar.generate_frame()

    print("=" * 60)
    print("PRISM - LiDAR Perception Prototype")
    print("=" * 60)

    print(
        f"LiDAR points generated : {len(points)}"
    )

    print(
        f"Point cloud shape      : {points.shape}"
    )

    # ----------------------------------------
    # Semantic information
    # ----------------------------------------

    ground_count = (
        labels == SyntheticLiDAR.GROUND
    ).sum()

    static_count = (
        labels == SyntheticLiDAR.STATIC_OBSTACLE
    ).sum()

    dynamic_count = (
        labels == SyntheticLiDAR.DYNAMIC_OBJECT
    ).sum()

    print()
    print("SEMANTIC CLASSES")
    print("-" * 40)

    print(
        f"Ground points          : {ground_count}"
    )

    print(
        f"Static obstacle points : {static_count}"
    )

    print(
        f"Dynamic object points  : {dynamic_count}"
    )

    # ----------------------------------------
    # Uniform 2.5D mapping
    # ----------------------------------------

    resolution = 0.5

    mapper = Uniform2DMapper(
        resolution=resolution
    )

    cells = mapper.map_points(
        points,
        labels
    )

    print()
    print("2.5D MAP")
    print("-" * 40)

    print(
        f"Grid resolution        : "
        f"{resolution} m"
    )

    print(
        f"Active cells           : "
        f"{len(cells)}"
    )

    # ----------------------------------------
    # Adaptive 2.5D mapping
    # ----------------------------------------

    adaptive_mapper = Adaptive2_5DMapper()

    adaptive_cells = adaptive_mapper.map_points(
        points,
        labels
    )

    # ----------------------------------------
    # Adaptive mapping validation
    # ----------------------------------------

    mapped_points = sum(
        cell["point_count"]
        for cell in adaptive_cells.values()
    )

    total_points = len(points)

    coverage = (
        mapped_points /
        total_points *
        100
        if total_points > 0
        else 0
    )

    print()
    print("ADAPTIVE MAPPING VALIDATION")
    print("----------------------------------------")

    print(
        f"Input LiDAR points     : "
        f"{total_points}"
    )

    print(
        f"Mapped LiDAR points    : "
        f"{mapped_points}"
    )

    print(
        f"Point coverage         : "
        f"{coverage:.2f}%"
    )

    print(
        f"Points not mapped      : "
        f"{total_points - mapped_points}"
    )

    # ----------------------------------------
    # Adaptive resolution statistics
    # ----------------------------------------

    print()
    print("ADAPTIVE 2.5D MAP")
    print("-" * 40)

    print(
        f"Adaptive cells         : "
        f"{len(adaptive_cells)}"
    )

    resolution_counts = {}

    for cell in adaptive_cells.values():

        cell_resolution = cell[
            "resolution"
        ]

        resolution_counts[
            cell_resolution
        ] = (
            resolution_counts.get(
                cell_resolution,
                0
            ) + 1
        )

    for cell_resolution, count in sorted(
        resolution_counts.items()
    ):

        print(
            f"{cell_resolution * 100:.0f} cm cells : "
            f"{count}"
        )

    # ----------------------------------------
    # Sample cells
    # ----------------------------------------

    print()
    print("Sample cells:")

    for i, (key, cell) in enumerate(
        cells.items()
    ):

        print(
            f"Cell {key}: "
            f"height={cell['height_mean']:.2f}m, "
            f"points={cell['point_count']}"
        )

        if i >= 4:
            break

    # ----------------------------------------
    # Raw 3D LiDAR visualization
    # ----------------------------------------

    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    print()
    print(
        "Opening raw 3D LiDAR visualization..."
    )

    o3d.visualization.draw_geometries(
        [cloud],
        window_name="PRISM - Raw 3D LiDAR"
    )

    # ----------------------------------------
    # 2.5D map visualization
    # ----------------------------------------

    print(
        "Opening uniform and adaptive "
        "2.5D maps..."
    )

    show_maps_together(
        cells,
        adaptive_cells,
        resolution
    )


if __name__ == "__main__":
    main()