import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def show_adaptive_map(cells):
    """
    Visualize the adaptive 2.5D map.

    Each rectangle represents one adaptive cell.
    Rectangle size = spatial resolution.
    Rectangle color = mean elevation.
    """

    if not cells:
        print("No adaptive cells to visualize.")
        return

    fig, ax = plt.subplots(figsize=(11, 9))

    # Find height range
    heights = [
        cell["height_mean"]
        for cell in cells.values()
    ]

    min_height = min(heights)
    max_height = max(heights)

    # Avoid zero-width normalization
    if min_height == max_height:
        max_height = min_height + 1.0

    norm = Normalize(
        vmin=min_height,
        vmax=max_height
    )

    colormap = cm.viridis

    # Draw every adaptive cell
    for (resolution, cell_x, cell_y), cell in cells.items():

        # Convert grid index back into world coordinates
        x = cell_x * resolution
        y = cell_y * resolution

        rectangle = Rectangle(
            (x, y),
            resolution,
            resolution,
            facecolor=colormap(
                norm(cell["height_mean"])
            ),
            edgecolor="black",
            linewidth=0.15
        )

        ax.add_patch(rectangle)

    # --------------------------------------------------
    # Draw LiDAR position
    # --------------------------------------------------

    ax.scatter(
        0,
        0,
        marker="x",
        s=100,
        linewidths=3,
        label="LiDAR"
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_title(
        "PRISM - Adaptive 2.5D LiDAR Map",
        fontsize=16
    )

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")

    ax.set_aspect("equal")

    # Automatically determine limits
    all_x = []
    all_y = []

    for resolution, cell_x, cell_y in cells.keys():

        all_x.append(cell_x * resolution)
        all_x.append((cell_x + 1) * resolution)

        all_y.append(cell_y * resolution)
        all_y.append((cell_y + 1) * resolution)

    margin = 1.0

    ax.set_xlim(
        min(all_x) - margin,
        max(all_x) + margin
    )

    ax.set_ylim(
        min(all_y) - margin,
        max(all_y) + margin
    )

    # --------------------------------------------------
    # Color bar
    # --------------------------------------------------

    scalar_map = cm.ScalarMappable(
        norm=norm,
        cmap=colormap
    )

    scalar_map.set_array([])

    colorbar = fig.colorbar(
        scalar_map,
        ax=ax
    )

    colorbar.set_label(
        "Mean Height (m)"
    )

    ax.legend()

    plt.tight_layout()

    plt.show()