import os

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize


OUTPUT_DIR = "output"


def _prepare_output_directory():
    """Create the output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _get_height_normalization(cells):
    """Create a common height normalization for a map."""

    heights = [
        cell["height_mean"]
        for cell in cells.values()
    ]

    min_height = min(heights)
    max_height = max(heights)

    if min_height == max_height:
        max_height = min_height + 1.0

    return Normalize(
        vmin=min_height,
        vmax=max_height
    )


def _build_patches(cells, resolution=None, adaptive=False):
    """
    Build map cell rectangles efficiently.

    Uniform map:
        Cell key = (cell_x, cell_y)
        Resolution comes from the function argument.

    Adaptive map:
        Cell key = (resolution, cell_x, cell_y)
        Resolution comes from the cell key.
    """

    patches = []
    values = []

    for key, cell in cells.items():

        if adaptive:

            cell_resolution, cell_x, cell_y = key

        else:

            cell_x, cell_y = key
            cell_resolution = resolution

        x = cell_x * cell_resolution
        y = cell_y * cell_resolution

        patches.append(
            Rectangle(
                (x, y),
                cell_resolution,
                cell_resolution
            )
        )

        values.append(
            cell["height_mean"]
        )

    return patches, values


def _set_map_limits(ax, cells, resolution=None, adaptive=False):
    """Set plot limits based on map cell boundaries."""

    all_x = []
    all_y = []

    for key, cell in cells.items():

        if adaptive:

            cell_resolution, cell_x, cell_y = key

        else:

            cell_x, cell_y = key
            cell_resolution = resolution

        x = cell_x * cell_resolution
        y = cell_y * cell_resolution

        all_x.extend([
            x,
            x + cell_resolution
        ])

        all_y.extend([
            y,
            y + cell_resolution
        ])

    margin = 1.0

    ax.set_xlim(
        min(all_x) - margin,
        max(all_x) + margin
    )

    ax.set_ylim(
        min(all_y) - margin,
        max(all_y) + margin
    )


def show_uniform_map(cells, resolution=0.5):
    """
    Visualize the uniform 2.5D map efficiently.
    """

    if not cells:
        print("No uniform cells to visualize.")
        return None

    _prepare_output_directory()

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    norm = _get_height_normalization(
        cells
    )

    patches, values = _build_patches(
        cells,
        resolution=resolution,
        adaptive=False
    )

    collection = PatchCollection(
        patches,
        cmap="viridis",
        norm=norm,
        edgecolor="none"
    )

    collection.set_array(values)

    ax.add_collection(
        collection
    )

    # LiDAR origin
    ax.scatter(
        0,
        0,
        marker="x",
        s=100,
        linewidths=3,
        label="LiDAR"
    )

    ax.set_title(
        f"PRISM - Uniform 2.5D Map "
        f"({resolution * 100:.0f} cm)",
        fontsize=16
    )

    ax.set_xlabel(
        "X position (m)"
    )

    ax.set_ylabel(
        "Y position (m)"
    )

    ax.set_aspect(
        "equal"
    )

    _set_map_limits(
        ax,
        cells,
        resolution=resolution,
        adaptive=False
    )

    colorbar = fig.colorbar(
        collection,
        ax=ax
    )

    colorbar.set_label(
        "Mean Height (m)"
    )

    ax.legend(
        loc="upper right"
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "uniform_2_5d.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Saved uniform map     : {output_path}"
    )

    return fig


def show_adaptive_map(cells):
    """
    Visualize the adaptive 2.5D map efficiently.

    Cell size represents spatial resolution.
    Color represents mean elevation.
    """

    if not cells:
        print("No adaptive cells to visualize.")
        return None

    _prepare_output_directory()

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    norm = _get_height_normalization(
        cells
    )

    patches, values = _build_patches(
        cells,
        adaptive=True
    )

    collection = PatchCollection(
        patches,
        cmap="viridis",
        norm=norm,
        edgecolor="none"
    )

    collection.set_array(values)

    ax.add_collection(
        collection
    )

    # LiDAR origin
    ax.scatter(
        0,
        0,
        marker="x",
        s=100,
        linewidths=3,
        label="LiDAR"
    )

    ax.set_title(
        "PRISM - Adaptive 2.5D LiDAR Map",
        fontsize=16
    )

    ax.set_xlabel(
        "X position (m)"
    )

    ax.set_ylabel(
        "Y position (m)"
    )

    ax.set_aspect(
        "equal"
    )

    _set_map_limits(
        ax,
        cells,
        adaptive=True
    )

    colorbar = fig.colorbar(
        collection,
        ax=ax
    )

    colorbar.set_label(
        "Mean Height (m)"
    )

    ax.legend(
        loc="upper right"
    )

    # ----------------------------------------
    # Resolution statistics
    # ----------------------------------------

    resolution_counts = {}

    for key in cells:

        resolution = key[0]

        resolution_counts[resolution] = (
            resolution_counts.get(
                resolution,
                0
            ) + 1
        )

    resolution_text = "\n".join(
        f"{int(resolution * 100)} cm : "
        f"{count:,} cells"
        for resolution, count
        in sorted(
            resolution_counts.items()
        )
    )

    ax.text(
        0.02,
        0.98,
        resolution_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85
        )
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "adaptive_2_5d.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Saved adaptive map    : {output_path}"
    )

    return fig


def show_maps_together(
    uniform_cells,
    adaptive_cells,
    resolution=0.5
):
    """
    Display uniform and adaptive maps together.
    """

    show_uniform_map(
        uniform_cells,
        resolution
    )

    show_adaptive_map(
        adaptive_cells
    )

    plt.show()