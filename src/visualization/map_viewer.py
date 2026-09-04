import os

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


OUTPUT_DIR = "output"


# Semantic class IDs
GROUND = 0
STATIC_OBSTACLE = 1
DYNAMIC_OBJECT = 2
UNKNOWN = -1


def _prepare_output_directory():
    """Create the output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _build_patches(cells, resolution=None, adaptive=False):
    """
    Build map cell rectangles efficiently.

    Uniform map:
        key = (cell_x, cell_y)

    Adaptive map:
        key = (resolution, cell_x, cell_y)
    """

    patches = []
    classes = []

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

        classes.append(
            cell.get(
                "dominant_class",
                UNKNOWN
            )
        )

    return patches, classes


def _semantic_colors():
    """
    Create categorical colors for semantic classes.

    0 -> Ground
    1 -> Static obstacle
    2 -> Dynamic object
    """

    return ListedColormap([
        "#55a868",  # Ground
        "#dd8452",  # Static obstacle
        "#c44e52",  # Dynamic object
        "#808080",  # Unknown
    ])


def _class_to_color_index(class_id):
    """Convert semantic class ID to colormap index."""

    if class_id == GROUND:
        return 0

    if class_id == STATIC_OBSTACLE:
        return 1

    if class_id == DYNAMIC_OBJECT:
        return 2

    return 3


def _set_map_limits(
    ax,
    cells,
    resolution=None,
    adaptive=False,
    view_range=None
):
    """Set plot limits."""

    if view_range is not None:

        ax.set_xlim(
            -view_range,
            view_range
        )

        ax.set_ylim(
            -view_range,
            view_range
        )

        return

    all_x = []
    all_y = []

    for key in cells:

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


def _add_semantic_legend(ax):

    legend_elements = [

        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=10,
            markerfacecolor="#55a868",
            markeredgecolor="none",
            label="Ground"
        ),

        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=10,
            markerfacecolor="#dd8452",
            markeredgecolor="none",
            label="Static obstacle"
        ),

        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=10,
            markerfacecolor="#c44e52",
            markeredgecolor="none",
            label="Dynamic object"
        ),

        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="None",
            markersize=10,
            markeredgewidth=3,
            label="LiDAR"
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right"
    )


def _get_resolution_counts(cells):

    resolution_counts = {}

    for key in cells:

        resolution = key[0]

        resolution_counts[resolution] = (
            resolution_counts.get(
                resolution,
                0
            ) + 1
        )

    return resolution_counts


def show_uniform_map(
    cells,
    resolution=0.5
):
    """
    Display the uniform semantic 2.5D map.
    """

    if not cells:

        print(
            "No uniform cells to visualize."
        )

        return None

    _prepare_output_directory()

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    patches, classes = _build_patches(
        cells,
        resolution=resolution,
        adaptive=False
    )

    class_indices = [
        _class_to_color_index(
            class_id
        )
        for class_id in classes
    ]

    colormap = _semantic_colors()

    collection = PatchCollection(
        patches,
        cmap=colormap,
        edgecolor="none"
    )

    collection.set_array(
        class_indices
    )

    ax.add_collection(
        collection
    )

    ax.scatter(
        0,
        0,
        marker="x",
        s=100,
        linewidths=3
    )

    ax.set_title(
        f"PRISM - Uniform Semantic 2.5D Map "
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

    _add_semantic_legend(
        ax
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


def show_adaptive_map(
    cells,
    view_range=None,
    filename="adaptive_2_5d.png",
    title="PRISM - Adaptive Semantic 2.5D LiDAR Map"
):
    """
    Display the adaptive semantic 2.5D map.

    view_range:
        None -> full map
        number -> +/- that distance around LiDAR
    """

    if not cells:

        print(
            "No adaptive cells to visualize."
        )

        return None

    _prepare_output_directory()

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    patches, classes = _build_patches(
        cells,
        adaptive=True
    )

    class_indices = [
        _class_to_color_index(
            class_id
        )
        for class_id in classes
    ]

    colormap = _semantic_colors()

    collection = PatchCollection(
        patches,
        cmap=colormap,
        edgecolor="none"
    )

    collection.set_array(
        class_indices
    )

    ax.add_collection(
        collection
    )

    # LiDAR position
    ax.scatter(
        0,
        0,
        marker="x",
        s=120,
        linewidths=3
    )

    ax.set_title(
        title,
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
        adaptive=True,
        view_range=view_range
    )

    _add_semantic_legend(
        ax
    )

    # ----------------------------------------
    # Resolution statistics
    # ----------------------------------------

    resolution_counts = _get_resolution_counts(
        cells
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
        filename
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


def show_fovea_map(cells):
    """
    Display a near-field foveated view.

    This focuses on the high-resolution region
    around the LiDAR sensor.
    """

    return show_adaptive_map(
        cells,
        view_range=30,
        filename="adaptive_fovea_30m.png",
        title=(
            "PRISM - Foveated Adaptive 2.5D Map "
            "(30 m View)"
        )
    )


def show_maps_together(
    uniform_cells,
    adaptive_cells,
    resolution=0.5
):
    """
    Display the uniform map, full adaptive map,
    and near-field fovea map.
    """

    show_uniform_map(
        uniform_cells,
        resolution
    )

    show_adaptive_map(
        adaptive_cells
    )

    show_fovea_map(
        adaptive_cells
    )

    plt.show()