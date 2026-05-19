from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Dict, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass


def make_cmap_from_base(
    label: np.ndarray, base_cmap: str = "tab10", alpha: float = 0.5
) -> mcolors.ListedColormap:
    label = np.asarray(label)
    max_label = int(label.max()) if label.size > 0 else 0
    num_classes = max_label + 1

    base = plt.get_cmap(base_cmap)
    colors = [(0.0, 0.0, 0.0, 0.0)]  # class 0 -> transparent
    for idx in range(1, num_classes):
        r, g, b, _ = base(idx % base.N)
        colors.append((r, g, b, alpha))
    return mcolors.ListedColormap(colors)


def figure_overlay_label_on_slices(
    volume: sitk.Image,
    label: sitk.Image,
    labelname: Optional[Sequence[str]] = None,
    color: str = "tab10",
    alpha: float = 0.25,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    image = sitk.GetArrayFromImage(volume)
    mask = sitk.GetArrayFromImage(label)
    if image.shape != mask.shape:
        raise ValueError("`volume` and `label` must have the same shape.")
    if image.ndim != 3:
        raise ValueError("`volume` and `label` must have shape (X, Y, Z).")

    labels = [int(v) for v in np.unique(mask) if v > 0]
    cmap = make_cmap_from_base(mask, base_cmap=color, alpha=alpha)
    vmin, vmax = float(image.min()), float(image.max())

    if len(labels) <= 1:
        coords = np.argwhere(mask > 0)
        if coords.size > 0:
            center = np.median(coords, axis=0).astype(int)
        else:
            center = np.array(image.shape) // 2

        x_idx = int(np.clip(center[0], 0, image.shape[0] - 1))
        y_idx = int(np.clip(center[1], 0, image.shape[1] - 1))
        z_idx = int(np.clip(center[2], 0, image.shape[2] - 1))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = np.atleast_1d(axes)

        axes[0].imshow(image[x_idx, :, :], vmin=vmin, vmax=vmax, cmap="gray")
        axes[0].imshow(mask[x_idx, :, :], cmap=cmap)
        axes[0].set_title(f"X slice {x_idx}")

        axes[1].imshow(image[:, y_idx, :], vmin=vmin, vmax=vmax, cmap="gray")
        axes[1].imshow(mask[:, y_idx, :], cmap=cmap)
        axes[1].set_title(f"Y slice {y_idx}")

        axes[2].imshow(image[:, :, z_idx], vmin=vmin, vmax=vmax, cmap="gray")
        axes[2].imshow(mask[:, :, z_idx], cmap=cmap)
        axes[2].set_title(f"Z slice {z_idx}")

        for ax in axes:
            ax.axis("off")
    else:
        target_labels = labels
        n_panels = len(target_labels)
        if labelname is None or len(labelname) == 0:
            titles = [f"Label {target}" for target in target_labels]
        else:
            titles = [
                labelname[i] if i < len(labelname) else f"Label {target}"
                for i, target in enumerate(target_labels)
            ]

        fig, axes = plt.subplots(1, n_panels, figsize=(max(5, 3 * n_panels), 5))
        axes = np.atleast_1d(axes)

        for i, target in enumerate(target_labels):
            idx_is = np.where(mask == target)[0]
            sl_idx = int(np.median(idx_is)) if idx_is.size > 0 else int(image.shape[0] / 2)
            axes[i].imshow(image[sl_idx], vmin=vmin, vmax=vmax, cmap="gray")
            axes[i].imshow(mask[sl_idx] == target, cmap=cmap)
            axes[i].set_title(titles[i])
            axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path))
    if show:
        plt.show()
    else:
        plt.close(fig)


def figure_overlay_label_reference_slice(
    volume: sitk.Image,
    label: sitk.Image,
    reference: sitk.Image,
    slice_idx: int = 100,
    labelname: Optional[Sequence[str]] = None,
    color: str = "tab10",
    alpha: float = 0.25,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    image = sitk.GetArrayFromImage(volume)
    masks = sitk.GetArrayFromImage(label)
    ref = sitk.GetArrayFromImage(reference) * (int(masks.max()) + 1)

    if slice_idx == 0 or slice_idx >= image.shape[0]:
        slice_idx = int(image.shape[0] / 2)
        title = "No Reference slice"
    else:
        title = "Reference slice"

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(image[slice_idx], cmap="gray")
    if ref.max() > 0:
        ref_cmap = make_cmap_from_base(ref[slice_idx], base_cmap="RdGy", alpha=0.75)
        plt.imshow(ref[slice_idx], cmap=ref_cmap)

    label_cmap = make_cmap_from_base(masks[slice_idx], base_cmap=color, alpha=alpha)
    plt.imshow(masks[slice_idx], cmap=label_cmap)

    max_label = int(masks.max())
    if max_label > 0:
        cbar = plt.colorbar(ticks=np.arange(max_label + 1), fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels([])
        centers = np.linspace(0, max_label, (max_label + 1) * 2 + 1)[1::2]
        cbar.set_ticks(centers[1:])
        if labelname is not None:
            cbar.set_ticklabels(labelname[: len(centers) - 1])

    plt.tight_layout()
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(str(save_path))
    if show:
        plt.show()
    else:
        plt.close(fig)


def figure_slices_with_umbilicus(
    volume: sitk.Image,
    umbilicus_coord: np.ndarray,
    color: str = "autumn_r",
    alpha: float = 0.75,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    image = sitk.GetArrayFromImage(volume)
    msize = max(1, int(image.shape[1] / 100))
    pos_x, pos_y, pos_z = np.asarray(umbilicus_coord, dtype=int)

    pos_x = int(np.clip(pos_x, 0, image.shape[0] - 1))
    pos_y = int(np.clip(pos_y, 0, image.shape[1] - 1))
    pos_z = int(np.clip(pos_z, 0, image.shape[2] - 1))

    mask = np.zeros(image.shape, dtype=int)
    mask[
        max(0, pos_x - msize) : pos_x + msize,
        max(0, pos_y - msize) : pos_y + msize,
        max(0, pos_z - msize) : pos_z + msize,
    ] = 1
    cmap = make_cmap_from_base(mask, base_cmap=color, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(image[pos_x, :, :], cmap="gray")
    axes[0].imshow(mask[pos_x, :, :], alpha=alpha, cmap=cmap)
    axes[1].imshow(image[:, pos_y, :], cmap="gray")
    axes[1].imshow(mask[:, pos_y, :], alpha=alpha, cmap=cmap)
    axes[2].imshow(image[:, :, pos_z], cmap="gray")
    axes[2].imshow(mask[:, :, pos_z], alpha=alpha, cmap=cmap)
    axes[0].axis("off")
    axes[1].axis("off")
    axes[2].axis("off")
    plt.tight_layout()
    plt.suptitle(f"Point X/Y/Z: {pos_x}, {pos_y}, {pos_z}")

    if save_dir:
        plt.savefig(str(save_dir))
    if show:
        plt.show()
    else:
        plt.close(fig)


def figure_slices_with_landmarks(
    volume: sitk.Image,
    landmark_coord: np.ndarray,
    color: str = "autumn_r",
    alpha: float = 0.75,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    image = sitk.GetArrayFromImage(volume)
    points = np.asarray(landmark_coord, dtype=int)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("`landmark_coord` must be an array with shape (N, >=3).")

    msize = max(1, int(image.shape[1] / 100))
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))

    for j in range(5):
        start = j * 5
        end = (j + 1) * 5
        if end > len(points):
            axes[j].axis("off")
            continue

        pset = points[start:end, 1:]
        si = int(np.clip(points[start, 0], 0, image.shape[0] - 1))
        slice_image = image[si, :, :]
        mask = np.zeros(slice_image.shape, dtype=int)
        for k in range(len(pset)):
            r, c = pset[k][0], pset[k][1]
            r = int(np.clip(r, 0, mask.shape[0] - 1))
            c = int(np.clip(c, 0, mask.shape[1] - 1))
            mask[max(0, r - msize) : r + msize, max(0, c - msize) : c + msize] = 1

        cmap = make_cmap_from_base(mask, base_cmap=color, alpha=alpha)
        axes[j].imshow(slice_image, vmin=float(image.min()), vmax=float(image.max()), cmap="gray")
        axes[j].imshow(mask, alpha=alpha, cmap=cmap)
        axes[j].axis("off")

    plt.tight_layout()
    plt.suptitle("Landmark position")
    if save_dir:
        plt.savefig(str(save_dir))
    if show:
        plt.show()
    else:
        plt.close(fig)


def figure_overlay_tissue_on_slices(
    image: np.ndarray,
    mask: np.ndarray,
    color: str = "tab10",
    alpha: float = 0.25,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    image = np.asarray(image)
    mask = np.asarray(mask)
    if image.shape != mask.shape:
        raise ValueError("`image` and `mask` must have the same shape.")
    if image.ndim != 3:
        raise ValueError("`image` and `mask` must have shape (N, H, W).")

    n_slices = min(5, image.shape[0])
    cmap = make_cmap_from_base(mask, base_cmap=color, alpha=alpha)
    fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
    axes = np.atleast_1d(axes)
    vmin, vmax = float(image.min()), float(image.max())

    for i in range(n_slices):
        axes[i].imshow(image[i], vmin=vmin, vmax=vmax, cmap="gray")
        axes[i].imshow(mask[i], cmap=cmap)
        axes[i].axis("off")

    fig.suptitle("Body Compositions [Fat/Muscle]", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(str(save_path))
    if show:
        plt.show()
    else:
        plt.close(fig)


def figure_patch_from_image(
    patch: np.ndarray,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    patch = np.asarray(patch)
    if patch.ndim != 3:
        raise ValueError("`patch` must have shape (D, H, W).")

    fig, axes = plt.subplots(3, 2, figsize=(15, 8))
    axes = axes.ravel()
    vmin, vmax = float(patch.min()), float(patch.max())

    if patch.shape[0] >= 250:
        sl_idx = [0, 50, 100, 150, 200, 249]
    else:
        sl_idx = np.linspace(0, patch.shape[0] - 1, 6, dtype=int).tolist()

    for i, idx in enumerate(sl_idx):
        axes[i].imshow(patch[idx], vmin=vmin, vmax=vmax, cmap="gray")
        axes[i].set_title(f"slice_index = {idx}")
        axes[i].axis("off")

    d, h, w = patch.shape
    fig.suptitle(f"Patch [size {d} * {h} * {w}]", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(str(save_path))
    if show:
        plt.show()
    else:
        plt.close(fig)


def figure_3d_region_with_centerline(
    volume: sitk.Image,
    centerline_zyx: np.ndarray,
    subsample: int = 5,
    figsize: Tuple[int, int] = (14, 12),
    show: bool = True,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Create 3D QC visualization of pancreas subregions
    and centerline in four standard views.

    Views
    -----
    - Axial
    - Coronal
    - Sagittal
    - Oblique

    Parameters
    ----------
    volume : sitk.Image
        3D region label mask.

        Values:
            0 = background
            1..N = pancreas subregions

    centerline_zyx : np.ndarray
        Smoothed pancreas centerline points.

        Shape:
            (N, 3)

        Coordinate order:
            (z, y, x)

    subsample : int, default=5
        Downsampling factor for visualization.

        Larger values:
            fewer plotted voxels
            faster rendering

    figsize : tuple of int, default=(14, 12)
        Figure size passed to matplotlib.

    show : bool, default=True
        If True:
            display figure using plt.show()

        If False:
            close figure after creation

    save_path : str or pathlib.Path, optional
        Output image path.

        Example:
            "qc.png"

    Returns
    -------
    matplotlib.figure.Figure
        Generated matplotlib figure object.
    """

    views = {
        "Axial": (90, -90),
        "Coronal": (0, -90),
        "Sagittal": (0, 0),
        "Oblique": (30, 45),
    }

    region_mask = sitk.GetArrayFromImage(volume)
    # foreground voxel coordinates
    coords = np.argwhere(region_mask > 0)

    # corresponding region labels
    labels = region_mask[region_mask > 0]

    # visualization downsampling
    coords = coords[::subsample]
    labels = labels[::subsample]

    # centerline array
    cl = np.asarray(centerline_zyx)

    # create figure
    fig = plt.figure(figsize=figsize)

    for i, (name, (elev, azim)) in enumerate(
        views.items(),
        start=1,
    ):
        ax = fig.add_subplot(
            2,
            2,
            i,
            projection="3d",
        )

        # region voxel visualization
        ax.scatter(
            coords[:, 2],  # x
            coords[:, 1],  # y
            coords[:, 0],  # z
            c=labels,
            s=1,
            alpha=0.15,
        )

        # centerline visualization
        ax.plot(
            cl[:, 2],
            cl[:, 1],
            cl[:, 0],
            c="red",
            linewidth=3,
        )

        # camera view
        ax.view_init(
            elev=elev,
            azim=azim,
        )

        ax.set_title(name)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # medical image convention
        ax.invert_zaxis()

    plt.tight_layout()

    # save figure
    if save_path is not None:
        plt.savefig(
            str(save_path),
            bbox_inches="tight",
        )

    # display or close
    if show:
        plt.show()
    else:
        plt.close(fig)


def figure_overlay_abdominal_distance_metrics(
    volume: sitk.Image,
    volume_sfat: sitk.Image,
    volume_l3: sitk.Image,
    distance_metrics: Dict[str, Any],
    color: str = "tab10",
    alpha: float = 0.3,
    show: bool = True,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Visualize abdominal distance metrics on the L3 axial slice.

    Parameters
    ----------
    volume : sitk.Image
        CT volume.

    volume_sfat : sitk.Image
        Subcutaneous fat mask.

    volume_l3 : sitk.Image
        L3 mask.

    distance_metrics : dict
        Output dictionary from extract_abdominal_distance_metrics().

        Expected keys:
            - slice_index
            - l3_reference_point_zyx
            - LRD_margin
            - APD_margin
            - fat_start_yx
            - fat_end_yx

    show : bool, default=True
        If True, display the figure.

    save_path : str or Path, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    img_arr = sitk.GetArrayFromImage(volume)
    sfat_arr = sitk.GetArrayFromImage(volume_sfat) > 0
    l3_arr = sitk.GetArrayFromImage(volume_l3) > 0

    if not (img_arr.shape == sfat_arr.shape == l3_arr.shape):
        raise ValueError("volume, volume_sfat, and volume_l3 must have the same shape.")

    l3_z, l3_y, l3_x = distance_metrics["l3_reference_point_zyx"]
    lrd_x_min, lrd_x_max = distance_metrics["LRD_margin"]
    apd_y_min, apd_y_max = distance_metrics["APD_margin"]

    p_max_start = distance_metrics.get("fat_max_start_yx")
    p_max_end = distance_metrics.get("fat_max_end_yx")
    p_min_start = distance_metrics.get("fat_min_start_yx")
    p_min_end = distance_metrics.get("fat_min_end_yx")
    p_r_start = distance_metrics.get("fat_right_start_yx")
    p_r_end = distance_metrics.get("fat_right_end_yx")
    p_l_start = distance_metrics.get("fat_left_start_yx")
    p_l_end = distance_metrics.get("fat_left_end_yx")
    p_a_start = distance_metrics.get("fat_anterior_start_yx")
    p_a_end = distance_metrics.get("fat_anterior_end_yx")

    mask = sfat_arr[l3_z] + l3_arr[l3_z] * 2
    cmap = make_cmap_from_base(mask, base_cmap=color, alpha=alpha)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        img_arr[l3_z],
        cmap="gray",
        vmin=-150,
        vmax=300,
    )

    ax.imshow(
        mask,
        cmap=cmap,
        interpolation="nearest",
    )

    # L3 reference point
    ax.scatter(
        l3_x,
        l3_y,
        color="pink",
        s=50,
        label="The anterior spine point of L3",
    )

    # Subcutaneous fat thickness
    if p_max_start is not None and p_max_end is not None:
        ax.plot(
            [p_max_start[1], p_max_end[1]],
            [p_max_start[0], p_max_end[0]],
            color="white",
            linewidth=2,
            label="Subcutaneous Fat Thickness",
        )
        ax.plot(
            [p_min_start[1], p_min_end[1]],
            [p_min_start[0], p_min_end[0]],
            color="white",
            linewidth=2,
        )
        ax.plot(
            [p_a_start[1], p_a_end[1]],
            [p_a_start[0], p_a_end[0]],
            color="white",
            linewidth=4,
        )
        ax.plot(
            [p_l_start[1], p_l_end[1]],
            [p_l_start[0], p_l_end[0]],
            color="white",
            linewidth=4,
        )
        ax.plot(
            [p_r_start[1], p_r_end[1]],
            [p_r_start[0], p_r_end[0]],
            color="white",
            linewidth=4,
        )

    # APD full line
    ax.plot(
        [l3_x, l3_x],
        [apd_y_min, apd_y_max],
        color="red",
        linewidth=2.5,
        label="Anterior-Posterior Diameter",
    )

    # AD / anterior distance to L3 reference
    ax.plot(
        [l3_x, l3_x],
        [apd_y_min, l3_y],
        color="yellow",
        linewidth=1,
        label="Anterior Distance to L3 Reference",
    )

    # LRD
    ax.plot(
        [lrd_x_min, lrd_x_max],
        [l3_y, l3_y],
        color="blue",
        linewidth=2.5,
        label="Left-Right Diameter at L3 Reference",
    )

    ax.legend(loc="lower right")
    ax.axis("off")
    ax.set_title(f"L3 abdominal distance metrics, slice={l3_z}")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def figure_overlay_pancreatic_distance_metrics(
    volume: sitk.Image,
    volume_torso: sitk.Image,
    volume_pancreas: sitk.Image,
    distance_metrics: Dict[str, Any],
    color: str = "tab10",
    alpha: float = 0.3,
    show: bool = True,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Visualize pancreas-to-anterior abdominal wall distance (PAAD).

    Visualization
    -------------
    Left:
        Axial slice at PAAD level.

    Right:
        Coronal slice through PAAD column.

    Parameters
    ----------
    volume : sitk.Image
        CT volume.

    volume_torso : sitk.Image
        Binary torso/body mask.

    volume_pancreas : sitk.Image
        Binary pancreas mask.

    distance_metrics : dict
        Output dictionary from
        extract_pancreatic_distance_metrics().

        Required keys:
            - slice_index
            - pancreas_point_yx
            - skin_point_yx
            - PAAD_mm

    show : bool, default=True
        If True, display figure.

    save_path : str or Path, optional
        Save figure path.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    img_arr = sitk.GetArrayFromImage(volume)

    torso_arr = sitk.GetArrayFromImage(volume_torso) > 0

    pancreas_arr = sitk.GetArrayFromImage(volume_pancreas) > 0

    if not (img_arr.shape == torso_arr.shape == pancreas_arr.shape):
        raise ValueError("volume, volume_torso, and volume_pancreas " "must have the same shape.")

    slice_index = int(distance_metrics["slice_index"])

    pancreas_y, pancreas_x = distance_metrics["pancreas_point_yx"]

    skin_y, skin_x = distance_metrics["skin_point_yx"]

    PAAD_mm = float(distance_metrics.get("PAAD_mm", 0.0))

    # --------------------------------------------------
    # cmap
    # --------------------------------------------------
    cmap = cmap = make_cmap_from_base(pancreas_arr, base_cmap=color, alpha=alpha)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )

    # ==================================================
    # axial view
    # ==================================================
    axes[0].imshow(
        img_arr[slice_index],
        cmap="gray",
        vmin=-150,
        vmax=300,
    )

    axes[0].imshow(
        pancreas_arr[slice_index],
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
    )

    axes[0].plot(
        [pancreas_x, pancreas_x],
        [skin_y, pancreas_y],
        color="red",
        linewidth=2,
        label=f"PAAD = {PAAD_mm:.1f} mm",
    )

    axes[0].set_title(f"Axial view (slice={slice_index})")

    axes[0].axis("off")
    axes[0].legend(loc="lower right")

    # ==================================================
    # coronal view
    # ==================================================
    axes[1].imshow(
        img_arr[:, :, pancreas_x],
        cmap="gray",
        origin="lower",
        aspect="auto",
        vmin=-150,
        vmax=300,
    )

    axes[1].imshow(
        pancreas_arr[:, :, pancreas_x],
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
        origin="lower",
        aspect="auto",
    )

    axes[1].plot(
        [skin_y, pancreas_y],
        [slice_index, slice_index],
        color="red",
        linewidth=2,
    )

    axes[1].set_title(f"Coronal view (x={pancreas_x})")

    axes[1].axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            str(save_path),
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def figure_overlay_pancreatic_craniocaudal_slices(
    volume: sitk.Image,
    volume_pancreas: sitk.Image,
    volume_visceralfat: sitk.Image,
    volume_metrics: Mapping[str, Any] | None = None,
    color: str = "tab10",
    alpha: float = 0.3,
    margin: int = 10,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """
    Visualize pancreas and visceral fat overlays across the pancreas craniocaudal extent.

    Twelve axial slices are sampled evenly from ``z_min - margin`` to
    ``z_max + margin``, where ``z_min`` and ``z_max`` are the first and last
    slices containing pancreas mask voxels. The slices are displayed as a
    4-by-3 subplot grid.

    Parameters
    ----------
    volume : sitk.Image
        CT volume.

    volume_pancreas : sitk.Image
        Binary pancreas mask.

    volume_visceralfat : sitk.Image
        Binary visceral fat mask.

    volume_metrics : Mapping, optional
        Optional metrics dictionary. If it contains ``center_of_mask``, the
        value is used as the displayed pancreas centroid; otherwise the centroid
        is computed from ``volume_pancreas``.

    color : str, default="tab10"
        Matplotlib colormap name used for overlays.

    alpha : float, default=0.3
        Overlay transparency.

    margin : int, default=10
        Number of slices added before and after the pancreas z-range.

    show : bool, default=True
        If True, display the figure.

    save_path : str or Path, optional
        If provided, save the figure to this path.
    """
    img_arr = sitk.GetArrayFromImage(volume)
    pancreas_arr = sitk.GetArrayFromImage(volume_pancreas) > 0
    vfat_arr = sitk.GetArrayFromImage(volume_visceralfat) > 0

    if not (img_arr.shape == pancreas_arr.shape == vfat_arr.shape):
        raise ValueError(
            "volume, volume_pancreas, and volume_visceralfat must have the same shape."
        )
    if not np.any(pancreas_arr):
        raise ValueError("volume_pancreas is empty. Cannot select pancreas slices.")

    if volume_metrics is not None and "center_of_mask" in volume_metrics:
        zc, yc, xc = volume_metrics["center_of_mask"]
    else:
        zc, yc, xc = center_of_mass(pancreas_arr)
        if np.any(np.isnan((zc, yc, xc))):
            raise ValueError("volume_pancreas is empty. Cannot compute pancreas centroid.")

    z_idx = np.where(pancreas_arr.max(axis=(1, 2)))[0]
    z_start = int(np.clip(z_idx[0] - margin, 0, img_arr.shape[0] - 1))
    z_stop = int(np.clip(z_idx[-1] + margin, 0, img_arr.shape[0] - 1))
    slice_indices = np.linspace(z_start, z_stop, num=12)
    slice_indices = np.round(slice_indices).astype(int)

    mask = pancreas_arr + 2 * vfat_arr
    cmap = make_cmap_from_base(mask, base_cmap=color, alpha=alpha)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes_flat = np.asarray(axes).ravel()

    for ax, slice_idx in zip(axes_flat, slice_indices):
        ax.imshow(
            img_arr[slice_idx],
            cmap="gray",
            vmin=-150,
            vmax=300,
        )
        ax.imshow(
            pancreas_arr[slice_idx],
            cmap=cmap,
            interpolation="nearest",
        )
        ax.imshow(
            vfat_arr[slice_idx],
            cmap=cmap,
            interpolation="nearest",
        )
        ax.axhline(int(yc), color="white", linewidth=0.5)
        ax.axvline(int(xc), color="white", linewidth=0.5)
        ax.set_title(f"z={slice_idx}")
        ax.axis("off")

    fig.suptitle(f"Pancreas centroid: z={zc:.1f}, y={yc:.1f}, x={xc:.1f}", y=0.995)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
