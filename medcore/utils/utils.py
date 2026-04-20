from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


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
