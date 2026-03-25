from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ..utils.sitk_utils import sitk_read_labelfiles, sitk_resampler

PathLike = Union[str, Path]
LabelFiles = Mapping[int, PathLike]

__all__ = [
    "compute_label_volumes",
    "compute_label_volumns",
    "compute_label_areas",
    "extract_patches_from_image",
]


def _label_names(labelfiles: LabelFiles) -> list[str]:
    names: list[str] = []
    for path in labelfiles.values():
        p = Path(path)
        names.append(p.name.replace(".nii.gz", "").replace(".nii", ""))
    return names


def _to_table(
    labelfiles: LabelFiles, label_ids: np.ndarray, counts: np.ndarray, unit: float
) -> pd.DataFrame:
    names = _label_names(labelfiles)
    table = pd.DataFrame(np.zeros(len(names), dtype=float), index=names, columns=["value"])
    valid = label_ids > 0
    for label_id, count in zip(label_ids[valid], counts[valid]):
        idx = int(label_id) - 1
        if 0 <= idx < len(names):
            table.iloc[idx, 0] = float(count) * unit
    return table.T


def compute_label_volumes(
    labelfiles: LabelFiles,
    transform: sitk.Transform | None = None,
    return_vols: bool = False,
) -> pd.DataFrame | Tuple[sitk.Image, pd.DataFrame]:
    vols = sitk_read_labelfiles(labelfiles)
    if transform is not None:
        vols = sitk_resampler(vols, transform=transform, interpolation="nn")

    sx, sy, sz = vols.GetSpacing()
    vol_unit = (sx * sy * sz) / 1000.0  # mm^3 -> cm^3

    labels = sitk.GetArrayFromImage(vols)
    label_ids, label_counts = np.unique(labels, return_counts=True)
    table = _to_table(labelfiles, label_ids, label_counts, vol_unit)

    return (vols, table) if return_vols else table


def compute_label_volumns(
    labelfiles: LabelFiles,
    transform: sitk.Transform | None = None,
    return_vols: bool = False,
) -> pd.DataFrame | Tuple[sitk.Image, pd.DataFrame]:
    """
    Backward-compatible alias of compute_label_volumes.
    """
    return compute_label_volumes(labelfiles, transform=transform, return_vols=return_vols)


def compute_label_areas(
    labelfiles: LabelFiles,
    slices_index: Union[int, Sequence[int], np.ndarray],
    transform: sitk.Transform | None = None,
    return_vols: bool = False,
) -> pd.DataFrame | Tuple[sitk.Image, pd.DataFrame]:
    vols = sitk_read_labelfiles(labelfiles)
    if transform is not None:
        vols = sitk_resampler(vols, transform=transform, interpolation="nn")

    sx, sy, _ = vols.GetSpacing()
    area_unit = (sx * sy) / 100.0  # mm^2 -> cm^2

    labels = sitk.GetArrayFromImage(vols)
    label_ids, label_counts = np.unique(labels[slices_index], return_counts=True)
    table = _to_table(labelfiles, label_ids, label_counts, area_unit)

    return (vols, table) if return_vols else table


def _extract_patch_safe(
    image: np.ndarray,
    center: np.ndarray,
    patch_size: int,
    middle_size: int,
    delta: int,
) -> np.ndarray:
    x, y, z = [int(v) for v in center]
    half = patch_size // 2

    out = np.zeros((patch_size, middle_size, patch_size), dtype=image.dtype)

    src_x0, src_x1 = x - half, x + half
    src_y0, src_y1 = y - delta, y + (middle_size - delta)
    src_z0, src_z1 = z - half, z + half

    dst_x0 = max(0, -src_x0)
    dst_y0 = max(0, -src_y0)
    dst_z0 = max(0, -src_z0)

    src_x0 = max(0, src_x0)
    src_y0 = max(0, src_y0)
    src_z0 = max(0, src_z0)
    src_x1 = min(image.shape[0], src_x1)
    src_y1 = min(image.shape[1], src_y1)
    src_z1 = min(image.shape[2], src_z1)

    dx = src_x1 - src_x0
    dy = src_y1 - src_y0
    dz = src_z1 - src_z0
    if dx <= 0 or dy <= 0 or dz <= 0:
        return out

    out[dst_x0 : dst_x0 + dx, dst_y0 : dst_y0 + dy, dst_z0 : dst_z0 + dz] = image[
        src_x0:src_x1, src_y0:src_y1, src_z0:src_z1
    ]
    return out


def extract_patches_from_image(
    points: np.ndarray,
    volume: sitk.Image,
    patch_size: int = 50,
    middle_size: int = 50,
    delta: int = 25,
) -> np.ndarray:
    image = sitk.GetArrayFromImage(volume)
    points = np.asarray(points, dtype=int)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("`points` must have shape (N, 3).")

    n = len(points)
    grid_size = int(np.sqrt(n))
    if grid_size * grid_size != n:
        raise ValueError("`points` length must be a perfect square (e.g., 25 for 5x5 grid).")

    patches = np.zeros(
        (grid_size * patch_size, middle_size, grid_size * patch_size), dtype=image.dtype
    )
    for idx, point in enumerate(points):
        sub_patch = _extract_patch_safe(
            image, np.clip(point[:3], 0, None), patch_size, middle_size, delta
        )
        is_idx = grid_size - 1 - (idx // grid_size)
        rl_idx = idx % grid_size
        patches[
            is_idx * patch_size : (is_idx + 1) * patch_size,
            :,
            rl_idx * patch_size : (rl_idx + 1) * patch_size,
        ] = sub_patch
    return patches
