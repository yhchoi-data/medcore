from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import measure

from ..utils.sitk_utils import sitk_read_labelfiles, sitk_resampler

PathLike = Union[str, Path]
LabelFiles = Mapping[int, PathLike]

__all__ = [
    "compute_label_volumes",
    "compute_label_volumns",
    "compute_label_areas",
    "extract_abdominal_body_composition_metrics",
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


def extract_abdominal_body_composition_metrics(
    slice_mask: np.ndarray,
    tissue_mask: np.ndarray,
    pixel_spacing: Tuple[float, float, float],
) -> dict[str, float]:
    slice_mask = np.asarray(slice_mask)
    tissue_mask = np.asarray(tissue_mask)

    if slice_mask.ndim != 2 or tissue_mask.ndim != 2:
        raise ValueError("`slice_mask` and `tissue_mask` must be 2D arrays.")
    if slice_mask.shape != tissue_mask.shape:
        raise ValueError("`slice_mask` and `tissue_mask` must have the same shape.")

    labeled = measure.label(slice_mask > 0)
    props = measure.regionprops(labeled)
    if not props:
        raise ValueError("No connected component found in `slice_mask`.")

    largest = max(props, key=lambda prop: prop.area)
    cy, cx = largest.centroid
    cx = int(round(cx))
    cy = int(round(cy))

    row_vals = np.where(slice_mask[cy, :] > 0)[0]
    width_px = int(row_vals.max() - row_vals.min() + 1) if len(row_vals) > 0 else 0
    tad_mm = width_px * pixel_spacing[0]

    col_vals = np.where(slice_mask[:, cx] > 0)[0]
    height_px = int(col_vals.max() - col_vals.min() + 1) if len(col_vals) > 0 else 0
    sad_mm = height_px * pixel_spacing[1]

    binary_mask = (slice_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("No contour found in `slice_mask`.")

    contour = max(contours, key=cv2.contourArea)
    pts = contour[:, 0, :].astype(np.float64)
    pts_mm = np.zeros_like(pts, dtype=np.float64)
    pts_mm[:, 0] = pts[:, 0] * pixel_spacing[0]
    pts_mm[:, 1] = pts[:, 1] * pixel_spacing[1]

    diffs = np.diff(np.vstack([pts_mm, pts_mm[0]]), axis=0)
    perimeter_mm = float(np.sum(np.sqrt((diffs**2).sum(axis=1))))

    pixel_area_mm2 = pixel_spacing[0] * pixel_spacing[1]
    muscle_area_mm2 = float(np.sum(tissue_mask == 1) * pixel_area_mm2)
    fat_area_mm2 = float(np.sum(tissue_mask == 2) * pixel_area_mm2)

    return {
        "MA_cm2": muscle_area_mm2 / 100.0,
        "SFA_cm2": fat_area_mm2 / 100.0,
        "perimeter_cm": perimeter_mm / 10.0,
        "TAD_cm": tad_mm / 10.0,
        "SAD_cm": sad_mm / 10.0,
        "Ratio": float(tad_mm / sad_mm) if sad_mm > 0 else np.nan,
    }
