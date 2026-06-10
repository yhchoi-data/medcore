from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, center_of_mass
from skimage import measure
from skimage.morphology import remove_small_holes, remove_small_objects

from ..utils.sitk_utils import sitk_read_labelfiles, sitk_resampler

PathLike = Union[str, Path]
LabelFiles = Mapping[int, PathLike]

__all__ = [
    "ContourFatThicknessMeasurer",
    "compute_label_volumes",
    "compute_label_volumns",
    "compute_label_areas",
    "extract_optimal_transverse_process_slice",
    "extract_abdominal_distance_metrics",
    "extract_abdominal_body_composition_metrics",
    "extract_pancreatic_distance_metrics",
    "extract_peripancreatic_fat_volume",
    "extract_craniocaudal_fat_volume",
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


def extract_optimal_transverse_process_slice(
    mask: sitk.Image,
    topk: int = 5,
    alpha: float = 0.75,
    use_transverse_process=True,
) -> int:
    """
    score = alpha * norm_width + (1-alpha) * norm_area
    alpha가 클수록 width(=transverse process 가중치) 중요.
    """

    m = sitk.GetArrayFromImage(mask).astype(np.uint8)
    if m.max() == 0:
        return 0

    if use_transverse_process == False:
        coord_is = np.where(m.max(2) > 0)[0]
        z_idx = int(np.median(coord_is))
        return z_idx

    z_list, widths, areas = [], [], []
    for z in range(m.shape[0]):
        sl = m[z]
        if sl.max() == 0:
            continue

        coords = np.where(sl > 0)
        x_coords = coords[1]
        widths.append(float(x_coords.max() - x_coords.min()))
        areas.append(float(coords[0].size))
        z_list.append(z)

    if not z_list:
        return 0

    widths = np.array(widths)
    areas = np.array(areas)

    # normalize (0~1)
    w = (widths - widths.min()) / (np.ptp(widths) + 1e-8)
    a = (areas - areas.min()) / (np.ptp(areas) + 1e-8)
    score = alpha * w + (1 - alpha) * a

    topk = min(topk, len(score))
    top_idx = np.argsort(score)[-topk:]
    z_top = [z_list[i] for i in top_idx]
    return int(np.median(z_top))


class ContourFatThicknessMeasurer:
    def __init__(
        self,
        fat_mask: np.ndarray,
        contour_yx: np.ndarray,
        spacing_yx: tuple[float, float] = (1.0, 1.0),
        body_centroid_yx: tuple[float, float] | np.ndarray | None = None,
        window: int | None = None,
        max_distance_mm: float = 100.0,
        step_mm: float = 1.0,
        max_normal_centroid_angle_deg: float | None = 60.0,
    ) -> None:
        self.fat_mask = np.asarray(fat_mask) > 0
        self.contour_yx = np.asarray(contour_yx, dtype=float)
        self.spacing_yx = np.asarray(spacing_yx, dtype=float)
        self.body_centroid_yx = (
            None if body_centroid_yx is None else np.asarray(body_centroid_yx, dtype=float)
        )
        self.max_distance_mm = float(max_distance_mm)
        self.step_mm = float(step_mm)
        self.max_normal_centroid_angle_deg = (
            None if max_normal_centroid_angle_deg is None else float(max_normal_centroid_angle_deg)
        )

        if self.fat_mask.ndim != 2:
            raise ValueError("`fat_mask` must be a 2D array.")
        if self.contour_yx.ndim != 2 or self.contour_yx.shape[1] != 2:
            raise ValueError("`contour_yx` must have shape (N, 2).")
        if self.spacing_yx.shape != (2,) or np.any(self.spacing_yx <= 0):
            raise ValueError("`spacing_yx` must contain two positive values.")
        if self.body_centroid_yx is not None and self.body_centroid_yx.shape != (2,):
            raise ValueError("`body_centroid_yx` must contain two values.")

        # Drop duplicated closing point if present.
        if len(self.contour_yx) > 1 and np.allclose(self.contour_yx[0], self.contour_yx[-1]):
            self.contour_yx = self.contour_yx[:-1]
        if len(self.contour_yx) < 3:
            raise ValueError("`contour_yx` must contain at least 3 points.")

        self.window = max(1, int(30.0 / self.spacing_yx[0])) if window is None else int(window)
        if self.window < 1:
            raise ValueError("`window` must be >= 1.")
        if self.max_normal_centroid_angle_deg is not None and not (
            0 < self.max_normal_centroid_angle_deg <= 180
        ):
            raise ValueError("`max_normal_centroid_angle_deg` must be in (0, 180] or None.")
        if self.max_distance_mm <= 0 or self.step_mm <= 0:
            raise ValueError("`max_distance_mm` and `step_mm` must be positive.")

    def estimate_inward_normal(
        self,
        start_point_yx: tuple[float, float] | np.ndarray,
    ) -> np.ndarray | None:
        contour = self.contour_yx
        p = np.asarray(start_point_yx, dtype=float)
        if p.shape != (2,):
            raise ValueError("`start_point_yx` must contain two values.")

        idx = int(np.argmin(np.linalg.norm(contour - p, axis=1)))
        n = len(contour)

        p0 = contour[(idx - self.window) % n]
        p1 = contour[(idx + self.window) % n]

        tangent = p1 - p0
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm == 0:
            raise ValueError("Cannot estimate tangent at this contour point.")

        tangent = tangent / tangent_norm

        normal1 = np.array([-tangent[1], tangent[0]])
        normal2 = np.array([tangent[1], -tangent[0]])

        if self.body_centroid_yx is None:
            body_centroid = contour.mean(axis=0)
        else:
            body_centroid = self.body_centroid_yx

        to_inside = body_centroid - p
        normal = normal1 if np.dot(normal1, to_inside) > np.dot(normal2, to_inside) else normal2
        normal = normal / np.linalg.norm(normal)

        centroid_norm = np.linalg.norm(to_inside)
        if centroid_norm == 0:
            return normal

        centroid_direction = to_inside / centroid_norm
        cos_angle = float(np.clip(np.dot(normal, centroid_direction), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if (
            self.max_normal_centroid_angle_deg is not None
            and angle_deg > self.max_normal_centroid_angle_deg
        ):
            return None

        blended_normal = 0.8 * normal + 0.2 * centroid_direction
        blended_norm = np.linalg.norm(blended_normal)
        if blended_norm == 0:
            return normal

        return blended_normal / blended_norm

    def measure_along_direction(
        self,
        start_point_yx: tuple[float, float] | np.ndarray,
        direction_yx: np.ndarray,
    ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        mask = self.fat_mask
        h, w = mask.shape

        p0 = np.asarray(start_point_yx, dtype=float)
        direction = np.asarray(direction_yx, dtype=float)
        if p0.shape != (2,) or direction.shape != (2,):
            raise ValueError("`start_point_yx` and `direction_yx` must contain two values.")

        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("`direction_yx` must be non-zero.")
        direction = direction / norm

        distances_mm = np.arange(0, self.max_distance_mm + self.step_mm, self.step_mm)

        dy = distances_mm * direction[0] / self.spacing_yx[0]
        dx = distances_mm * direction[1] / self.spacing_yx[1]

        ys = p0[0] + dy
        xs = p0[1] + dx

        yi = np.round(ys).astype(int)
        xi = np.round(xs).astype(int)

        valid = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)

        ys = ys[valid]
        xs = xs[valid]
        yi = yi[valid]
        xi = xi[valid]
        distances_mm = distances_mm[valid]

        hit = mask[yi, xi]

        if not np.any(hit):
            return 0.0, None, None

        hit_idx = np.where(hit)[0]
        start_idx = int(hit_idx[0])

        end_idx = start_idx
        while end_idx + 1 < len(hit) and hit[end_idx + 1]:
            end_idx += 1

        thickness_mm = float(distances_mm[end_idx] - distances_mm[start_idx])
        p_start = np.array([ys[start_idx], xs[start_idx]])
        p_end = np.array([ys[end_idx], xs[end_idx]])

        return thickness_mm, p_start, p_end

    def measure_from_point(
        self,
        start_point_yx: tuple[float, float] | np.ndarray,
    ) -> tuple[float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        normal_yx = self.estimate_inward_normal(start_point_yx)
        if normal_yx is None:
            return 0.0, None, None, None
        thickness_mm, p_start, p_end = self.measure_along_direction(start_point_yx, normal_yx)
        return thickness_mm, p_start, p_end, normal_yx


def _polygon_area_yx(contour_yx: np.ndarray) -> float:
    y = contour_yx[:, 0]
    x = contour_yx[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def extract_abdominal_distance_metrics(
    volume: sitk.Image,
    volume_torso: sitk.Image,
    volume_sfat: sitk.Image,
    volume_l3: sitk.Image,
    l3_index: int,
    hu_range: tuple[float, float] = (0, 1000),
    min_l3_component_size: int = 100,
    min_sfat_hole_size: int = 100,
    max_normal_centroid_angle_deg: float | None = 45.0,
) -> dict[str, Any]:
    """
    Extract abdominal distance metrics at the L3 slice.

    The axial array coordinate convention follows ``sitk.GetArrayFromImage``:
    ``(z, y, x)`` for volumes and ``(y, x)`` for slices. Anterior direction is
    assumed to be decreasing row index, matching LPS-oriented axial images.
    """
    img_arr = sitk.GetArrayFromImage(volume)
    torso_arr = sitk.GetArrayFromImage(volume_torso) > 0
    sfat_arr = sitk.GetArrayFromImage(volume_sfat) > 0
    l3_arr = sitk.GetArrayFromImage(volume_l3) > 0

    if not (img_arr.shape == torso_arr.shape == sfat_arr.shape == l3_arr.shape):
        raise ValueError(
            "volume, volume_torso, volume_sfat, and volume_l3 must have the same array shape."
        )

    if not (0 <= l3_index < img_arr.shape[0]):
        raise ValueError("`l3_index` is out of range.")

    hu_min, hu_max = hu_range
    if hu_min > hu_max:
        raise ValueError("`hu_range` must be ordered as (min, max).")

    spacing_x, spacing_y, _ = volume.GetSpacing()

    l3_slice = (img_arr[l3_index] >= hu_min) & (img_arr[l3_index] <= hu_max) & l3_arr[l3_index]
    l3_slice = remove_small_objects(
        l3_slice.astype(bool),
        min_size=min_l3_component_size,
        connectivity=2,
    )

    if not np.any(l3_slice):
        raise ValueError("No valid L3 mask found after cleanup.")

    torso_slice = binary_dilation(torso_arr[l3_index], structure=np.ones((3, 3), dtype=bool))

    sfat_slice = remove_small_objects(
        sfat_arr[l3_index].astype(bool),
        min_size=min_sfat_hole_size,
        connectivity=2,
    )
    sfat_slice = remove_small_holes(
        sfat_slice.astype(bool),
        area_threshold=min_sfat_hole_size,
        connectivity=2,
    )
    sfat_slice = cv2.morphologyEx(
        sfat_slice.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)
    sfat_slice[0, :] = 0  # 위
    sfat_slice[-1:, :] = 0  # 아래
    sfat_slice[:, 0] = 0  # 왼쪽
    sfat_slice[:, -1] = 0  # 오른쪽

    l3_rows, l3_cols = np.where(l3_slice)
    l3_ref_row = int(l3_rows.min())
    anterior_row_indices = np.where(l3_rows == l3_ref_row)[0]
    l3_ref_col = int(round(l3_cols[anterior_row_indices].mean()))
    l3_ref_col = int(np.mean(l3_cols))
    l3_reference_point_zyx = (int(l3_index), l3_ref_row, l3_ref_col)

    lrd_cols = np.where(torso_slice[l3_ref_row, :] > 0)[0]
    if len(lrd_cols) == 0:
        raise ValueError("No torso pixels found for LRD measurement.")
    LRD_mm = float((lrd_cols.max() - lrd_cols.min()) * spacing_x)

    apd_rows = np.where(torso_slice[:, l3_ref_col] > 0)[0]
    if len(apd_rows) == 0:
        raise ValueError("No torso pixels found for APD measurement.")
    APD_mm = float((apd_rows.max() - apd_rows.min()) * spacing_y)
    AP_mm = float((l3_ref_row - apd_rows.min()) * spacing_y)

    def _zero_sft_metrics(n_anterior_contour_points: int = 0) -> dict[str, Any]:
        return {
            "Max_SFT(cm)": 0.0,
            "Min_SFT(cm)": 0.0,
            "Left_SFT(cm)": 0.0,
            "Right_SFT(cm)": 0.0,
            "Anterior_SFT(cm)": 0.0,
            "Mean_SFT(cm)": 0.0,
            "Median_SFT(cm)": 0.0,
            "LRD(cm)": LRD_mm / 10.0,
            "APD(cm)": APD_mm / 10.0,
            "AP(cm)": AP_mm / 10.0,
            "SFT_LR_margin": None,
            "LRD_margin": (int(lrd_cols.min()), int(lrd_cols.max())),
            "APD_margin": (int(apd_rows.min()), int(apd_rows.max())),
            "l3_reference_point_zyx": l3_reference_point_zyx,
            "max_sft_point_yx": None,
            "left_sft_point_yx": None,
            "right_sft_point_yx": None,
            "anterior_sft_point_yx": None,
            "fat_max_start_yx": None,
            "fat_max_end_yx": None,
            "fat_min_start_yx": None,
            "fat_min_end_yx": None,
            "fat_left_start_yx": None,
            "fat_left_end_yx": None,
            "fat_right_start_yx": None,
            "fat_right_end_yx": None,
            "fat_anterior_start_yx": None,
            "fat_anterior_end_yx": None,
            "normal_yx": None,
            "n_anterior_contour_points": int(n_anterior_contour_points),
            "n_positive_sft_points": 0,
        }

    contours = measure.find_contours(sfat_slice.astype(float), level=0.5)
    if len(contours) == 0:
        return _zero_sft_metrics()

    # contour = max(contours, key=_polygon_area_yx)
    contour = np.concatenate(contours)
    contour = np.unique(contour, axis=0)
    anterior_contour = contour[contour[:, 0] < l3_ref_row]
    if len(anterior_contour) == 0:
        return _zero_sft_metrics()

    contour_cols = np.round(anterior_contour[:, 1]).astype(int)
    new_contour = []
    for col in np.unique(contour_cols):
        point_set = anterior_contour[contour_cols == col]
        new_contour.append(point_set[np.argmin(point_set[:, 0])])
    anterior_contour_filtered = np.asarray(new_contour, dtype=float)

    measurer = ContourFatThicknessMeasurer(
        fat_mask=sfat_slice,
        contour_yx=contour,
        spacing_yx=(spacing_y, spacing_x),
        max_normal_centroid_angle_deg=max_normal_centroid_angle_deg,
    )

    thickness_values = np.asarray(
        [measurer.measure_from_point(point_yx)[0] for point_yx in anterior_contour_filtered],
        dtype=float,
    )
    positive_thickness = thickness_values[thickness_values > 0]

    max_SFT_mm = min_SFT_mm = left_SFT_mm = right_SFT_mm = anterior_SFT_mm = 0.0
    max_sft_point_yx = left_sft_point_yx = right_sft_point_yx = anterior_sft_point_yx = None
    fat_max_start_yx = fat_max_end_yx = fat_min_start_yx = fat_min_end_yx = None
    fat_left_start_yx = fat_left_end_yx = fat_right_start_yx = fat_right_end_yx = None
    fat_anterior_start_yx = fat_anterior_end_yx = normal_yx = None
    sft_lr_margin = None

    if len(positive_thickness) > 0:
        max_idx = int(np.argmax(thickness_values))
        max_sft_point_yx = anterior_contour_filtered[max_idx]
        max_SFT_mm, fat_max_start_yx, fat_max_end_yx, normal_yx = measurer.measure_from_point(
            max_sft_point_yx
        )

        positive_idx = np.where(thickness_values > 0)[0]
        min_idx = int(positive_idx[np.argmin(thickness_values[positive_idx])])
        min_sft_point_yx = anterior_contour_filtered[min_idx]
        min_SFT_mm, fat_min_start_yx, fat_min_end_yx, _ = measurer.measure_from_point(
            min_sft_point_yx
        )

    target_lr_row = float(l3_ref_row - 1)
    row_delta = np.abs(anterior_contour[:, 0] - target_lr_row)
    lr_idx = np.where(np.isclose(row_delta, row_delta.min()))[0]
    left_loc = int(np.argmin(anterior_contour[lr_idx][:, 1]))
    right_loc = int(np.argmax(anterior_contour[lr_idx][:, 1]))

    left_idx = int(lr_idx[left_loc])
    right_idx = int(lr_idx[right_loc])
    left_sft_point_yx = anterior_contour[left_idx]
    right_sft_point_yx = anterior_contour[right_idx]
    sft_lr_margin = (left_idx, right_idx)

    left_SFT_mm, fat_left_start_yx, fat_left_end_yx = measurer.measure_along_direction(
        left_sft_point_yx,
        np.array([0.0, 1.0]),
    )
    right_SFT_mm, fat_right_start_yx, fat_right_end_yx = measurer.measure_along_direction(
        right_sft_point_yx,
        np.array([0.0, -1.0]),
    )

    col_delta = np.abs(anterior_contour[:, 1] - l3_ref_col)
    anterior_candidates = np.where(np.isclose(col_delta, col_delta.min()))[0]
    anterior_idx = int(anterior_candidates[np.argmin(anterior_contour[anterior_candidates, 0])])
    anterior_sft_point_yx = anterior_contour[anterior_idx]
    anterior_SFT_mm, fat_anterior_start_yx, fat_anterior_end_yx = measurer.measure_along_direction(
        anterior_sft_point_yx,
        np.array([1.0, 0.0]),
    )

    mean_SFT_mm = float(positive_thickness.mean()) if len(positive_thickness) > 0 else 0.0
    median_SFT_mm = float(np.median(positive_thickness)) if len(positive_thickness) > 0 else 0.0

    return {
        "Max_SFT(cm)": float(max_SFT_mm / 10.0),
        "Min_SFT(cm)": float(min_SFT_mm / 10.0),
        "Left_SFT(cm)": float(left_SFT_mm / 10.0),
        "Right_SFT(cm)": float(right_SFT_mm / 10.0),
        "Anterior_SFT(cm)": float(anterior_SFT_mm / 10.0),
        "Mean_SFT(cm)": mean_SFT_mm / 10.0,
        "Median_SFT(cm)": median_SFT_mm / 10.0,
        "LRD(cm)": LRD_mm / 10.0,
        "APD(cm)": APD_mm / 10.0,
        "AP(cm)": AP_mm / 10.0,
        "SFT_LR_margin": sft_lr_margin,
        "LRD_margin": (int(lrd_cols.min()), int(lrd_cols.max())),
        "APD_margin": (int(apd_rows.min()), int(apd_rows.max())),
        "l3_reference_point_zyx": l3_reference_point_zyx,
        "max_sft_point_yx": max_sft_point_yx,
        "left_sft_point_yx": left_sft_point_yx,
        "right_sft_point_yx": right_sft_point_yx,
        "anterior_sft_point_yx": anterior_sft_point_yx,
        "fat_max_start_yx": fat_max_start_yx,
        "fat_max_end_yx": fat_max_end_yx,
        "fat_min_start_yx": fat_min_start_yx,
        "fat_min_end_yx": fat_min_end_yx,
        "fat_left_start_yx": fat_left_start_yx,
        "fat_left_end_yx": fat_left_end_yx,
        "fat_right_start_yx": fat_right_start_yx,
        "fat_right_end_yx": fat_right_end_yx,
        "fat_anterior_start_yx": fat_anterior_start_yx,
        "fat_anterior_end_yx": fat_anterior_end_yx,
        "normal_yx": normal_yx,
        "n_anterior_contour_points": int(len(anterior_contour)),
        "n_positive_sft_points": int(len(positive_thickness)),
    }


def extract_pancreatic_distance_metrics(
    volume_torso: sitk.Image,
    volume_pancreas: sitk.Image,
) -> dict[str, Any]:
    """
    Extract pancreas-to-anterior abdominal wall distance (PAAD).

    The axial array coordinate convention follows ``sitk.GetArrayFromImage``:
    ``(z, y, x)`` for volumes and ``(y, x)`` for slices. Anterior direction is
    assumed to be decreasing row index, matching LPS-oriented axial images.
    """
    torso_arr = sitk.GetArrayFromImage(volume_torso) > 0
    pancreas_arr = sitk.GetArrayFromImage(volume_pancreas) > 0

    if torso_arr.shape != pancreas_arr.shape:
        raise ValueError("volume_torso and volume_pancreas must have the same shape.")

    if not np.any(pancreas_arr):
        raise ValueError("Pancreas mask is empty.")

    _, spacing_y, _ = volume_torso.GetSpacing()

    z_idx, y_idx = np.where(pancreas_arr.max(axis=2) > 0)
    if len(z_idx) == 0:
        raise ValueError("No valid pancreas voxels found.")

    slice_index = int(z_idx[np.argmin(y_idx)])
    pancreas_slice = pancreas_arr[slice_index]
    torso_slice = binary_dilation(torso_arr[slice_index], structure=np.ones((3, 3), dtype=bool))

    pancreas_rows, pancreas_cols = np.where(pancreas_slice)
    anterior_row = int(pancreas_rows.min())
    anterior_indices = np.where(pancreas_rows == anterior_row)[0]
    anterior_col = int(round(pancreas_cols[anterior_indices].mean()))

    pancreas_point_yx = (anterior_row, anterior_col)

    torso_rows = np.where(torso_slice[:, anterior_col])[0]
    if len(torso_rows) == 0:
        raise ValueError("No torso pixels found along pancreas column.")

    skin_row = int(torso_rows.min())
    skin_point_yx = (skin_row, anterior_col)
    PAAD_mm = float((anterior_row - skin_row) * spacing_y)

    return {
        "PAAD(cm)": PAAD_mm / 10.0,
        "slice_index": slice_index,
        "pancreas_point_yx": pancreas_point_yx,
        "skin_point_yx": skin_point_yx,
    }


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
        "MA(cm2)": muscle_area_mm2 / 100.0,
        "SFA(cm2)": fat_area_mm2 / 100.0,
        "perimeter(cm)": perimeter_mm / 10.0,
        "TAD(cm)": tad_mm / 10.0,
        "SAD(cm)": sad_mm / 10.0,
        "Ratio": float(tad_mm / sad_mm) if sad_mm > 0 else np.nan,
    }


def extract_peripancreatic_fat_volume(
    vol: sitk.Image,
    vol_mask: sitk.Image,
    vol_shell: sitk.Image,
    vol_shell_region: sitk.Image,
    hu_range: tuple[float, float] = (-190, -30),
    center_mask: tuple[float, float, float] | np.ndarray | None = None,
) -> dict[str, int | float]:
    """
    Extract peripancreatic fat voxel counts and volumes from shell masks.

    ``center_mask`` is retained for compatibility and is ignored.
    """
    hu_min, hu_max = hu_range
    if hu_min > hu_max:
        raise ValueError("`hu_range` must be ordered as (min, max).")

    img = sitk.GetArrayFromImage(vol)
    mask = sitk.GetArrayFromImage(vol_mask) > 0
    shell = sitk.GetArrayFromImage(vol_shell) > 0
    shell_region = sitk.GetArrayFromImage(vol_shell_region)

    if img.shape != mask.shape or img.shape != shell.shape or img.shape != shell_region.shape:
        raise ValueError("vol, vol_mask, vol_shell, and vol_shell_region must have the same shape.")

    voxel_volume_cm3 = float(np.prod(vol.GetSpacing()) / 1000.0)
    fat_hu_mask = (img >= hu_min) & (img <= hu_max)

    total_fat_mask = shell & fat_hu_mask
    total_count = int(np.count_nonzero(shell))
    total_fat_count = int(np.count_nonzero(total_fat_mask))

    result: dict[str, int | float] = {
        "total_shell_voxel_count": total_count,
        "total_shell_volume(cm3)": total_count * voxel_volume_cm3,
        "total_shell_fat_voxel_count": total_fat_count,
        "total_shell_fat_volume(cm3)": total_fat_count * voxel_volume_cm3,
    }

    region_labels = np.unique(shell_region)
    region_labels = region_labels[region_labels > 0]
    for label in region_labels:
        region_fat_mask = (shell_region == label) & shell & fat_hu_mask
        count = int(np.count_nonzero(region_fat_mask))
        key = f"region_{int(label)}"
        result[f"{key}_fat_voxel_count"] = count
        result[f"{key}_fat_volume(cm3)"] = count * voxel_volume_cm3

    return result


def extract_craniocaudal_fat_volume(
    vol: sitk.Image,
    vol_mask: sitk.Image,
    vol_shell: sitk.Image,
    hu_range: tuple[float, float] = (-190, -30),
    margin: int = 10,
    center_mask: tuple[float, float, float] | np.ndarray | None = None,
) -> dict[str, int | float | tuple[float, float, float]]:
    """
    Extract peripancreatic fat voxel counts and volumes from a shell mask.

    Fat voxels are defined by ``hu_range`` inside ``vol_shell`` and are limited
    to the craniocaudal extent of ``vol_mask`` with ``margin`` slices added at
    both ends. The function returns total fat volume, anterior-superior fat
    volume, and 8 octant-wise fat volumes split by ``center_mask``. If
    ``center_mask`` is ``None``, the split center is computed from ``vol_mask``
    with ``center_of_mass``.

    ``vol`` must be oriented as either LPS or RAS. The anatomical half-spaces
    are inferred from ``vol.GetDirection()``:
    - LPS: anterior is decreasing array-y, left is increasing array-x.
    - RAS: anterior is increasing array-y, left is decreasing array-x.
    """
    hu_min, hu_max = hu_range
    if hu_min > hu_max:
        raise ValueError("`hu_range` must be ordered as (min, max).")

    orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(vol.GetDirection())
    if orientation not in {"LPS", "RAS"}:
        raise ValueError(f"`vol` orientation must be LPS or RAS. Got: {orientation}")

    img = sitk.GetArrayFromImage(vol)
    mask = sitk.GetArrayFromImage(vol_mask) > 0
    shell = sitk.GetArrayFromImage(vol_shell) > 0

    if img.shape != mask.shape or img.shape != shell.shape:
        raise ValueError("vol, vol_mask, vol_shell must have the same shape.")

    voxel_volume_cm3 = float(np.prod(vol.GetSpacing()) / 1000.0)
    shell_margin = (img >= hu_min) & (img <= hu_max) & shell
    z_idx = np.where(mask.max(2))[0]
    if len(z_idx) == 0:
        raise ValueError("vol_mask is empty. Cannot select craniocaudal extent.")

    shell_margin[: z_idx[0] - margin, ...] = 0
    shell_margin[z_idx[-1] + margin :, ...] = 0

    total_fat_count = int(np.count_nonzero(shell_margin))

    if center_mask is None:
        center_mask = center_of_mass(mask)
        if np.any(np.isnan(center_mask)):
            raise ValueError("vol_mask is empty. Cannot compute pancreas centroid.")
    else:
        center_mask = np.asarray(center_mask, dtype=float)
        if center_mask.shape != (3,) or not np.all(np.isfinite(center_mask)):
            raise ValueError("`center_mask` must contain three finite values.")

    zc, yc, xc = center_mask
    z_grid = np.arange(mask.shape[0])[:, None, None]
    y_grid = np.arange(mask.shape[1])[None, :, None]
    x_grid = np.arange(mask.shape[2])[None, None, :]
    superior_halfspace = z_grid >= zc
    inferior_halfspace = z_grid < zc

    if orientation == "LPS":
        anterior_halfspace = y_grid <= yc
        posterior_halfspace = y_grid > yc
        left_halfspace = x_grid >= xc
        right_halfspace = x_grid < xc
    else:
        anterior_halfspace = y_grid >= yc
        posterior_halfspace = y_grid < yc
        left_halfspace = x_grid < xc
        right_halfspace = x_grid >= xc

    octant_masks = {
        "superior_anterior_left": superior_halfspace & anterior_halfspace & left_halfspace,
        "superior_anterior_right": superior_halfspace & anterior_halfspace & right_halfspace,
        "superior_posterior_left": superior_halfspace & posterior_halfspace & left_halfspace,
        "superior_posterior_right": superior_halfspace & posterior_halfspace & right_halfspace,
        "inferior_anterior_left": inferior_halfspace & anterior_halfspace & left_halfspace,
        "inferior_anterior_right": inferior_halfspace & anterior_halfspace & right_halfspace,
        "inferior_posterior_left": inferior_halfspace & posterior_halfspace & left_halfspace,
        "inferior_posterior_right": inferior_halfspace & posterior_halfspace & right_halfspace,
    }

    result: dict[str, int | float | tuple[float, float, float]] = {
        "total_fat_voxel_count": total_fat_count,
        "total_fat_volume(cm3)": total_fat_count * voxel_volume_cm3,
    }

    for key, octant_mask in octant_masks.items():
        count = int(np.count_nonzero(shell_margin & octant_mask))
        result[f"{key}_fat_voxel_count"] = count
        result[f"{key}_fat_volume(cm3)"] = count * voxel_volume_cm3

    return result
