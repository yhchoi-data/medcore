from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

from ..utils.sitk_utils import sitk_copy_metainfo, sitk_get_array, sitk_resampler


@dataclass(frozen=True)
class ROI:
    axial: Tuple[int, int]
    coronal: Tuple[int, int]
    sagittal: Tuple[int, int]


def get_median_slice_index(mask, use_transverse_process=True):
    """
    mask: (Z, Y, X) binary mask
    return: z-index of median slice
    """

    image = sitk.GetArrayFromImage(mask)

    coord_is = np.where(image.max(2) > 0)[0]
    if len(coord_is) > 0:
        if use_transverse_process == True:
            widths = []

            for z in range(image.shape[0]):
                coords = np.where(image[z] > 0)
                if coords[0].size == 0:
                    widths.append(0)
                    continue

                x_coords = coords[1]  # assuming (Y, X)
                width = x_coords.max() - x_coords.min()
                widths.append(width)

            z_idx = int(np.median(np.argsort(widths)[-5:]))

        else:
            z_idx = int(np.median(coord_is))
    else:
        z_idx = 0

    return z_idx


def get_longest_segment(arr: Sequence[int]) -> Tuple[Optional[int], Optional[int], int]:
    """
    Find the longest contiguous segment where values are >= 1.
    Returns (start, end, length) with inclusive end index.
    """
    arr = np.asarray(arr, dtype=int)
    is_one = arr >= 1

    diff = np.diff(np.r_[0, is_one, 0].astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    lengths = ends - starts + 1

    if lengths.size == 0:
        return None, None, 0

    idx = int(np.argmax(lengths))
    return int(starts[idx]), int(ends[idx]), int(lengths[idx])


def get_coronal_plane_degree(volume: sitk.Image, margin: int = 150) -> float:
    """
    Estimate coronal plane angle (degrees) from body trunk principal axis.
    """
    # 0. resample ISO-voxel
    volume_iso = sitk_resampler(volume, new_spacing=(1.0, 1.0, 1.0))
    image = sitk_get_array(volume_iso, normalize=True)

    # 1. MIP 영상 추출
    mip = image.max(axis=2)
    mip[mip < 1] = 0
    mip_is = image.max(axis=0)
    mip_is[mip_is < 1] = 0

    # 2. 갈비뼈 영역 탐색
    fg = np.column_stack(np.where(mip_is < 1))
    com = fg.mean(axis=0) if fg.size > 0 else np.array([np.nan, np.nan], dtype=float)
    if np.isnan(com).any():
        raise ValueError("Failed to compute center of mass from input volume.")

    col = int(np.clip(com[1], 0, mip_is.shape[1] - 1))
    x0 = max(0, col - 25)
    x1 = min(mip_is.shape[1], col + 25)
    if x1 <= x0:
        raise ValueError("Invalid rib-search window while estimating coronal angle.")

    start, _, length = get_longest_segment(mip_is[:, x0:x1].sum(axis=1))
    if start is None or length == 0:
        raise ValueError("Could not detect trunk segment from MIP image.")
    mip[:, :start] = 0

    # 3. 몸통에 해당하는 픽셀 좌표 추출
    if mip.shape[0] - margin > 0:
        coords = np.column_stack(np.where(mip[margin:, :] > 0))  # (y, x)
    else:
        coords = np.column_stack(np.where(mip > 0))  # (y, x)
    if coords.shape[0] < 2:
        raise ValueError("Not enough foreground points for PCA angle estimation.")

    # 4. PCA로 주축 방향 계산
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 5. 주요 축 방향의 벡터로 각도 계산
    main_axis = eigvecs[:, int(np.argmax(eigvals))]
    angle_deg = float(np.rad2deg(np.arctan2(main_axis[0], main_axis[1])))
    return angle_deg + 90.0


class UmbilicusPredictor:
    """
    Umbilicus predictor using two heuristics:
      1) Contour-based corner/indentation scoring
      2) Intensity-based 3D kernel summation

    Parameters
    ----------
    window_min, window_max : int
        HU window for preprocessing.
    contour_threshold : int
        Threshold used in contour extraction (binary threshold).
    contour_min_score : float
        If contour score < this threshold, fallback to intensity method.
    intensity_kernel_mm : float
        Kernel size in mm for intensity method.
    """

    def __init__(
        self,
        window_min: int = -300,
        window_max: int = 100,
        contour_threshold: int = 32,
        contour_min_score: float = 7.0,
        intensity_kernel_mm: float = 5.0,
    ):
        self.window_min = window_min
        self.window_max = window_max
        self.contour_threshold = contour_threshold
        self.contour_min_score = contour_min_score
        self.intensity_kernel_mm = intensity_kernel_mm

    # ---------------------------
    # Public API
    # ---------------------------
    def predict(
        self,
        volume: sitk.Image,
        roi: Optional[ROI] = None,
        copy: bool = True,
    ) -> List[int]:
        """
        Predict umbilicus position [x, y, z] in voxel index coordinates:
            x = sagittal index
            y = coronal index
            z = axial index

        Notes
        -----
        Your original contour method returns [x, y, z] where x,y are in-slice coords,
        z is slice index. We keep that convention.
        """
        ct = sitk.GetArrayFromImage(volume)
        spacing = np.asarray(volume.GetSpacing(), dtype=float)
        if spacing.shape[0] < 3:
            raise ValueError(
                "spacing must have 3 elements: [sy, sx, sz] or similar order used in your code"
            )

        ct_proc = ct.copy() if copy else ct
        ct_proc = self._ct_windowing(ct_proc, self.window_min, self.window_max)

        roi_eff = roi if roi is not None else self._default_roi(ct_proc.shape, spacing)

        predicted, score = self._method_contour(ct_proc, spacing, roi_eff)
        if score < self.contour_min_score:
            predicted, score = self._method_intensity(
                ct_proc, spacing, kernel_size_mm=self.intensity_kernel_mm, roi=roi_eff
            )

        points = pd.DataFrame(
            [int(predicted[2]), int(predicted[1]), int(predicted[0])], index=["SI", "AP", "LR"]
        ).T
        points["SCORE"] = score
        return points

    # ---------------------------
    # Helpers / preprocessing
    # ---------------------------
    @staticmethod
    def _mm2pix(mm: float, spacing: float) -> int:
        return int(np.ceil(mm / spacing))

    @staticmethod
    def _ct_windowing(ct: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """
        Window to [vmin, vmax] then normalize to [0, 255].
        Avoids unexpected dtype issues by working in float and returning float.
        """
        ct = ct.astype(np.float32, copy=False)
        np.clip(ct, vmin, vmax, out=ct)
        ct = (ct - vmin) / (vmax - vmin) * 255.0
        return ct

    @staticmethod
    def windowing(image: np.ndarray, window: Tuple[float, float], norm: float = 1.0) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        np.clip(image, window[0], window[1], out=image)
        return (image - window[0]) * norm / (window[1] - window[0])

    def _default_roi(self, shape: Tuple[int, int, int], spacing: np.ndarray) -> ROI:
        """
        Replicates your original ROI logic in predictUmbilicus().
        ct shape: (axial, coronal, sagittal)
        spacing indexing follows your original usage:
          - axial spacing used spacing[2]
          - coronal uses spacing[0]
          - sagittal uses spacing[1]
        """
        a, c, s = shape
        axial = (
            int(a / 2 - self._mm2pix(100, spacing[2])),
            int(a / 2 + self._mm2pix(50, spacing[2])),
        )
        coronal = (0, int(c / 2))
        sagittal = (
            int(s / 2 - self._mm2pix(50, spacing[1])),
            int(s / 2 + self._mm2pix(50, spacing[1])),
        )
        # clamp into valid range
        axial = (max(0, axial[0]), min(a, axial[1]))
        coronal = (max(0, coronal[0]), min(c, coronal[1]))
        sagittal = (max(0, sagittal[0]), min(s, sagittal[1]))
        return ROI(axial=axial, coronal=coronal, sagittal=sagittal)

    # ---------------------------
    # Method 1: contour-based
    # ---------------------------
    def _method_contour(
        self, ct: np.ndarray, spacing: np.ndarray, roi: ROI
    ) -> Tuple[List[int], float]:
        axial_crop = roi.axial
        coronal_crop = roi.coronal
        sagittal_crop = roi.sagittal

        axial_scores: List[float] = []
        axial_point: List[np.ndarray] = []

        for z in range(axial_crop[0], axial_crop[1]):
            img = np.uint8(ct[z])

            # Remove noise
            img = cv2.erode(img, np.ones((5, 5), dtype=np.uint8))
            blur = cv2.GaussianBlur(img, (7, 7), 0)

            # Find contour
            _, thresh = cv2.threshold(blur, self.contour_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if len(contours) == 0:
                return [0, 0, 0], 0.0

            # Find largest contour
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours[0] = cv2.approxPolyDP(contours[0], 2, True)
            contour = contours[0][:, 0, :]  # (N,2) with (x,y)

            # ROI mask in this slice
            slice_mask = np.full(shape=img.shape, fill_value=255, dtype=np.uint8)
            slice_mask[: coronal_crop[0], :] = 0
            slice_mask[coronal_crop[1] :, :] = 0
            slice_mask[:, : sagittal_crop[0]] = 0
            slice_mask[:, sagittal_crop[1] :] = 0

            simplified_contour = contour[
                np.array(slice_mask[contour[:, 1], contour[:, 0]], dtype=bool)
            ]

            scores: List[float] = []
            for i in range(1, len(simplified_contour) - 1):
                l1 = simplified_contour[i] - simplified_contour[i - 1]
                l2 = simplified_contour[i + 1] - simplified_contour[i]
                n1 = np.linalg.norm(l1)
                n2 = np.linalg.norm(l2)
                if n1 == 0 or n2 == 0:
                    scores.append(0.0)
                    continue

                l1 = l1 / n1
                l2 = l2 / n2

                dist = 0.0
                if l1[0] <= 0 and l1[1] >= 0 and l2[0] <= 0 and l2[1] <= 0:
                    l3 = simplified_contour[i + 1] - simplified_contour[i - 1]
                    n3 = np.linalg.norm(l3)
                    if n3 != 0:
                        l3 = l3 / n3
                        p = simplified_contour[i - 1] + l3 * np.dot(
                            l3, simplified_contour[i] - simplified_contour[i - 1]
                        )
                        p = (p - simplified_contour[i]) * spacing[:2]
                        dist = float(np.linalg.norm(p) + np.sqrt(np.sum((l1 - l2) ** 2)))

                scores.append(dist)

            if scores:
                best_i = int(np.argmax(scores)) + 1
                axial_point.append(simplified_contour[best_i])
                axial_scores.append(float(np.max(scores)))
            else:
                axial_point.append(np.array([0, 0], dtype=int))
                axial_scores.append(0.0)

        max_idx = int(np.argmax(axial_scores)) if axial_scores else 0
        navel = [
            int(axial_point[max_idx][0]),  # x
            int(axial_point[max_idx][1]),  # y
            int(axial_crop[0] + max_idx),  # z
        ]
        return navel, float(axial_scores[max_idx]) if axial_scores else 0.0

    # ---------------------------
    # Method 2: intensity-based
    # ---------------------------
    def _method_intensity(
        self,
        ct: np.ndarray,
        spacing: np.ndarray,
        kernel_size_mm: float = 5.0,
        roi: Optional[ROI] = None,
    ) -> Tuple[List[int], float]:
        if roi is None:
            axial_crop = (0, ct.shape[0])
            coronal_crop = (0, ct.shape[1])
            sagittal_crop = (0, ct.shape[2])
        else:
            axial_crop, coronal_crop, sagittal_crop = roi.axial, roi.coronal, roi.sagittal

        # Find ROI mask
        mask = np.zeros(shape=ct.shape, dtype=np.float32)
        for z in range(axial_crop[0], axial_crop[1]):
            img = np.uint8(ct[z])

            blur = cv2.GaussianBlur(img, (7, 7), 0)
            _, thresh = cv2.threshold(blur, self.contour_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                return [0, 0, 0], 0.0

            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours[0] = cv2.approxPolyDP(contours[0], 2, True)

            slice_mask = np.zeros(shape=img.shape, dtype=np.uint8)
            slice_mask = cv2.drawContours(
                slice_mask,
                contours=contours,
                contourIdx=0,
                color=255,
                thickness=cv2.FILLED,
            )
            slice_mask[: coronal_crop[0], :] = 0
            slice_mask[coronal_crop[1] :, :] = 0
            slice_mask[:, : sagittal_crop[0]] = 0
            slice_mask[:, sagittal_crop[1] :] = 0

            mask[z] = slice_mask

        ct_masked = ct * (mask / 255.0)

        kernel_shape = [
            self._mm2pix(kernel_size_mm, spacing[2]),  # axial
            self._mm2pix(kernel_size_mm, spacing[0]),  # coronal
            self._mm2pix(kernel_size_mm, spacing[1]),  # sagittal
        ]
        kernel_shape = [k if k % 2 != 0 else max(1, k - 1) for k in kernel_shape]
        kernel3d = np.ones(shape=kernel_shape, dtype=np.float32)

        output = np.zeros(shape=ct.shape, dtype=np.float32)
        arg_max = [0, 0, 0]
        val_max = -np.inf

        # NOTE: Original code has a suspicious 'break' that exits only the innermost loop,
        # and uses c,s after break. We keep the structure but make it safer.
        for a in range(axial_crop[0], max(axial_crop[0], axial_crop[1] - kernel3d.shape[0])):
            for s in range(
                sagittal_crop[0], max(sagittal_crop[0], sagittal_crop[1] - kernel3d.shape[2])
            ):
                c = None
                for c in range(
                    coronal_crop[0], max(coronal_crop[0], coronal_crop[1] - kernel3d.shape[1])
                ):
                    if mask[a, c, s] != 0:
                        break  # keep original intent

                if c is None:
                    continue
                # guard: ensure c is in range
                if c < 0 or c >= ct.shape[1]:
                    continue
                if s < 0 or s >= ct.shape[2]:
                    continue

                output[a, c, s] = np.sum(
                    ct_masked[
                        a : a + kernel3d.shape[0],
                        c : c + kernel3d.shape[1],
                        s : s + kernel3d.shape[2],
                    ]
                    * kernel3d
                )

                if output[a, c, s] > val_max:
                    val_max = float(output[a, c, s])
                    arg_max = [
                        int(s + (kernel3d.shape[2] - 1) / 2),  # x (sagittal)
                        int(c + (kernel3d.shape[1] - 1) / 2),  # y (coronal)
                        int(a + (kernel3d.shape[0] - 1) / 2),  # z (axial)
                    ]

        # Original returned: np.mean(np.nonzero(output.reshape(-1))) which is a bit odd.
        # Keep similar magnitude but safer:
        nonzero = np.count_nonzero(output)
        score = float(nonzero)  # or float(val_max) depending on your intent

        return arg_max, score


class UmbilicusDetector:
    """
    Umbilicus (navel) detection class for medical images.

    This class provides functionality to:
    1. Find hole masks that may represent umbilicus
    2. Detect umbilicus point from the mask
    3. Return anatomical coordinates
    """

    def __init__(self) -> None:
        super().__init__()

        """
        Initialize the UmbilicusDetector.

        Parameters
        ----------

        """

    def detect(
        self,
        region_image: np.ndarray,
        region_mask: np.ndarray,
        region_contour: np.ndarray,
        region_info: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Comprehensive umbilicus detection using multiple approaches.

        Parameters
        ----------
        region_image : np.ndarray
            Extracted abdominal region image
        region_mask : np.ndarray
            Extracted abdominal region mask
        region_info : Dict[str, Any]
            Region boundary information containing:
            - width_start, width_end: width boundaries
            - depth_start, depth_end: depth boundaries
            - height_start, height_end: height boundaries

        Returns
        -------
        pd.DataFrame
            Detected umbilicus points with columns:
            - SI, AP, LR: coordinates
            - MIN_CV: curvature
            - MEAN_VAL: intensity values
        """

        self.region_image = region_image
        self.region_mask = region_mask
        self.region_contour = region_contour
        self.region_info = region_info

        contour_info = self.extract_region_contour_information()

        if contour_info.empty:
            return contour_info

        basis = [
            region_info["height_start"],
            region_info["depth_start"],
            region_info["width_start"],
        ]
        points = contour_info[["SI", "AP", "LR"]] + basis
        points[["MIN_CV", "MEAN_VAL"]] = contour_info[["MIN_CV", "MEAN_VAL"]]

        return points

    def extract_region_contour_information(self) -> pd.DataFrame:
        contours = self.region_contour.copy()
        image = self.region_image.copy()

        contour_info = []
        for i in range(2, len(contours) - 2):
            contour = contours[i]
            min_cv, start_idx, end_idx = self._find_min_curvature_location(contour)
            if start_idx >= end_idx:
                continue

            j, k = contour[start_idx:end_idx, :].mean(0).astype(int)
            if k <= 1:
                k += 2
            mean_val = image[i - 2 : i + 3, j : j + 5, k - 2 : k + 3].mean()

            contour_info.append([i, j, k, min_cv, mean_val, start_idx, end_idx])

        col_list = ["SI", "AP", "LR", "MIN_CV", "MEAN_VAL", "START_IDX", "END_IDX"]
        contour_pd = pd.DataFrame(contour_info, columns=col_list)
        if contour_pd.empty:
            return contour_pd

        contour_pd = contour_pd.sort_values(by=["MIN_CV", "MEAN_VAL"], ascending=[True, False])
        dist = (contour_pd[["SI", "AP", "LR"]] - contour_pd[["SI", "AP", "LR"]].iloc[0]).values
        contour_pd["DIST"] = np.sqrt(np.square(dist).sum(1))

        contour_pd_filtered = contour_pd[contour_pd["DIST"] < 10].reset_index(drop=True)

        return contour_pd_filtered

    def _find_min_curvature_location(self, contour: np.ndarray) -> Tuple[float, int, int]:
        curvature = self._compute_curvature(contour)

        arr = contour[:, 0]
        # 변경 지점 추출 (값이 변한 지점의 index)
        change_indices = np.where(arr[1:] != arr[:-1])[0] + 1
        split_indices = np.concatenate(([0], change_indices))
        if split_indices.size < 2:
            return float(np.min(curvature)), 0, len(contour) - 1

        min_cv_list = []
        for idx in range(len(split_indices) - 1):
            start = split_indices[idx]
            end = split_indices[idx + 1] - 1
            if end > start:
                min_cv_list.append(curvature[[start, end]].sum())
            else:
                min_cv_list.append(curvature[start])

        min_idx = int(np.argmin(min_cv_list))

        return (
            float(min_cv_list[min_idx]),
            int(split_indices[min_idx]),
            int(split_indices[min_idx + 1] - 1),
        )

    def _compute_curvature(self, xy: np.ndarray) -> np.ndarray:
        x = xy[:, 1]
        y = xy[:, 0]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        numerator = dx * ddy - dy * ddx
        denominator = (dx**2 + dy**2) ** 1.5 + 1e-8
        return numerator / denominator


@dataclass(frozen=True)
class GridConfig:
    spacing_mm: Tuple[float, float, float] = (
        50.0,
        50.0,
        50.0,
    )  # (z,y,x) or (??) -> 아래 코드의 vox 매핑 기준
    grid_size: int = 5


@dataclass(frozen=True)
class LandmarkConfig:
    offset_vox: Tuple[int, int, int] = (40, 0, 0)  # center에 더하는 offset (index space)


class LandmarkMaskGenerator:
    """
    Generate a grid of landmark points around a center, project them to nearest valid voxels
    in a reference volume, and return a labeled landmark mask image.

    Notes
    -----
    - center / points are treated as index coordinates in the SAME order used in your code:
      point = [i(=z slice), j(=y/AP), k(=x/LR)] 느낌의 인덱스 체계로 보입니다.
    - generate_grid_points()는 'z축 고정' 주석과 달리 실제로는 point = [dx, 0, dy]를 더합니다.
      즉, 두 축만 움직이고 한 축을 0으로 둡니다(원본 유지).
    """

    def __init__(
        self,
        grid: GridConfig = GridConfig(),
        landmark: LandmarkConfig = LandmarkConfig(),
    ):
        self.grid = grid
        self.landmark = landmark

    # -------------------------
    # Public API
    # -------------------------
    def generate_landmark_mask(
        self,
        center: np.ndarray,
        reference_volume: sitk.Image,
        *,
        offset_vox: Optional[Tuple[int, int, int]] = None,
        spacing_mm: Optional[Tuple[float, float, float]] = None,
        grid_size: Optional[int] = None,
    ) -> Tuple[sitk.Image, np.ndarray]:
        """
        Parameters
        ----------
        center : np.ndarray shape (3,)
            Center point (index space).
        reference_volume : sitk.Image
            Reference image used for projection and for copying metadata.
        offset_vox : tuple[int,int,int], optional
            Offset added to center in index space. Default: self.landmark.offset_vox
        spacing_mm : tuple[float,float,float], optional
            Grid spacing in mm. Default: self.grid.spacing_mm
        grid_size : int, optional
            Grid size (N -> N x N points). Default: self.grid.grid_size

        Returns
        -------
        landmark_volume : sitk.Image
            Labeled mask image where each point has label i+1.
        projected_points : np.ndarray
            Projected points (N*N, 3) int indices.
        """
        self._validate_center(center)

        offset_vox = offset_vox if offset_vox is not None else self.landmark.offset_vox
        spacing_mm = spacing_mm if spacing_mm is not None else self.grid.spacing_mm
        grid_size = grid_size if grid_size is not None else self.grid.grid_size

        # 1) center calib
        center_calib = center + np.array(offset_vox, dtype=int)

        # 2) generate grid points (index space)
        vox_size = reference_volume.GetSpacing()  # (sx, sy, sz) in SITK convention
        grid_points = self.generate_grid_points(
            center=center_calib,
            spacing_mm=spacing_mm,
            vox_size=vox_size,
            grid_size=grid_size,
        )

        # 3) project points to nearest valid voxels
        projected_points = self.project_to_nearest_vox(reference_volume, grid_points)

        # 4) create label mask (numpy z,y,x indexing from sitk.GetArrayFromImage)
        ref_arr = sitk.GetArrayFromImage(reference_volume)
        landmark_mask = np.zeros(ref_arr.shape, dtype=np.uint16)

        for i, lm_pos in enumerate(projected_points):
            z, y, x = lm_pos
            if (
                0 <= z < landmark_mask.shape[0]
                and 0 <= y < landmark_mask.shape[1]
                and 0 <= x < landmark_mask.shape[2]
            ):
                landmark_mask[z, y, x] = i + 1

        # NOTE: 아래 함수는 사용자가 이미 갖고 있다고 가정합니다.
        landmark_volume = sitk_copy_metainfo(reference_volume, landmark_mask)

        return landmark_volume, projected_points

    # -------------------------
    # Core methods (refactor)
    # -------------------------
    def generate_grid_points(
        self,
        center: np.ndarray,
        *,
        spacing_mm: Tuple[float, float, float],
        vox_size: Tuple[float, float, float],
        grid_size: int,
    ) -> np.ndarray:
        """
        Generate N x N grid points around center using spacing (mm),
        converted to voxel offsets using vox_size.

        Returns
        -------
        np.ndarray shape (N*N, 3) int
        """
        self._validate_center(center)
        if grid_size <= 0 or grid_size % 2 == 0:
            raise ValueError("grid_size should be a positive odd integer (e.g., 5)")

        half = grid_size // 2

        # spacing_mm -> voxel units
        # 원본 코드: spacing_vox = round(spacing/vox)
        # 주의: SITK GetSpacing은 (sx, sy, sz)인데,
        # 원본은 vox[2], vox[1], vox[0]을 섞어 사용합니다. 그대로 유지합니다.
        spacing_vox = np.round(np.array(spacing_mm) / np.array(vox_size)).astype(int)

        offsets_x = np.linspace(-half, half, grid_size) * spacing_vox[2]
        offsets_y = np.linspace(-half, half, grid_size) * spacing_vox[1]
        # offsets_z는 원본에 있었지만 실제로 사용하지 않아 제거(원본 로직 유지)
        # offsets_z = np.linspace(-half, half, grid_size) * spacing_vox[0]

        grid_points = []
        for dx in offsets_x[::-1]:
            for dy in offsets_y:
                # 원본 유지: point = center + [dx, 0, dy]
                point = center + np.array([dx, 0, dy])
                grid_points.append(point)

        return np.asarray(grid_points, dtype=int)

    def project_to_nearest_vox(
        self,
        reference_volume: sitk.Image,
        grid_points: np.ndarray,
    ) -> np.ndarray:
        """
        Project grid points to nearest valid voxels based on reference mask content.

        Returns
        -------
        np.ndarray shape (N,3) int
        """
        ref = sitk.GetArrayFromImage(reference_volume)  # (z,y,x)

        if ref.ndim != 3:
            raise ValueError("reference_volume must be 3D")

        if grid_points.ndim != 2 or grid_points.shape[1] != 3:
            raise ValueError("grid_points must be shape (N, 3)")

        # ref.max(2) -> (z,y) collapsed over x
        idx_is, _ = np.where(ref.max(2))
        if idx_is.size == 0:
            # reference에 유효 영역이 없으면 그대로 클램프만 해서 반환
            gp = grid_points.copy().astype(int)
            gp[:, 0] = np.clip(gp[:, 0], 0, ref.shape[0] - 1)
            gp[:, 1] = np.clip(gp[:, 1], 0, ref.shape[1] - 1)
            gp[:, 2] = np.clip(gp[:, 2], 0, ref.shape[2] - 1)
            return gp

        gp = grid_points.copy().astype(int)

        # clamp slice index(z)
        gp[:, 0] = np.clip(gp[:, 0], idx_is.min(), idx_is.max())
        gp[:, 0] = np.clip(gp[:, 0], 0, ref.shape[0] - 1)

        for i in range(len(gp)):
            z = int(gp[i, 0])

            idx_r, idx_c = np.where(ref[z])  # idx_r: y, idx_c: x of nonzero
            if idx_r.size == 0:
                # 해당 슬라이스가 비어있으면 범위 클램프만
                gp[i, 1] = int(np.clip(gp[i, 1], 0, ref.shape[1] - 1))
                gp[i, 2] = int(np.clip(gp[i, 2], 0, ref.shape[2] - 1))
                continue

            # 원본 로직 유지(다만 y/x min/max에 맞춰 clamp)
            if gp[i, 1] < idx_r.min():
                gp[i, 1] = int(idx_r.min())
            if gp[i, 2] > idx_c.max():
                gp[i, 2] = int(idx_c.max())

            y = int(np.clip(gp[i, 1], 0, ref.shape[1] - 1))
            x = int(np.clip(gp[i, 2], 0, ref.shape[2] - 1))

            # 원본: reference_image[z, :, x]에서 nonzero 첫 위치를 AP로 설정
            col = np.where(ref[z, :, x])[0]
            ap_pos = int(col[0]) if col.size > 0 else 0

            gp[i, 0] = z
            gp[i, 1] = ap_pos
            gp[i, 2] = x

        return gp

    # -------------------------
    # Validation
    # -------------------------
    @staticmethod
    def _validate_center(center: np.ndarray) -> None:
        if not isinstance(center, np.ndarray):
            raise TypeError("center must be a numpy array")
        if center.shape != (3,):
            raise ValueError("center must be shape (3,)")
