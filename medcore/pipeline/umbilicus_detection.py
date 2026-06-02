from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ..detect import UmbilicusDetector, UmbilicusPredictor, get_coronal_plane_degree
from ..io import ImageReader
from ..segment import AbdomenSegmenter
from ..utils import (
    sitk_get_array,
    sitk_make_euler3dtransform,
    sitk_resample_point_between_volumes,
    sitk_resampler,
)


def _safe_mean_hu(image: np.ndarray, si: int, ap: int, lr: int) -> float:
    depth, height, width = image.shape

    si0, si1 = max(si - 2, 0), min(si + 3, depth)
    ap0, ap1 = max(ap, 0), min(ap + 5, height)
    lr0, lr1 = max(lr - 2, 0), min(lr + 3, width)

    if si0 >= si1 or ap0 >= ap1 or lr0 >= lr1:
        return float("-inf")

    patch = image[si0:si1, ap0:ap1, lr0:lr1]
    if patch.size == 0:
        return float("-inf")
    return float(patch.mean())


@dataclass
class UmbilicusDetectionResult:
    umbilicus_points: pd.DataFrame
    candidates: pd.DataFrame
    raw_candidates: pd.DataFrame
    supine_volume: sitk.Image
    transform: sitk.Transform
    inverse_transform: sitk.Transform
    coronal_degree: float
    abdomen_info: dict[str, Any]


class UmbilicusDetectionPipeline:
    def __init__(
        self,
        *,
        predictor: UmbilicusPredictor | None = None,
        detector: UmbilicusDetector | None = None,
        abdomen_segmenter: AbdomenSegmenter | None = None,
        coronal_degree_threshold: float = 5.0,
        correction_angle: float = 15.0,
        resample_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        min_height_start_ratio: float = 0.9,
    ) -> None:
        self.predictor = predictor or UmbilicusPredictor()
        self.detector = detector or UmbilicusDetector()
        self.abdomen_segmenter = abdomen_segmenter or AbdomenSegmenter()
        self.coronal_degree_threshold = coronal_degree_threshold
        self.correction_angle = correction_angle
        self.resample_spacing = resample_spacing
        self.min_height_start_ratio = min_height_start_ratio

    def run(self, ct_input: str | Path | sitk.Image) -> UmbilicusDetectionResult:
        ct_vol = self._load_volume(ct_input)
        tfm_axis, tfm_axis_inv, coronal_degree = self._make_supine_transform(ct_vol)
        ct_vol_sup_rsl = self._resample_to_supine(ct_vol, tfm_axis)
        ct_img_sup_rsl = sitk_get_array(ct_vol_sup_rsl)

        point_xyz = self.predictor.predict(ct_vol_sup_rsl)
        abdominal_image, abdomen_mask, abdomen_info, abdomen_contour = (
            self.abdomen_segmenter.segment(ct_vol_sup_rsl)
        )
        points_df = self.detector.detect(
            region_image=abdominal_image,
            region_mask=abdomen_mask,
            region_contour=abdomen_contour,
            region_info=abdomen_info,
        )

        raw_candidates = self._build_candidates(point_xyz, points_df)
        raw_candidates["MEAN_HU"] = [
            _safe_mean_hu(
                ct_img_sup_rsl,
                si=int(row.SI),
                ap=int(row.AP),
                lr=int(row.LR),
            )
            for row in raw_candidates.itertuples(index=False)
        ]

        candidates = self._filter_candidates(raw_candidates, abdomen_info)
        point = self._select_candidate(candidates)
        umbilicus_points = self._build_umbilicus_points(
            point=point,
            ct_vol=ct_vol,
            ct_vol_sup_rsl=ct_vol_sup_rsl,
            transform=tfm_axis,
        )

        return UmbilicusDetectionResult(
            umbilicus_points=umbilicus_points,
            candidates=candidates,
            raw_candidates=raw_candidates,
            supine_volume=ct_vol_sup_rsl,
            transform=tfm_axis,
            inverse_transform=tfm_axis_inv,
            coronal_degree=coronal_degree,
            abdomen_info=abdomen_info,
        )

    def _load_volume(self, ct_input: str | Path | sitk.Image) -> sitk.Image:
        if isinstance(ct_input, sitk.Image):
            return ct_input
        return ImageReader(ct_input).read()

    def _make_supine_transform(
        self,
        ct_vol: sitk.Image,
    ) -> tuple[sitk.Transform, sitk.Transform, float]:
        coronal_degree = get_coronal_plane_degree(ct_vol)
        angle = self.correction_angle if coronal_degree > self.coronal_degree_threshold else 0.0
        tfm_axis = sitk_make_euler3dtransform(ct_vol, angle, axis="x")
        return tfm_axis, tfm_axis.GetInverse(), coronal_degree

    def _resample_to_supine(self, ct_vol: sitk.Image, transform: sitk.Transform) -> sitk.Image:
        return sitk_resampler(ct_vol, transform, self.resample_spacing)

    def _build_candidates(
        self,
        point_xyz: pd.DataFrame,
        points_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[pd.Series] = []
        for source, points in (("predictor", point_xyz), ("detector", points_df)):
            if points is None or points.empty:
                continue
            row = points.iloc[0].loc[["SI", "AP", "LR"]].copy()
            row["SOURCE"] = source
            rows.append(row)

        if not rows:
            raise RuntimeError("Failed to detect any umbilicus candidate.")

        candidates = pd.DataFrame(rows).reset_index(drop=True)
        candidates[["SI", "AP", "LR"]] = candidates[["SI", "AP", "LR"]].astype(int)
        return candidates

    def _filter_candidates(
        self,
        candidates: pd.DataFrame,
        abdomen_info: dict[str, Any],
    ) -> pd.DataFrame:
        min_si = float(abdomen_info["height_start"]) * self.min_height_start_ratio
        filtered = candidates[candidates["SI"] > min_si].reset_index(drop=True)
        if filtered.empty:
            raise RuntimeError("Failed to select an umbilicus candidate.")
        return filtered

    def _select_candidate(self, candidates: pd.DataFrame) -> pd.Series:
        return candidates.sort_values("MEAN_HU", ascending=False).iloc[0]

    def _build_umbilicus_points(
        self,
        point: pd.Series,
        ct_vol: sitk.Image,
        ct_vol_sup_rsl: sitk.Image,
        transform: sitk.Transform,
    ) -> pd.DataFrame:
        point_iso = point.values[:3].astype(int)
        point_raw = np.asarray(
            sitk_resample_point_between_volumes(
                point_iso,
                ct_vol_sup_rsl,
                ct_vol,
                transform,
            )
        )

        return pd.DataFrame(
            [point_raw, point_iso, ct_vol.GetSpacing()],
            index=["RAW", "ISO", "VOX"],
            columns=["IS", "AP", "LR"],
        )
