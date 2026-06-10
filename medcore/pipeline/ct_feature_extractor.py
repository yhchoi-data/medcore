from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import center_of_mass

from ..detect import get_coronal_plane_degree
from ..feature import (
    compute_label_areas,
    compute_label_volumns,
    extract_abdominal_distance_metrics,
    extract_craniocaudal_fat_volume,
    extract_optimal_transverse_process_slice,
    extract_pancreatic_distance_metrics,
    extract_peripancreatic_fat_volume,
)
from ..io import ImageReader
from ..segment import RegionCenterlinePartitioner, TorsoSegmenter
from ..utils import (
    figure_3d_region_with_centerline,
    figure_overlay_abdominal_distance_metrics,
    figure_overlay_label_on_slices,
    figure_overlay_label_reference_slice,
    figure_overlay_pancreatic_craniocaudal_slices,
    figure_overlay_pancreatic_distance_metrics,
    sitk_get_shape_features,
    sitk_make_euler3dtransform,
    sitk_resampler,
)

PANCREATIC_MORPHOLOGY_KEYS = (
    "curvature_head_pancreas",
    "curvature_body_pancreas",
    "curvature_tail_pancreas",
    "curvature_first_pancreas",
    "curvature_last_pancreas",
    "curvature_pancreas",
    "elongation_pancreas",
    "flatness_pancreas",
    "roundness_pancreas",
    "length(cm)_pancres",
    "volume(cm3)_pancreas",
)
PANCREATIC_DISTANCE_KEYS = (
    "center_of_mass_l1",
    "anterior_point_l1",
    "center_point_pancreas",
    "head_point_pancreas",
    "tail_point_pancreas",
    "IS_dist_center_from_l1(mm)",
    "AP_dist_center_from_l1(mm)",
    "RL_dist_center_from_l1(mm)",
    "IS_dist_head_from_l1(mm)",
    "AP_dist_head_from_l1(mm)",
    "RL_dist_head_from_l1(mm)",
    "IS_dist_tail_from_l1(mm)",
    "AP_dist_tail_from_l1(mm)",
    "RL_dist_tail_from_l1(mm)",
    "IS_dist_center_from_l1_anterior(mm)",
    "AP_dist_center_from_l1_anterior(mm)",
    "RL_dist_center_from_l1_anterior(mm)",
    "IS_dist_head_from_l1_anterior(mm)",
    "AP_dist_head_from_l1_anterior(mm)",
    "RL_dist_head_from_l1_anterior(mm)",
    "IS_dist_tail_from_l1_anterior(mm)",
    "AP_dist_tail_from_l1_anterior(mm)",
    "RL_dist_tail_from_l1_anterior(mm)",
)


def pancreatic_morphology(
    volume_pancreas: sitk.Image,
    *,
    partitioner: RegionCenterlinePartitioner | None = None,
    cutoffs: tuple[float, ...] = (0.25, 0.75),
    target_spacing_mm: float = 1.0,
    smoothing: float = 15.0,
    label: int = 1,
) -> dict[str, Any]:
    """Extract pancreas centerline curvature and SimpleITK shape features."""
    if partitioner is None:
        partitioner = RegionCenterlinePartitioner(
            mask_volume=volume_pancreas,
            cutoffs=cutoffs,
            target_spacing_mm=target_spacing_mm,
            smoothing=smoothing,
        )
        partitioner.run()

    if partitioner.region_curvature_indices is None:
        raise RuntimeError("Pancreas curvature indices are not available.")
    if partitioner.centerline_length is None:
        raise RuntimeError("Pancreas centerline length is not available.")

    curvature = partitioner.region_curvature_indices
    shape_info = sitk_get_shape_features(volume_pancreas, label=label)

    return {
        "curvature_head_pancreas": _first_existing_value(curvature, "region_1", "head"),
        "curvature_body_pancreas": _first_existing_value(curvature, "region_2", "body"),
        "curvature_tail_pancreas": _first_existing_value(curvature, "region_3", "tail"),
        "curvature_first_pancreas": curvature["first"],
        "curvature_last_pancreas": curvature["last"],
        "curvature_pancreas": partitioner.curvature_index,
        "elongation_pancreas": shape_info["elongation"],
        "flatness_pancreas": shape_info["flatness"],
        "roundness_pancreas": shape_info["roundness"],
        "length(cm)_pancres": partitioner.centerline_length / 10.0,
        "volume(cm3)_pancreas": shape_info["volume_ml"],
    }


def pancreatic_distance(
    volume_l1: sitk.Image,
    *,
    partitioner: RegionCenterlinePartitioner,
) -> dict[str, Any]:
    """Extract pancreas point offsets from L1 centroid and anterior L1 point."""
    if partitioner.smooth_centerline is None:
        raise RuntimeError("Pancreas smooth centerline is not available.")
    if partitioner.centerline_midpoint_voxel_zyx is None:
        raise RuntimeError("Pancreas centerline midpoint is not available.")

    mask_l1 = sitk.GetArrayFromImage(volume_l1) > 0
    if not np.any(mask_l1):
        raise ValueError("L1 mask is empty.")

    center_l1 = np.asarray(center_of_mass(mask_l1), dtype=np.float64).astype(np.int64)
    optimal_idx = extract_optimal_transverse_process_slice(volume_l1)
    l1_slice = mask_l1[optimal_idx]
    if not np.any(l1_slice):
        raise ValueError("No L1 pixels found on the selected transverse process slice.")

    center_l1_anterior = np.array(
        [optimal_idx, np.where(l1_slice)[0].min(), center_l1[2]],
        dtype=np.int64,
    )
    center_pancreas = np.asarray(partitioner.centerline_midpoint_voxel_zyx, dtype=np.int64)
    head_pancreas = np.asarray(partitioner.smooth_centerline[0], dtype=np.float64).astype(np.int64)
    tail_pancreas = np.asarray(partitioner.smooth_centerline[-1], dtype=np.float64).astype(np.int64)

    results: dict[str, Any] = {
        "center_of_mass_l1": center_l1,
        "anterior_point_l1": center_l1_anterior,
        "center_point_pancreas": center_pancreas,
        "head_point_pancreas": head_pancreas,
        "tail_point_pancreas": tail_pancreas,
    }
    points = {
        "center": center_pancreas,
        "head": head_pancreas,
        "tail": tail_pancreas,
    }
    for ref_name, ref_point in (
        ("l1", center_l1),
        ("l1_anterior", center_l1_anterior),
    ):
        for point_name, point in points.items():
            _update_zyx_distance_metrics(
                results,
                name=f"{point_name}_from_{ref_name}",
                diff_zyx=point - ref_point,
                spacing_zyx=np.asarray(volume_l1.GetSpacing()[::-1], dtype=np.float64),
            )
    return results


def _first_existing_value(values: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _update_zyx_distance_metrics(
    results: dict[str, Any],
    *,
    name: str,
    diff_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
) -> None:
    diff_mm = np.rint(np.asarray(diff_zyx, dtype=np.float64) * spacing_zyx).astype(int)
    results[f"IS_dist_{name}(mm)"] = int(diff_mm[0])
    results[f"AP_dist_{name}(mm)"] = int(diff_mm[1])
    results[f"RL_dist_{name}(mm)"] = int(diff_mm[2])


class CTFeatureExtractor:
    metadata_keys = ("hutom_id", "study_uid", "series_uid", "fpath")
    peripancreatic_base_metric_keys = (
        "total_shell_voxel_count",
        "total_shell_volume(cm3)",
        "total_shell_fat_voxel_count",
        "total_shell_fat_volume(cm3)",
    )
    craniocaudal_base_metric_keys = (
        "total_fat_voxel_count",
        "total_fat_volume(cm3)",
    )
    craniocaudal_output_key_map = {
        "total_fat_voxel_count": "craniocaudal_total_fat_voxel_count",
        "total_fat_volume(cm3)": "craniocaudal_total_fat_volume(cm3)",
    }
    craniocaudal_octants = (
        "superior_anterior_left",
        "superior_anterior_right",
        "superior_posterior_left",
        "superior_posterior_right",
        "inferior_anterior_left",
        "inferior_anterior_right",
        "inferior_posterior_left",
        "inferior_posterior_right",
    )
    abdominal_distance_metric_keys = (
        "Max_SFT(cm)",
        "Min_SFT(cm)",
        "Mean_SFT(cm)",
        "Median_SFT(cm)",
        "Left_SFT(cm)",
        "Right_SFT(cm)",
        "Anterior_SFT(cm)",
        "LRD(cm)",
        "APD(cm)",
        "AP(cm)",
    )
    pancreatic_morphology_metric_keys = PANCREATIC_MORPHOLOGY_KEYS
    pancreatic_distance_metric_keys = PANCREATIC_DISTANCE_KEYS

    def __init__(
        self,
        save_dir: str | Path | None = None,
        *,
        save_figures: bool | None = None,
        pancreas_cutoffs: tuple[float, float] = (0.25, 0.75),
        shell_inner_mm: float = 5,
        shell_outer_mm: float = 10,
        hu_range: tuple[float, float] = (-190, -30),
        coronal_degree_threshold: float = 5,
        craniocaudal_center_cutoff: float | None = None,
    ) -> None:
        if save_figures is None:
            save_figures = save_dir is not None
        if save_figures and save_dir is None:
            raise ValueError("save_dir must be provided when save_figures=True.")
        if craniocaudal_center_cutoff is not None:
            craniocaudal_center_cutoff = float(craniocaudal_center_cutoff)
            if not math.isfinite(craniocaudal_center_cutoff):
                raise ValueError("`craniocaudal_center_cutoff` must be finite.")
            if craniocaudal_center_cutoff <= 0.0 or craniocaudal_center_cutoff >= 1.0:
                raise ValueError("`craniocaudal_center_cutoff` must be between 0 and 1.")

        self.save_dir = None if save_dir is None else Path(save_dir)
        self.save_figures = save_figures
        self.pancreas_cutoffs = pancreas_cutoffs
        self.shell_inner_mm = shell_inner_mm
        self.shell_outer_mm = shell_outer_mm
        self.hu_range = hu_range
        self.coronal_degree_threshold = coronal_degree_threshold
        self.craniocaudal_center_cutoff = craniocaudal_center_cutoff
        self.torso_segmentor = TorsoSegmenter()

    def run(
        self,
        ct_path: str | Path,
        mask_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self._path_context(ct_path, mask_path, metadata)
        volumes = self._load_volumes(ctx)

        vol_pancreas_calib, n_components = self._largest_component(volumes["vol_pancreas"])
        pancreas_is_empty = n_components == 0
        if pancreas_is_empty:
            pancreas_fat_metrics = self._empty_peripancreatic_metrics()
        else:
            pancreas_fat_metrics = self._extract_peripancreatic_metrics(
                vol=volumes["vol"],
                vol_pancreas=vol_pancreas_calib,
                ctx=ctx,
            )

        tfm_axis, coronal_degree = self._make_supine_transform(volumes["vol"])
        sup = self._resample_to_supine(volumes, vol_pancreas_calib, tfm_axis)
        optimal_idx = extract_optimal_transverse_process_slice(sup["vol_l3"])

        abdominal_distance_metrics = extract_abdominal_distance_metrics(
            volume=sup["vol"],
            volume_torso=sup["vol_torso"],
            volume_sfat=sup["vol_sfat"],
            volume_l3=sup["vol_l3"],
            l3_index=optimal_idx,
        )
        self._save_abdominal_metric_figure(ctx, sup, abdominal_distance_metrics)

        if pancreas_is_empty:
            pancreatic_distance_metrics = self._empty_pancreatic_distance_metrics()
            pancreas_ccfat_metrics = self._empty_craniocaudal_fat_metrics()
            pancreatic_geometry_metrics = self._empty_pancreatic_geometry_metrics()
        else:
            pancreatic_distance_metrics = extract_pancreatic_distance_metrics(
                volume_torso=sup["vol_torso"],
                volume_pancreas=sup["vol_pancreas"],
            )
            self._save_pancreatic_distance_figure(ctx, sup, pancreatic_distance_metrics)

            pancreas_ccfat_center_mask = self._craniocaudal_center_mask(sup["vol_pancreas"])
            pancreas_ccfat_metrics = extract_craniocaudal_fat_volume(
                vol=sup["vol"],
                vol_mask=sup["vol_pancreas"],
                vol_shell=sup["vol_vfat"],
                hu_range=self.hu_range,
                center_mask=pancreas_ccfat_center_mask,
            )
            self._save_pancreatic_ccfat_figure(
                ctx,
                sup,
                pancreas_ccfat_metrics,
                center_mask=pancreas_ccfat_center_mask,
            )
            pancreatic_geometry_metrics = self._extract_pancreatic_geometry_metrics(
                volumes=volumes,
                vol_pancreas=vol_pancreas_calib,
                tfm_axis=tfm_axis,
            )

        metric_row = self._build_metric_row(
            n_components=n_components,
            coronal_degree=coronal_degree,
            pancreas_fat_metrics=pancreas_fat_metrics,
            pancreas_ccfat_metrics=pancreas_ccfat_metrics,
            pancreatic_distance_metrics=pancreatic_distance_metrics,
            pancreatic_geometry_metrics=pancreatic_geometry_metrics,
            abdominal_distance_metrics=abdominal_distance_metrics,
        )
        tissue_metrics = self._extract_organ_tissue_metrics(
            sample=ctx["metadata"]["fpath"],
            mpath=ctx["mask_path"],
            vol=sup["vol"],
            vol_l3=sup["vol_l3"],
            tfm_axis=tfm_axis,
            optimal_idx=optimal_idx,
        )
        if pancreas_is_empty:
            tissue_metrics["pancreas_volume(cm3)"] = None

        result = {key: ctx["metadata"].get(key, "") for key in self.metadata_keys}
        result.update(metric_row)
        result.update(tissue_metrics)
        return result

    def _path_context(
        self,
        ct_path: str | Path,
        mask_path: str | Path,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        fpath = Path(ct_path)
        mpath = Path(mask_path)
        inferred_metadata = self._metadata_from_ct_path(fpath)
        if metadata is not None:
            inferred_metadata.update(
                {key: "" if value is None else str(value) for key, value in metadata.items()}
            )

        return {
            "ct_path": fpath,
            "mask_path": mpath,
            "metadata": inferred_metadata,
            "prefix": self._figure_prefix(fpath),
        }

    def _load_volumes(self, ctx: dict[str, Any]) -> dict[str, sitk.Image]:
        mpath = ctx["mask_path"]
        return {
            "vol": ImageReader(ctx["ct_path"]).read(),
            "vol_l1": ImageReader(self._l1_volume_path(mpath)).read(),
            "vol_l3": ImageReader(mpath / "vertebrae_L3.nii.gz").read(),
            "vol_vfat": ImageReader(mpath / "visceral_fat.nii.gz").read(),
            "vol_sfat": ImageReader(mpath / "subcutaneous_fat.nii.gz").read(),
            "vol_pancreas": ImageReader(mpath / "pancreas.nii.gz").read(),
        }

    def _l1_volume_path(self, mpath: Path) -> Path:
        l1_path = mpath / "vertebrae_l1.nii.gz"
        if l1_path.exists():
            return l1_path
        return mpath / "vertebrae_L1.nii.gz"

    def _largest_component(self, vol_pancreas: sitk.Image) -> tuple[sitk.Image, int]:
        cc = sitk.RelabelComponent(sitk.ConnectedComponent(vol_pancreas))
        n_components = int(sitk.GetArrayFromImage(cc).max())
        if n_components == 0:
            return vol_pancreas > 0, n_components
        return cc == 1, n_components

    def _empty_peripancreatic_metrics(self) -> dict[str, Any]:
        keys = list(self.peripancreatic_base_metric_keys)
        for region_name in self._pancreas_fat_region_names():
            keys.append(f"{region_name}_fat_voxel_count")
            keys.append(f"{region_name}_fat_volume(cm3)")
        # keys.append("curvature_index")
        # for region_name in self._pancreas_curvature_region_names():
        #     keys.append(f"{region_name}_curvature_index")
        # keys.append("first_curvature_index")
        # keys.append("last_curvature_index")
        return {key: None for key in keys}

    def _n_pancreas_regions(self) -> int:
        try:
            return len(self.pancreas_cutoffs) + 1
        except TypeError:
            return 2

    def _pancreas_curvature_region_names(self) -> tuple[str, ...]:
        n_regions = self._n_pancreas_regions()
        if n_regions == 3:
            # RegionCenterlinePartitioner's pancreas centerline currently follows head -> tail.
            return ("head", "body", "tail")
        return tuple(f"region_{label}" for label in range(1, n_regions + 1))

    def _pancreas_fat_region_names(self) -> tuple[str, ...]:
        n_regions = self._n_pancreas_regions()
        if n_regions == 3:
            return ("pancreas_head", "pancreas_body", "pancreas_tail")
        return tuple(f"region_{label}" for label in range(1, n_regions + 1))

    def _empty_pancreatic_distance_metrics(self) -> dict[str, Any]:
        return {"PAAD(cm)": None}

    def _empty_pancreatic_geometry_metrics(self) -> dict[str, Any]:
        keys = (*self.pancreatic_morphology_metric_keys, *self.pancreatic_distance_metric_keys)
        return {key: None for key in keys}

    def _empty_craniocaudal_fat_metrics(self) -> dict[str, Any]:
        keys = list(self.craniocaudal_base_metric_keys)
        for octant in self.craniocaudal_octants:
            keys.append(f"{octant}_fat_voxel_count")
            keys.append(f"{octant}_fat_volume(cm3)")
        return {key: None for key in keys}

    def _craniocaudal_center_mask(
        self,
        vol_pancreas: sitk.Image,
    ) -> tuple[float, float, float] | None:
        if self.craniocaudal_center_cutoff is None:
            return None

        partitioner = RegionCenterlinePartitioner(
            mask_volume=vol_pancreas,
            cutoffs=self.craniocaudal_center_cutoff,
            target_spacing_mm=1.0,
            smoothing=15.0,
        )
        partitioner.run()
        if partitioner.cutoff_voxels_zyx is None or len(partitioner.cutoff_voxels_zyx) == 0:
            raise RuntimeError("Failed to compute craniocaudal center cutoff voxel.")

        center_mask = tuple(float(value) for value in partitioner.cutoff_voxels_zyx[0])
        if len(center_mask) != 3:
            raise RuntimeError("Craniocaudal center cutoff voxel must have three coordinates.")
        return center_mask

    def _extract_peripancreatic_metrics(
        self,
        vol: sitk.Image,
        vol_pancreas: sitk.Image,
        ctx: dict[str, Any],
    ) -> dict[str, Any]:
        partitioner = RegionCenterlinePartitioner(
            mask_volume=vol_pancreas,
            cutoffs=self.pancreas_cutoffs,
            target_spacing_mm=1.0,
            smoothing=15.0,
        )
        vol_pancreas_region = partitioner.run()
        vol_pancreas_shell = partitioner.create_shell_mask(
            inner_mm=self.shell_inner_mm,
            outer_mm=self.shell_outer_mm,
        )
        vol_pancreas_shell_region = partitioner.create_region_label(volume=vol_pancreas_shell)

        metrics = extract_peripancreatic_fat_volume(
            vol=vol,
            vol_mask=vol_pancreas,
            vol_shell=vol_pancreas_shell,
            vol_shell_region=vol_pancreas_shell_region,
            hu_range=self.hu_range,
        )
        # metrics.update(
        #     partitioner.compute_curvature_indices(
        #         region_names=self._pancreas_curvature_region_names(),
        #     )
        # )
        self._save_region_figure(ctx, vol_pancreas_region, partitioner)
        return metrics

    def _extract_pancreatic_geometry_metrics(
        self,
        volumes: dict[str, sitk.Image],
        vol_pancreas: sitk.Image,
        tfm_axis: sitk.Transform,
    ) -> dict[str, Any]:
        vol_pancreas_sup_rsl = sitk_resampler(
            vol_pancreas,
            tfm_axis,
            new_spacing=(1.0, 1.0, 1.0),
            interpolation="nn",
            default_pixel=0,
        )
        partitioner = RegionCenterlinePartitioner(
            mask_volume=vol_pancreas_sup_rsl,
            cutoffs=self.pancreas_cutoffs,
            target_spacing_mm=1.0,
            smoothing=15.0,
        )
        partitioner.run()

        metrics = pancreatic_morphology(
            vol_pancreas_sup_rsl,
            partitioner=partitioner,
        )
        vol_l1 = volumes.get("vol_l1")
        if vol_l1 is None:
            metrics.update({key: None for key in self.pancreatic_distance_metric_keys})
            return metrics

        vol_l1_sup_rsl = sitk_resampler(
            vol_l1,
            tfm_axis,
            new_spacing=(1.0, 1.0, 1.0),
            interpolation="nn",
            default_pixel=0,
        )
        metrics.update(pancreatic_distance(vol_l1_sup_rsl, partitioner=partitioner))
        return metrics

    def _make_supine_transform(self, vol: sitk.Image) -> tuple[sitk.Transform, float]:
        coronal_degree = get_coronal_plane_degree(vol, margin=int(vol.GetSize()[2] / 3))
        angle = 15 if coronal_degree > self.coronal_degree_threshold else 0
        return sitk_make_euler3dtransform(vol, angle, axis="x"), coronal_degree

    def _resample_to_supine(
        self,
        volumes: dict[str, sitk.Image],
        vol_pancreas: sitk.Image,
        tfm_axis: sitk.Transform,
    ) -> dict[str, sitk.Image]:
        vol_sup = sitk_resampler(volumes["vol"], tfm_axis)
        return {
            "vol": vol_sup,
            "vol_torso": self.torso_segmentor.segment(volume=vol_sup),
            "vol_sfat": sitk_resampler(volumes["vol_sfat"], tfm_axis, interpolation="nn"),
            "vol_vfat": sitk_resampler(volumes["vol_vfat"], tfm_axis, interpolation="nn"),
            "vol_pancreas": sitk_resampler(vol_pancreas, tfm_axis, interpolation="nn"),
            "vol_l3": sitk_resampler(volumes["vol_l3"], tfm_axis, interpolation="nn"),
        }

    def _extract_organ_tissue_metrics(
        self,
        sample: str,
        mpath: Path,
        vol: sitk.Image,
        vol_l3: sitk.Image,
        tfm_axis: sitk.Transform,
        optimal_idx: int,
    ) -> dict[str, Any]:
        organfiles = {
            1: mpath / "gallbladder.nii.gz",
            2: mpath / "liver.nii.gz",
            3: mpath / "pancreas.nii.gz",
            4: mpath / "spleen.nii.gz",
            5: mpath / "stomach.nii.gz",
        }
        tissuefiles = {
            1: mpath / "subcutaneous_fat.nii.gz",
            2: mpath / "visceral_fat.nii.gz",
            3: mpath / "muscle.nii.gz",
            4: mpath / "iliopsoas_right.nii.gz",
            5: mpath / "iliopsoas_left.nii.gz",
        }

        organ_masks, organ_vols = compute_label_volumns(
            organfiles,
            transform=tfm_axis,
            return_vols=True,
        )
        organ_list = organ_vols.columns.tolist()
        organ_vols = organ_vols.rename(columns={name: f"{name}_volume(cm3)" for name in organ_list})

        tissue_slice_masks, tissue_slice_areas = compute_label_areas(
            tissuefiles,
            slices_index=optimal_idx,
            transform=tfm_axis,
            return_vols=True,
        )
        tissue_list = tissue_slice_areas.columns.tolist()
        tissue_slice_areas = tissue_slice_areas.rename(
            columns={name: f"{name}_l3_area(cm2)" for name in tissue_list}
        )
        tissue_slice_areas["slice_index_l3"] = optimal_idx

        self._save_organ_tissue_qc(
            sample=sample,
            vol=vol,
            organ_masks=organ_masks,
            organ_list=organ_list,
            tissue_slice_masks=tissue_slice_masks,
            tissue_list=tissue_list,
            vol_l3=vol_l3,
            optimal_idx=optimal_idx,
        )

        metrics_tissue = pd.concat([organ_vols, tissue_slice_areas], axis=1)
        return metrics_tissue.iloc[0].to_dict()

    def _build_metric_row(
        self,
        n_components: int,
        coronal_degree: float,
        pancreas_fat_metrics: dict[str, Any],
        pancreas_ccfat_metrics: dict[str, Any],
        pancreatic_distance_metrics: dict[str, Any],
        pancreatic_geometry_metrics: dict[str, Any],
        abdominal_distance_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "N.Pancreas_Connected_Components": n_components if n_components > 0 else None
        }
        row.update(self._peripancreatic_fat_metrics_for_output(pancreas_fat_metrics))
        row["coronal_plane_degree"] = coronal_degree
        row.update(
            {key: abdominal_distance_metrics[key] for key in self.abdominal_distance_metric_keys}
        )
        row.update(pancreatic_distance_metrics)
        row.update(pancreatic_geometry_metrics)
        row.update(self._craniocaudal_fat_metrics_for_output(pancreas_ccfat_metrics))
        return row

    def _craniocaudal_fat_metrics_for_output(
        self,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            self.craniocaudal_output_key_map.get(key, key): value for key, value in metrics.items()
        }

    def _peripancreatic_fat_metrics_for_output(
        self,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        region_names = self._pancreas_fat_region_names()
        output: dict[str, Any] = {}
        for key, value in metrics.items():
            output_key = key
            for label, region_name in enumerate(region_names, start=1):
                prefix = f"region_{label}_"
                if key.startswith(prefix):
                    output_key = f"{region_name}_{key[len(prefix):]}"
                    break
            output[output_key] = value
        return output

    def _save_region_figure(
        self,
        ctx: dict[str, Any],
        vol_pancreas_region: sitk.Image,
        partitioner: RegionCenterlinePartitioner,
    ) -> None:
        if not self.save_figures:
            return
        fname = f"{ctx['prefix']}_region.png"
        figure_3d_region_with_centerline(
            vol_pancreas_region,
            partitioner.smooth_centerline,
            show=False,
            save_path=self._figure_path("abdominal_metric", fname),
        )

    def _save_abdominal_metric_figure(
        self,
        ctx: dict[str, Any],
        sup: dict[str, sitk.Image],
        abdominal_distance_metrics: dict[str, Any],
    ) -> None:
        if not self.save_figures:
            return
        fname = f"{ctx['prefix']}_abdominal_metrics.png"
        figure_overlay_abdominal_distance_metrics(
            volume=sup["vol"],
            volume_sfat=sup["vol_sfat"],
            volume_l3=sup["vol_l3"],
            distance_metrics=abdominal_distance_metrics,
            show=False,
            save_path=self._figure_path("abdominal_metric", fname),
        )

    def _save_pancreatic_distance_figure(
        self,
        ctx: dict[str, Any],
        sup: dict[str, sitk.Image],
        pancreatic_distance_metrics: dict[str, Any],
    ) -> None:
        if not self.save_figures:
            return
        fname = f"{ctx['prefix']}_pancreatic_metrics.png"
        figure_overlay_pancreatic_distance_metrics(
            volume=sup["vol"],
            volume_torso=sup["vol_torso"],
            volume_pancreas=sup["vol_pancreas"],
            distance_metrics=pancreatic_distance_metrics,
            alpha=0.75,
            show=False,
            save_path=self._figure_path("abdominal_metric", fname),
        )

    def _save_pancreatic_ccfat_figure(
        self,
        ctx: dict[str, Any],
        sup: dict[str, sitk.Image],
        pancreas_ccfat_metrics: dict[str, Any],
        center_mask: tuple[float, float, float] | None = None,
    ) -> None:
        if not self.save_figures:
            return
        fname = f"{ctx['prefix']}_pancreatic_vfat_volumes.png"
        volume_metrics = pancreas_ccfat_metrics
        if center_mask is not None:
            volume_metrics = {**pancreas_ccfat_metrics, "center_of_mask": center_mask}
        figure_overlay_pancreatic_craniocaudal_slices(
            volume=sup["vol"],
            volume_pancreas=sup["vol_pancreas"],
            volume_visceralfat=sup["vol_vfat"],
            volume_metrics=volume_metrics,
            show=False,
            save_path=self._figure_path("abdominal_metric", fname),
        )

    def _save_organ_tissue_qc(
        self,
        sample: str,
        vol: sitk.Image,
        organ_masks: sitk.Image,
        organ_list: list[str],
        tissue_slice_masks: sitk.Image,
        tissue_list: list[str],
        vol_l3: sitk.Image,
        optimal_idx: int,
    ) -> None:
        if not self.save_figures:
            return
        qc_prefix = self._qc_prefix(sample)
        figure_overlay_label_on_slices(
            vol,
            organ_masks,
            labelname=organ_list,
            alpha=0.75,
            show=False,
            save_path=self._figure_path("organ", f"{qc_prefix}_qc_organ.png"),
        )
        figure_overlay_label_reference_slice(
            vol,
            tissue_slice_masks,
            vol_l3,
            slice_idx=optimal_idx,
            labelname=tissue_list,
            alpha=0.5,
            show=False,
            save_path=self._figure_path("tissue", f"{qc_prefix}_qc_tissue_slice.png"),
        )

    def _figure_path(self, subdir: str, fname: str) -> Path:
        if self.save_dir is None:
            raise ValueError("save_dir must be provided to save figures.")
        out_dir = self.save_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / fname

    @classmethod
    def _metadata_from_ct_path(cls, fpath: Path) -> dict[str, str]:
        sample = cls._sample_from_ct_path(fpath)
        parts = sample.split("/")
        return {
            "hutom_id": parts[0] if len(parts) > 0 else "",
            "study_uid": parts[3] if len(parts) > 3 else "",
            "series_uid": parts[4] if len(parts) > 4 else "",
            "fpath": sample,
        }

    @classmethod
    def _figure_prefix(cls, fpath: Path) -> str:
        parts = list(fpath.parts)
        left = parts[5] if len(parts) > 5 else cls._strip_nii_suffix(fpath.name)
        right = cls._strip_nii_suffix(fpath.name)
        return f"{left}_{right}"

    @classmethod
    def _sample_from_ct_path(cls, fpath: Path) -> str:
        parts = list(fpath.parts)
        if "CT_Nifti" in parts:
            rel_parts = parts[parts.index("CT_Nifti") + 1 :]
        else:
            rel_parts = [fpath.name]

        if not rel_parts:
            return cls._strip_nii_suffix(fpath.name)

        rel_parts = list(rel_parts)
        rel_parts[-1] = cls._strip_nii_suffix(rel_parts[-1])
        return "/".join(rel_parts)

    @staticmethod
    def _strip_nii_suffix(name: str) -> str:
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return Path(name).stem

    @classmethod
    def _qc_prefix(cls, sample: str) -> str:
        parts = [part for part in str(sample).split("/") if part]
        if not parts:
            return ""
        if len(parts) == 1:
            return cls._strip_nii_suffix(parts[0])

        if "CT_Nifti" in parts and parts.index("CT_Nifti") + 1 < len(parts):
            hutom_id = parts[parts.index("CT_Nifti") + 1]
        else:
            hutom_id = parts[0]

        name = cls._strip_nii_suffix(parts[-1])
        if not hutom_id or hutom_id == name:
            return name
        return f"{hutom_id}_{name}"
