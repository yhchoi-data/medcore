from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import SimpleITK as sitk

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
    sitk_make_euler3dtransform,
    sitk_resampler,
)


class CTFeatureExtractor:
    metadata_keys = ("hutom_id", "study_uid", "series_uid", "fpath")

    def __init__(
        self,
        save_dir: str | Path | None = None,
        *,
        save_figures: bool = True,
        pancreas_cutoffs: tuple[float, float] = (0.25, 0.75),
        shell_inner_mm: float = 5,
        shell_outer_mm: float = 10,
        hu_range: tuple[float, float] = (-190, -30),
        coronal_degree_threshold: float = 5,
    ) -> None:
        if save_figures and save_dir is None:
            raise ValueError("save_dir must be provided when save_figures=True.")

        self.save_dir = None if save_dir is None else Path(save_dir)
        self.save_figures = save_figures
        self.pancreas_cutoffs = pancreas_cutoffs
        self.shell_inner_mm = shell_inner_mm
        self.shell_outer_mm = shell_outer_mm
        self.hu_range = hu_range
        self.coronal_degree_threshold = coronal_degree_threshold
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

        pancreatic_distance_metrics = extract_pancreatic_distance_metrics(
            volume_torso=sup["vol_torso"],
            volume_pancreas=sup["vol_pancreas"],
        )
        self._save_pancreatic_distance_figure(ctx, sup, pancreatic_distance_metrics)

        pancreas_ccfat_metrics = extract_craniocaudal_fat_volume(
            vol=sup["vol"],
            vol_mask=sup["vol_pancreas"],
            vol_shell=sup["vol_vfat"],
            hu_range=self.hu_range,
        )
        self._save_pancreatic_ccfat_figure(ctx, sup, pancreas_ccfat_metrics)

        metric_row = self._build_metric_row(
            n_components=n_components,
            coronal_degree=coronal_degree,
            pancreas_fat_metrics=pancreas_fat_metrics,
            pancreas_ccfat_metrics=pancreas_ccfat_metrics,
            pancreatic_distance_metrics=pancreatic_distance_metrics,
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
            "vol_l3": ImageReader(mpath / "vertebrae_L3.nii.gz").read(),
            "vol_vfat": ImageReader(mpath / "visceral_fat.nii.gz").read(),
            "vol_sfat": ImageReader(mpath / "subcutaneous_fat.nii.gz").read(),
            "vol_pancreas": ImageReader(mpath / "pancreas.nii.gz").read(),
        }

    def _largest_component(self, vol_pancreas: sitk.Image) -> tuple[sitk.Image, int]:
        cc = sitk.RelabelComponent(sitk.ConnectedComponent(vol_pancreas))
        n_components = int(sitk.GetArrayFromImage(cc).max())
        if n_components == 0:
            raise ValueError("Pancreas mask is empty.")
        return cc == 1, n_components

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
        self._save_region_figure(ctx, vol_pancreas_region, partitioner)
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
        organ_vols = organ_vols.rename(columns={name: f"{name}_volume_cm3" for name in organ_list})

        tissue_slice_masks, tissue_slice_areas = compute_label_areas(
            tissuefiles,
            slices_index=optimal_idx,
            transform=tfm_axis,
            return_vols=True,
        )
        tissue_list = tissue_slice_areas.columns.tolist()
        tissue_slice_areas = tissue_slice_areas.rename(
            columns={name: f"{name}_l3_area_cm2" for name in tissue_list}
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
        abdominal_distance_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "N.Pancreas_Connected_Components": n_components,
            "coronal_plane_degree": coronal_degree,
            **pancreas_fat_metrics,
            **pancreas_ccfat_metrics,
            "PAAD_cm": pancreatic_distance_metrics["PAAD_cm"],
            "Max_SFT_cm": abdominal_distance_metrics["Max_SFT_cm"],
            "Min_SFT_cm": abdominal_distance_metrics["Min_SFT_cm"],
            "Mean_SFT_cm": abdominal_distance_metrics["Mean_SFT_cm"],
            "Median_SFT_cm": abdominal_distance_metrics["Median_SFT_cm"],
            "Left_SFT_cm": abdominal_distance_metrics["Left_SFT_cm"],
            "Right_SFT_cm": abdominal_distance_metrics["Right_SFT_cm"],
            "Anterior_SFT_cm": abdominal_distance_metrics["Anterior_SFT_cm"],
            "LRD_cm": abdominal_distance_metrics["LRD_cm"],
            "APD_cm": abdominal_distance_metrics["APD_cm"],
            "AP_cm": abdominal_distance_metrics["AP_cm"],
        }

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
    ) -> None:
        if not self.save_figures:
            return
        fname = f"{ctx['prefix']}_pancreatic_vfat_volumes.png"
        figure_overlay_pancreatic_craniocaudal_slices(
            volume=sup["vol"],
            volume_pancreas=sup["vol_pancreas"],
            volume_visceralfat=sup["vol_vfat"],
            volume_metrics=pancreas_ccfat_metrics,
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
