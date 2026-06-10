import numpy as np
import pytest
import SimpleITK as sitk

from medcore.pipeline import ct_feature_extractor
from medcore.pipeline.ct_feature_extractor import (
    CTFeatureExtractor,
    pancreatic_distance,
    pancreatic_morphology,
)


def _image(array: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def _abdominal_metrics() -> dict[str, float]:
    return {
        "Max_SFT_cm": 1.0,
        "Min_SFT_cm": 0.5,
        "Mean_SFT_cm": 0.75,
        "Median_SFT_cm": 0.7,
        "Left_SFT_cm": 0.6,
        "Right_SFT_cm": 0.8,
        "Anterior_SFT_cm": 0.9,
        "LRD_cm": 30.0,
        "APD_cm": 20.0,
        "AP_cm": 10.0,
    }


def test_pancreas_curvature_region_names_follow_head_body_tail_order() -> None:
    extractor = CTFeatureExtractor(save_figures=False)

    assert extractor._pancreas_curvature_region_names() == ("head", "body", "tail")


def test_save_figures_defaults_from_save_dir(tmp_path) -> None:
    extractor = CTFeatureExtractor()
    assert extractor.save_figures is False
    assert extractor.save_dir is None

    extractor_with_dir = CTFeatureExtractor(save_dir=tmp_path)
    assert extractor_with_dir.save_figures is True
    assert extractor_with_dir.save_dir == tmp_path

    extractor_disabled = CTFeatureExtractor(save_dir=tmp_path, save_figures=False)
    assert extractor_disabled.save_figures is False
    assert extractor_disabled.save_dir == tmp_path


def test_save_figures_true_requires_save_dir() -> None:
    with pytest.raises(ValueError, match="save_dir must be provided"):
        CTFeatureExtractor(save_figures=True)


def test_pancreatic_morphology_returns_notebook_style_keys() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1

    class FakePartitioner:
        region_curvature_indices = {
            "region_1": 1.1,
            "region_2": 1.2,
            "region_3": 1.3,
            "first": 1.4,
            "last": 1.5,
        }
        centerline_length = 42.0
        curvature_index = 1.6

    result = pancreatic_morphology(_image(mask), partitioner=FakePartitioner())

    assert result["curvature_ahead_pancreas"] == 1.1
    assert result["curvature_body_pancreas"] == 1.2
    assert result["curvature_tail_pancreas"] == 1.3
    assert result["curvature_first_pancreas"] == 1.4
    assert result["curvature_last_pancreas"] == 1.5
    assert result["curvature_pancreas"] == 1.6
    assert result["length(cm)_pancres"] == 4.2
    assert result["volume(cm3)_pancreas"] == pytest.approx(0.027)
    assert result["elongation_pancreas"] > 0
    assert result["flatness_pancreas"] > 0
    assert result["roundness_pancreas"] > 0


def test_pancreatic_distance_returns_l1_offsets() -> None:
    mask_l1 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_l1[2, 4:7, 3:6] = 1

    class FakePartitioner:
        smooth_centerline = np.array([[1.0, 2.0, 3.0], [8.0, 9.0, 7.0]])
        centerline_midpoint_voxel_zyx = np.array([5, 7, 6])

    result = pancreatic_distance(_image(mask_l1), partitioner=FakePartitioner())

    np.testing.assert_array_equal(result["center_of_mass_l1"], np.array([2, 5, 4]))
    np.testing.assert_array_equal(result["anterior_point_l1"], np.array([2, 4, 4]))
    np.testing.assert_array_equal(result["center_point_pancreas"], np.array([5, 7, 6]))
    np.testing.assert_array_equal(result["head_point_pancreas"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(result["tail_point_pancreas"], np.array([8, 9, 7]))
    assert result["IS_dist_center_from_l1(mm)"] == 3
    assert result["AP_dist_center_from_l1(mm)"] == 2
    assert result["RL_dist_center_from_l1(mm)"] == 2
    assert result["IS_dist_head_from_l1(mm)"] == -1
    assert result["AP_dist_head_from_l1(mm)"] == -3
    assert result["RL_dist_head_from_l1(mm)"] == -1
    assert result["IS_dist_tail_from_l1_anterior(mm)"] == 6
    assert result["AP_dist_tail_from_l1_anterior(mm)"] == 5
    assert result["RL_dist_tail_from_l1_anterior(mm)"] == 3


def test_craniocaudal_center_cutoff_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValueError, match="craniocaudal_center_cutoff"):
        CTFeatureExtractor(save_figures=False, craniocaudal_center_cutoff=1.0)


def test_craniocaudal_center_mask_uses_configured_centerline_cutoff(monkeypatch) -> None:
    calls = {}

    class FakePartitioner:
        def __init__(self, *, mask_volume, cutoffs, target_spacing_mm, smoothing):
            calls["mask_volume"] = mask_volume
            calls["cutoffs"] = cutoffs
            calls["target_spacing_mm"] = target_spacing_mm
            calls["smoothing"] = smoothing
            self.cutoff_voxels_zyx = None

        def run(self):
            self.cutoff_voxels_zyx = np.array([[4, 5, 6]])

    monkeypatch.setattr(ct_feature_extractor, "RegionCenterlinePartitioner", FakePartitioner)

    vol_pancreas = _image(np.ones((2, 2, 2), dtype=np.uint8))
    extractor = CTFeatureExtractor(
        save_figures=False,
        craniocaudal_center_cutoff=0.75,
    )

    center_mask = extractor._craniocaudal_center_mask(vol_pancreas)

    assert center_mask == (4.0, 5.0, 6.0)
    assert calls == {
        "mask_volume": vol_pancreas,
        "cutoffs": 0.75,
        "target_spacing_mm": 1.0,
        "smoothing": 15.0,
    }


def test_build_metric_row_groups_outputs_by_extraction_order() -> None:
    extractor = CTFeatureExtractor(save_figures=False)

    row = extractor._build_metric_row(
        n_components=1,
        coronal_degree=3.0,
        pancreas_fat_metrics={
            "total_shell_voxel_count": 10,
            "total_shell_fat_voxel_count": 20,
            "region_1_fat_voxel_count": 30,
            "region_2_fat_volume_cm3": 0.2,
            "region_3_fat_voxel_count": 50,
        },
        abdominal_distance_metrics=_abdominal_metrics(),
        pancreatic_distance_metrics={"PAAD_cm": 2.0},
        pancreatic_geometry_metrics={"curvature_ahead_pancreas": 1.5},
        pancreas_ccfat_metrics={
            "total_fat_voxel_count": 40,
            "total_fat_volume_cm3": 0.04,
            "superior_anterior_left_fat_voxel_count": 5,
        },
    )

    assert list(row) == [
        "N.Pancreas_Connected_Components",
        "total_shell_voxel_count",
        "total_shell_fat_voxel_count",
        "pancreas_head_fat_voxel_count",
        "pancreas_body_fat_volume_cm3",
        "pancreas_tail_fat_voxel_count",
        "coronal_plane_degree",
        "Max_SFT_cm",
        "Min_SFT_cm",
        "Mean_SFT_cm",
        "Median_SFT_cm",
        "Left_SFT_cm",
        "Right_SFT_cm",
        "Anterior_SFT_cm",
        "LRD_cm",
        "APD_cm",
        "AP_cm",
        "PAAD_cm",
        "curvature_ahead_pancreas",
        "craniocaudal_total_fat_voxel_count",
        "craniocaudal_total_fat_volume_cm3",
        "superior_anterior_left_fat_voxel_count",
    ]
    assert row["total_shell_fat_voxel_count"] == 20
    assert row["pancreas_head_fat_voxel_count"] == 30
    assert row["pancreas_body_fat_volume_cm3"] == 0.2
    assert row["pancreas_tail_fat_voxel_count"] == 50
    assert "region_1_fat_voxel_count" not in row
    assert "region_2_fat_volume_cm3" not in row
    assert row["craniocaudal_total_fat_voxel_count"] == 40


def test_run_passes_configured_craniocaudal_center_mask(monkeypatch) -> None:
    extractor = CTFeatureExtractor(
        save_figures=False,
        craniocaudal_center_cutoff=0.75,
    )

    shape = (3, 8, 8)
    vol = _image(np.zeros(shape, dtype=np.int16))
    pancreas = np.zeros(shape, dtype=np.uint8)
    pancreas[1, 3:5, 3:5] = 1
    vol_pancreas = _image(pancreas)
    mask = _image(np.ones(shape, dtype=np.uint8))
    l3 = _image(np.ones(shape, dtype=np.uint8))

    volumes = {
        "vol": vol,
        "vol_l3": l3,
        "vol_vfat": mask,
        "vol_sfat": mask,
        "vol_pancreas": vol_pancreas,
    }
    sup = {
        "vol": vol,
        "vol_torso": mask,
        "vol_sfat": mask,
        "vol_vfat": mask,
        "vol_pancreas": vol_pancreas,
        "vol_l3": l3,
    }
    captured = {}

    def capture_craniocaudal_fat_volume(**kwargs):
        captured.update(kwargs)
        return {
            "total_fat_voxel_count": 1,
            "total_fat_volume_cm3": 0.001,
        }

    monkeypatch.setattr(extractor, "_load_volumes", lambda ctx: volumes)
    monkeypatch.setattr(extractor, "_make_supine_transform", lambda vol: (sitk.Transform(), 0.0))
    monkeypatch.setattr(extractor, "_resample_to_supine", lambda volumes, vol_pancreas, tfm: sup)
    monkeypatch.setattr(extractor, "_extract_peripancreatic_metrics", lambda **kwargs: {})
    monkeypatch.setattr(
        extractor, "_craniocaudal_center_mask", lambda vol_pancreas: (1.0, 2.0, 3.0)
    )
    monkeypatch.setattr(
        extractor,
        "_extract_pancreatic_geometry_metrics",
        lambda **kwargs: {"curvature_ahead_pancreas": 1.0},
    )
    monkeypatch.setattr(extractor, "_extract_organ_tissue_metrics", lambda **kwargs: {})
    monkeypatch.setattr(
        ct_feature_extractor, "extract_optimal_transverse_process_slice", lambda _: 0
    )
    monkeypatch.setattr(
        ct_feature_extractor,
        "extract_abdominal_distance_metrics",
        lambda **kwargs: _abdominal_metrics(),
    )
    monkeypatch.setattr(
        ct_feature_extractor,
        "extract_pancreatic_distance_metrics",
        lambda **kwargs: {"PAAD_cm": 2.0},
    )
    monkeypatch.setattr(
        ct_feature_extractor,
        "extract_craniocaudal_fat_volume",
        capture_craniocaudal_fat_volume,
    )

    extractor.run(
        "ct.nii.gz",
        "mask",
        metadata={"hutom_id": "H", "study_uid": "S", "series_uid": "R"},
    )

    assert captured["center_mask"] == (1.0, 2.0, 3.0)


def test_run_returns_none_for_pancreas_features_when_pancreas_mask_is_empty(
    monkeypatch,
) -> None:
    extractor = CTFeatureExtractor(save_figures=False)

    shape = (3, 8, 8)
    vol = _image(np.zeros(shape, dtype=np.int16))
    empty_pancreas = _image(np.zeros(shape, dtype=np.uint8))
    mask = _image(np.ones(shape, dtype=np.uint8))
    l3 = _image(np.ones(shape, dtype=np.uint8))

    volumes = {
        "vol": vol,
        "vol_l3": l3,
        "vol_vfat": mask,
        "vol_sfat": mask,
        "vol_pancreas": empty_pancreas,
    }
    sup = {
        "vol": vol,
        "vol_torso": mask,
        "vol_sfat": mask,
        "vol_vfat": mask,
        "vol_pancreas": empty_pancreas,
        "vol_l3": l3,
    }

    def fail_pancreas_extractor(*args, **kwargs):
        raise AssertionError("pancreas-specific extractor should be skipped")

    monkeypatch.setattr(extractor, "_load_volumes", lambda ctx: volumes)
    monkeypatch.setattr(extractor, "_make_supine_transform", lambda vol: (sitk.Transform(), 0.0))
    monkeypatch.setattr(extractor, "_resample_to_supine", lambda volumes, vol_pancreas, tfm: sup)
    monkeypatch.setattr(
        extractor,
        "_extract_organ_tissue_metrics",
        lambda **kwargs: {"pancreas_volume_cm3": 12.3, "liver_volume_cm3": 4.5},
    )
    monkeypatch.setattr(extractor, "_extract_peripancreatic_metrics", fail_pancreas_extractor)
    monkeypatch.setattr(
        ct_feature_extractor, "extract_optimal_transverse_process_slice", lambda _: 0
    )
    monkeypatch.setattr(
        ct_feature_extractor,
        "extract_abdominal_distance_metrics",
        lambda **kwargs: _abdominal_metrics(),
    )
    monkeypatch.setattr(
        ct_feature_extractor,
        "extract_pancreatic_distance_metrics",
        fail_pancreas_extractor,
    )
    monkeypatch.setattr(
        ct_feature_extractor,
        "extract_craniocaudal_fat_volume",
        fail_pancreas_extractor,
    )

    result = extractor.run(
        "ct.nii.gz",
        "mask",
        metadata={"hutom_id": "H", "study_uid": "S", "series_uid": "R"},
    )

    assert result["N.Pancreas_Connected_Components"] is None
    assert result["PAAD_cm"] is None
    assert result["total_shell_voxel_count"] is None
    assert result["total_shell_fat_voxel_count"] is None
    assert result["craniocaudal_total_fat_voxel_count"] is None
    assert result["craniocaudal_total_fat_volume_cm3"] is None
    assert "superior_fat_voxel_count" not in result
    assert "superior_fat_volume_cm3" not in result
    assert "anterior_fat_voxel_count" not in result
    assert "anterior_fat_volume_cm3" not in result
    assert result["pancreas_tail_fat_volume_cm3"] is None
    assert "region_1_fat_voxel_count" not in result
    assert "region_2_fat_voxel_count" not in result
    assert "region_3_fat_volume_cm3" not in result
    assert result["curvature_index"] is None
    assert result["head_curvature_index"] is None
    assert result["body_curvature_index"] is None
    assert result["tail_curvature_index"] is None
    assert result["first_curvature_index"] is None
    assert result["last_curvature_index"] is None
    assert result["curvature_ahead_pancreas"] is None
    assert result["elongation_pancreas"] is None
    assert result["length(cm)_pancres"] is None
    assert result["center_of_mass_l1"] is None
    assert result["IS_dist_center_from_l1(mm)"] is None
    assert result["superior_anterior_left_fat_voxel_count"] is None
    assert result["pancreas_volume_cm3"] is None
    assert result["liver_volume_cm3"] == 4.5
    assert result["LRD_cm"] == 30.0
