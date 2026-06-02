import numpy as np
import SimpleITK as sitk

from medcore.pipeline import ct_feature_extractor
from medcore.pipeline.ct_feature_extractor import CTFeatureExtractor


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
    monkeypatch.setattr(ct_feature_extractor, "extract_optimal_transverse_process_slice", lambda _: 0)
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
    assert result["region_3_fat_volume_cm3"] is None
    assert result["superior_anterior_left_fat_voxel_count"] is None
    assert result["pancreas_volume_cm3"] is None
    assert result["liver_volume_cm3"] == 4.5
    assert result["LRD_cm"] == 30.0
