import numpy as np
import pytest
import SimpleITK as sitk

from medcore.feature import (
    extract_abdominal_distance_metrics,
    extract_craniocaudal_fat_volume,
    extract_peripancreatic_fat_volume,
)


def _image(array: np.ndarray, direction: tuple[float, ...] | None = None) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((2.0, 3.0, 4.0))
    if direction is not None:
        image.SetDirection(direction)
    return image


def _fat_inputs(direction: tuple[float, ...] | None = None) -> tuple[sitk.Image, ...]:
    volume = np.full((3, 5, 3), -100, dtype=np.int16)
    mask = np.zeros_like(volume, dtype=np.uint8)
    mask[1, 1, 1] = 1
    shell = np.ones_like(volume, dtype=np.uint8)
    shell_region = np.zeros_like(volume, dtype=np.uint8)
    shell_region[:, 0, :] = 1
    shell_region[:, 4, :] = 2

    return (
        _image(volume, direction),
        _image(mask, direction),
        _image(shell, direction),
        _image(shell_region, direction),
    )


def test_extract_peripancreatic_fat_volume_returns_total_and_region_metrics() -> None:
    result = extract_peripancreatic_fat_volume(*_fat_inputs())

    assert result["total_shell_voxel_count"] == 45
    assert result["total_shell_volume_cm3"] == pytest.approx(1.08)
    assert result["total_shell_fat_voxel_count"] == 45
    assert result["total_shell_fat_volume_cm3"] == pytest.approx(1.08)
    assert result["region_1_fat_voxel_count"] == 9
    assert result["region_2_fat_voxel_count"] == 9
    assert "superior_fat_voxel_count" not in result
    assert "superior_fat_volume_cm3" not in result
    assert "anterior_fat_voxel_count" not in result
    assert "anterior_fat_volume_cm3" not in result


def test_extract_peripancreatic_fat_volume_accepts_non_directional_metrics_for_ras() -> None:
    ras_direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

    result = extract_peripancreatic_fat_volume(*_fat_inputs(ras_direction))

    assert result["total_shell_voxel_count"] == 45
    assert "superior_fat_voxel_count" not in result
    assert "anterior_fat_voxel_count" not in result


def test_extract_peripancreatic_fat_volume_accepts_unsupported_orientation() -> None:
    rip_direction = (-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0)

    result = extract_peripancreatic_fat_volume(*_fat_inputs(rip_direction))

    assert result["total_shell_voxel_count"] == 45


def _abdominal_distance_inputs(sfat_slice: np.ndarray) -> tuple[sitk.Image, ...]:
    volume = np.zeros((1, 64, 64), dtype=np.int16)
    torso = np.zeros_like(volume, dtype=np.uint8)
    l3 = np.zeros_like(volume, dtype=np.uint8)

    torso[0, 10:55, 5:60] = 1
    l3[0, 30:36, 28:38] = 1

    sfat = np.zeros_like(volume, dtype=np.uint8)
    sfat[0] = sfat_slice

    return _image(volume), _image(torso), _image(sfat), _image(l3)


def test_extract_abdominal_distance_metrics_returns_zero_sft_without_contour() -> None:
    sfat_slice = np.zeros((64, 64), dtype=np.uint8)
    result = extract_abdominal_distance_metrics(
        *_abdominal_distance_inputs(sfat_slice),
        l3_index=0,
        min_l3_component_size=1,
    )

    assert result["Max_SFT_cm"] == 0.0
    assert result["Min_SFT_cm"] == 0.0
    assert result["Left_SFT_cm"] == 0.0
    assert result["Right_SFT_cm"] == 0.0
    assert result["Anterior_SFT_cm"] == 0.0
    assert result["Mean_SFT_cm"] == 0.0
    assert result["Median_SFT_cm"] == 0.0
    assert result["LRD_cm"] > 0.0
    assert result["APD_cm"] > 0.0
    assert result["SFT_LR_margin"] is None
    assert result["n_anterior_contour_points"] == 0
    assert result["n_positive_sft_points"] == 0


def test_extract_abdominal_distance_metrics_returns_zero_sft_without_anterior_contour() -> None:
    sfat_slice = np.zeros((64, 64), dtype=np.uint8)
    sfat_slice[42:53, 20:45] = 1

    result = extract_abdominal_distance_metrics(
        *_abdominal_distance_inputs(sfat_slice),
        l3_index=0,
        min_l3_component_size=1,
    )

    assert result["Max_SFT_cm"] == 0.0
    assert result["Min_SFT_cm"] == 0.0
    assert result["Left_SFT_cm"] == 0.0
    assert result["Right_SFT_cm"] == 0.0
    assert result["Anterior_SFT_cm"] == 0.0
    assert result["Mean_SFT_cm"] == 0.0
    assert result["Median_SFT_cm"] == 0.0
    assert result["LRD_cm"] > 0.0
    assert result["APD_cm"] > 0.0
    assert result["SFT_LR_margin"] is None
    assert result["n_anterior_contour_points"] == 0
    assert result["n_positive_sft_points"] == 0


def _craniocaudal_fat_inputs() -> tuple[sitk.Image, ...]:
    volume = np.full((4, 4, 4), -100, dtype=np.int16)
    mask = np.zeros_like(volume, dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    shell = np.ones_like(volume, dtype=np.uint8)

    return _image(volume), _image(mask), _image(shell)


def test_extract_craniocaudal_fat_volume_uses_mask_center_by_default() -> None:
    result = extract_craniocaudal_fat_volume(*_craniocaudal_fat_inputs())

    assert result["total_fat_voxel_count"] == 64
    assert result["superior_anterior_left_fat_voxel_count"] == 8
    assert result["inferior_posterior_right_fat_voxel_count"] == 8


def test_extract_craniocaudal_fat_volume_uses_input_center_mask() -> None:
    result = extract_craniocaudal_fat_volume(
        *_craniocaudal_fat_inputs(),
        center_mask=(0.5, 0.5, 0.5),
    )

    assert result["total_fat_voxel_count"] == 64
    assert result["superior_anterior_left_fat_voxel_count"] == 9
    assert result["inferior_posterior_right_fat_voxel_count"] == 3


def test_extract_craniocaudal_fat_volume_rejects_invalid_center_mask() -> None:
    with pytest.raises(ValueError, match="center_mask"):
        extract_craniocaudal_fat_volume(
            *_craniocaudal_fat_inputs(),
            center_mask=(0.5, 0.5),
        )
