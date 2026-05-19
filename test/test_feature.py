import numpy as np
import pytest
import SimpleITK as sitk

from medcore.feature import extract_peripancreatic_fat_volume


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


def test_extract_peripancreatic_fat_volume_uses_lps_anterior_direction() -> None:
    result = extract_peripancreatic_fat_volume(*_fat_inputs())

    assert result["total_shell_voxel_count"] == 45
    assert result["total_shell_volume_ml"] == pytest.approx(1.08)
    assert result["total_fat_voxel_count"] == 45
    assert result["superior_fat_voxel_count"] == 30
    assert result["anterior_fat_voxel_count"] == 18
    assert result["region_1_fat_voxel_count"] == 9
    assert result["region_2_fat_voxel_count"] == 9


def test_extract_peripancreatic_fat_volume_uses_ras_anterior_direction() -> None:
    ras_direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

    result = extract_peripancreatic_fat_volume(*_fat_inputs(ras_direction))

    assert result["superior_fat_voxel_count"] == 30
    assert result["anterior_fat_voxel_count"] == 36


def test_extract_peripancreatic_fat_volume_rejects_unsupported_orientation() -> None:
    rip_direction = (-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0)

    with pytest.raises(ValueError, match="orientation must be LPS or RAS"):
        extract_peripancreatic_fat_volume(*_fat_inputs(rip_direction))
