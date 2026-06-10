import numpy as np
import pytest
import SimpleITK as sitk

from medcore.utils.sitk_utils import sitk_get_shape_features, sitk_resample_point_between_volumes


def _image(size=(6, 7, 8)) -> sitk.Image:
    return sitk.Image(size, sitk.sitkUInt8)


def test_get_shape_features_uses_physical_voxel_size() -> None:
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[1:4, 2:5, 3:6] = 2

    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((0.5, 2.0, 3.0))

    features = sitk_get_shape_features(image, label=2)

    assert features["volume_mm3"] == pytest.approx(81.0)
    assert features["volume_ml"] == pytest.approx(0.081)
    assert features["elongation"] > 0
    assert features["flatness"] > 0
    assert features["roundness"] > 0


def test_get_shape_features_raises_for_missing_label() -> None:
    image = _image()

    with pytest.raises(ValueError, match="Label 1 not found in mask"):
        sitk_get_shape_features(image)


def test_resample_point_returns_inside_point_with_mask_path() -> None:
    image = _image()
    transform = sitk.Transform(3, sitk.sitkIdentity)

    assert sitk_resample_point_between_volumes([3, 4, 3], image, image, transform) == [
        3,
        4,
        3,
    ]


def test_resample_point_maps_outside_source_point_by_physical_geometry() -> None:
    image = _image()
    transform = sitk.TranslationTransform(3, (2.0, 0.0, 0.0))

    assert sitk_resample_point_between_volumes([1, 2, -3], image, image, transform) == [
        1,
        2,
        -1,
    ]


def test_resample_point_treats_negative_float_as_outside_source() -> None:
    image = _image()
    transform = sitk.Transform(3, sitk.sitkIdentity)

    assert sitk_resample_point_between_volumes([-0.6, 2, 4], image, image, transform) == [
        -1,
        2,
        4,
    ]


def test_resample_point_can_clip_outside_result_to_target_grid() -> None:
    image = _image()
    transform = sitk.Transform(3, sitk.sitkIdentity)

    assert sitk_resample_point_between_volumes(
        [-2, 2, 4],
        image,
        image,
        transform,
        clip_output=True,
    ) == [0, 2, 4]


def test_resample_point_can_keep_previous_raise_behavior() -> None:
    image = _image()
    transform = sitk.Transform(3, sitk.sitkIdentity)

    with pytest.raises(ValueError, match="outside source_volume grid"):
        sitk_resample_point_between_volumes(
            [-2, 2, 4],
            image,
            image,
            transform,
            allow_outside=False,
        )
