import pytest
import SimpleITK as sitk

from medcore.utils.sitk_utils import sitk_resample_point_between_volumes


def _image(size=(6, 7, 8)) -> sitk.Image:
    return sitk.Image(size, sitk.sitkUInt8)


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
