import numpy as np
import pytest
import SimpleITK as sitk

from medcore.segment import RegionCenterlinePartitioner
from medcore.utils import sitk_create_shell_mask


def test_sitk_create_shell_mask_uses_physical_distance() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((1.0, 1.0, 1.0))

    shell = sitk_create_shell_mask(image, inner_mm=0.5, outer_mm=1.5)
    shell_arr = sitk.GetArrayFromImage(shell)

    assert shell_arr[2, 2, 2] == 0
    assert shell_arr.sum() == 18
    assert shell.GetSpacing() == image.GetSpacing()
    assert shell.GetPixelID() == sitk.sitkUInt8


def test_region_centerline_partitioner_labels_straight_region() -> None:
    mask = np.ones((7, 1, 1), dtype=np.uint8)
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((1.0, 1.0, 1.0))

    partitioner = RegionCenterlinePartitioner(
        image,
        cutoffs=(0.3, 0.7),
        target_spacing_mm=1.0,
        smoothing=0.0,
    )
    region = partitioner.run()
    region_arr = sitk.GetArrayFromImage(region)[:, 0, 0]

    np.testing.assert_array_equal(region_arr, np.array([1, 1, 2, 2, 2, 3, 3]))
    assert partitioner.centerline_length == 6.0
    assert partitioner.curvature_index == pytest.approx(1.0)
    assert partitioner.region_curvature_indices == {
        "region_1": pytest.approx(1.0),
        "region_2": pytest.approx(1.0),
        "region_3": pytest.approx(1.0),
        "first": pytest.approx(1.0),
        "last": pytest.approx(1.0),
    }
    assert partitioner.first_curvature_index == pytest.approx(1.0)
    assert partitioner.last_curvature_index == pytest.approx(1.0)
    np.testing.assert_allclose(partitioner.centerline_midpoint_zyx, np.array([3.0, 0.0, 0.0]))
    np.testing.assert_array_equal(partitioner.centerline_midpoint_voxel_zyx, np.array([3, 0, 0]))
    np.testing.assert_allclose(
        partitioner.cutoff_points_zyx,
        np.array([[1.8, 0.0, 0.0], [4.2, 0.0, 0.0]]),
    )
    np.testing.assert_array_equal(
        partitioner.cutoff_voxels_zyx,
        np.array([[2, 0, 0], [4, 0, 0]]),
    )
    assert region.GetSpacing() == image.GetSpacing()
    assert partitioner.shell_volume is None


def test_region_centerline_partitioner_maps_iso_skeleton_to_anisotropic_original_grid() -> None:
    mask = np.ones((4, 1, 1), dtype=np.uint8)
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((1.0, 1.0, 2.0))

    partitioner = RegionCenterlinePartitioner(
        image,
        cutoffs=0.5,
        target_spacing_mm=1.0,
        smoothing=0.0,
    )
    region = partitioner.run()

    assert partitioner.skeleton is not None
    assert partitioner.skeleton.shape != mask.shape
    assert partitioner.raw_centerline is not None
    assert partitioner.raw_centerline_skeleton is not None
    assert partitioner.centerline_length == pytest.approx(6.0)
    assert partitioner.raw_centerline[:, 0].min() == pytest.approx(0.0)
    assert partitioner.raw_centerline[:, 0].max() == pytest.approx(3.0)
    np.testing.assert_allclose(partitioner.centerline_midpoint_zyx, np.array([1.5, 0.0, 0.0]))
    np.testing.assert_array_equal(partitioner.centerline_midpoint_voxel_zyx, np.array([2, 0, 0]))
    np.testing.assert_allclose(partitioner.cutoff_points_zyx, np.array([[1.5, 0.0, 0.0]]))
    np.testing.assert_array_equal(partitioner.cutoff_voxels_zyx, np.array([[2, 0, 0]]))
    assert sitk.GetArrayFromImage(region).shape == mask.shape


def test_region_centerline_partitioner_computes_curvature_indices_by_cutoff() -> None:
    mask = np.ones((3, 3, 1), dtype=np.uint8)
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((1.0, 1.0, 1.0))

    partitioner = RegionCenterlinePartitioner(
        image,
        cutoffs=(0.25, 0.75),
    )
    centerline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    metrics = partitioner.compute_curvature_indices(
        centerline,
        region_names=("tail", "body", "head"),
    )

    assert metrics["curvature_index"] == pytest.approx(np.sqrt(2.0))
    assert metrics["tail_curvature_index"] == pytest.approx(1.0)
    assert metrics["body_curvature_index"] == pytest.approx(np.sqrt(2.0))
    assert metrics["head_curvature_index"] == pytest.approx(1.0)
    assert metrics["first_curvature_index"] == pytest.approx(1.0)
    assert metrics["last_curvature_index"] == pytest.approx(1.0)
    assert partitioner.curvature_index == pytest.approx(metrics["curvature_index"])
    assert partitioner.first_curvature_index == pytest.approx(metrics["first_curvature_index"])
    assert partitioner.last_curvature_index == pytest.approx(metrics["last_curvature_index"])
    np.testing.assert_allclose(partitioner.centerline_midpoint_zyx, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(partitioner.centerline_midpoint_voxel_zyx, np.array([1, 0, 0]))
    assert partitioner.region_curvature_indices == {
        "tail": pytest.approx(1.0),
        "body": pytest.approx(np.sqrt(2.0)),
        "head": pytest.approx(1.0),
        "first": pytest.approx(1.0),
        "last": pytest.approx(1.0),
    }


def test_region_centerline_partitioner_creates_shell_masks_explicitly() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 1
    image = sitk.GetImageFromArray(mask)
    image.SetSpacing((1.0, 1.0, 1.0))

    partitioner = RegionCenterlinePartitioner(
        image,
        shell_inner_mm=0.5,
        shell_outer_mm=1.5,
    )

    default_shell = sitk.GetArrayFromImage(partitioner.create_shell_mask())
    wider_shell = sitk.GetArrayFromImage(partitioner.create_shell_mask(inner_mm=1.5, outer_mm=2.5))

    assert default_shell.sum() == 18
    assert wider_shell.sum() == 62
    assert partitioner.shell_volume is not None
