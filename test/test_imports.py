from medcore.detect import UmbilicusDetector, UmbilicusPredictor
from medcore.feature import (
    compute_label_areas,
    compute_label_volumes,
    extract_patches_from_image,
    extract_peripancreatic_fat_volume,
)
from medcore.io import ImageReader, convert_dicom_to_nifti
from medcore.pipeline import CTFeatureExtractor
from medcore.segment import AbdomenSegmenter, RegionCenterlinePartitioner, TorsoSegmenter
from medcore.utils import (
    figure_overlay_label_on_slices,
    sitk_create_shell_mask,
    sitk_get_array,
    sitk_make_euler3dtransform,
    sitk_read_labelfiles,
    sitk_resampler,
    sitk_write_nii,
)


def test_public_imports_are_available() -> None:
    assert ImageReader is not None
    assert convert_dicom_to_nifti is not None
    assert CTFeatureExtractor is not None
    assert sitk_get_array is not None
    assert sitk_make_euler3dtransform is not None
    assert sitk_read_labelfiles is not None
    assert sitk_resampler is not None
    assert sitk_write_nii is not None
    assert sitk_create_shell_mask is not None
    assert figure_overlay_label_on_slices is not None
    assert TorsoSegmenter is not None
    assert AbdomenSegmenter is not None
    assert RegionCenterlinePartitioner is not None
    assert UmbilicusPredictor is not None
    assert UmbilicusDetector is not None
    assert compute_label_volumes is not None
    assert compute_label_areas is not None
    assert extract_patches_from_image is not None
    assert extract_peripancreatic_fat_volume is not None
