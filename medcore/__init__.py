import pandas as pd

from ._version import __version__
from .detect import (
    LandmarkMaskGenerator,
    UmbilicusDetector,
    UmbilicusPredictor,
    get_coronal_plane_degree,
    get_longest_segment,
    get_median_slice_index,
)
from .feature import (
    compute_label_areas,
    compute_label_volumes,
    compute_label_volumns,
    extract_abdominal_body_composition_metrics,
    extract_patches_from_image,
)
from .io import ImageReader, convert_dicom_to_nifti
from .segment import (
    AbdomenSegmenter,
    TorsoSegmenter,
)
from .utils import (
    figure_overlay_label_on_slices,
    figure_overlay_label_reference_slice,
    figure_overlay_tissue_on_slices,
    figure_patch_from_image,
    figure_slices_with_landmarks,
    figure_slices_with_umbilicus,
    sitk_copy_metainfo,
    sitk_get_array,
    sitk_make_euler3dtransform,
    sitk_read_labelfiles,
    sitk_resample_point_between_volumes,
    sitk_resampler,
    sitk_write_nii,
)

pd.set_option("display.float_format", "{:.4f}".format)

__all__ = [
    "__version__",
    "ImageReader",
    "convert_dicom_to_nifti",
    "sitk_write_nii",
    "sitk_get_array",
    "sitk_make_euler3dtransform",
    "sitk_resampler",
    "sitk_resample_point_between_volumes",
    "sitk_read_labelfiles",
    "sitk_copy_metainfo",
    "get_median_slice_index",
    "get_longest_segment",
    "get_coronal_plane_degree",
    "UmbilicusPredictor",
    "UmbilicusDetector",
    "LandmarkMaskGenerator",
    "TorsoSegmenter",
    "AbdomenSegmenter",
    "compute_label_volumes",
    "compute_label_volumns",
    "compute_label_areas",
    "extract_abdominal_body_composition_metrics",
    "extract_patches_from_image",
    "figure_overlay_tissue_on_slices",
    "figure_overlay_label_on_slices",
    "figure_overlay_label_reference_slice",
    "figure_patch_from_image",
    "figure_slices_with_umbilicus",
    "figure_slices_with_landmarks",
]
