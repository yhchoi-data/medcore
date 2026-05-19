# Usage

This document describes common usage patterns for **medcore**.

## 1. Imports

```python
from medcore.io import ImageReader

from medcore.utils import (
    sitk_get_array,
    sitk_write_nii,
    sitk_make_euler3dtransform,
    sitk_resampler,
    sitk_read_labelfiles,
)

from medcore.detect import UmbilicusPredictor, UmbilicusDetector
from medcore.segment import TorsoSegmenter, AbdomenSegmenter
from medcore.feature import (
    compute_label_volumes,
    compute_label_areas,
    extract_patches_from_image,
)
```

## 2. Load image (DICOM Series / NIfTI)

```python
from medcore.io import ImageReader

vol = ImageReader("/path/to/image.nii.gz").read()
print(vol.GetSize(), vol.GetSpacing())
```

```python
# To set the orientation, use check_coord_flag
vol = ImageReader(
	"/path/to/dicom_dir",
	check_coord_flag=True,
	target_orientation='LPS'
).read()
```

## 3. Convert to NumPy and normalize

```python
from medcore.utils import sitk_get_array

arr = sitk_get_array(vol)  # raw array
arr_norm = sitk_get_array(vol, normalize=True, norm_min=-500, norm_max=2000)
```

## 4. Resampling and transforms

```python
from medcore.utils import sitk_make_euler3dtransform, sitk_resampler

# apply to transform matrix
tfm = sitk_make_euler3dtransform(vol, rotation_deg=15, axis="x")
vol_rot = sitk_resampler(vol, transform=tfm, interpolation="linear")
# resample iso-voxel space
vol_iso = sitk_resampler(vol, new_spacing=(1.0, 1.0, 1.0))
```

## 5. Save NIfTI

```python
from medcore.utils import sitk_write_nii, sitk_get_array

img = sitk_get_array(vol)
# ... processing ...
sitk_write_nii(vol, "/path/to/out_volume.nii.gz")
sitk_write_nii(img, "/path/to/out_array.nii.gz", reference=vol)
```

## 6. Segmentation

```python
from medcore.segment import TorsoSegmenter, AbdomenSegmenter

# Torso (Skin) Segmentation
torso_seg = TorsoSegmenter()
torso_vol =torso_seg.segment(vol)

# Abdominal region Segmentation
abd_seg = AbdomenSegmenter()
abdomen_image, abdomen_mask, abdomen_region, contour_list = abd_seg.segment(vol)
```

## 7. Detection

```python
from medcore.detect import UmbilicusPredictor

predictor = UmbilicusPredictor()
point_xyz = predictor.predict(vol)
print(point_xyz)
```

```python
from medcore.detect import UmbilicusDetector

detector = UmbilicusDetector()
points_df = detector.detect(
    region_image=abdominal_image,
    region_mask=abdomen_mask,
    region_contour=contour_list,
    region_info=abdomen_region,
)
print(points_df.head())
```

## 8. Feature extraction

```python
from medcore.feature import compute_label_volumes, compute_label_areas

labelfiles = {
    1: "/path/to/muscle.nii.gz",
    2: "/path/to/fat.nii.gz",
}

volumes_cm3 = compute_label_volumes(labelfiles)
areas_cm2 = compute_label_areas(labelfiles, slices_index=100)
print(volumes_cm3)
print(areas_cm2)
```

```python
from medcore.feature import extract_patches_from_image

# points: (N, 3), commonly 25 points for 5x5 grid
patches = extract_patches_from_image(points, vol, patch_size=50, middle_size=50, delta=25)
print(patches.shape)
```

## 9. Label merge utility

```python
from medcore.utils import sitk_read_labelfiles

merged = sitk_read_labelfiles(labelfiles)  # labels merged into one UInt8 volume
```

## 10. Convert dicom to nifti

```python
from medcore.utils import sitk_read_labelfiles

dcm_dir = "/path/to/dicom_dir"
out_dir = "/path/to/out_dir"
convert_dicom_to_nifti(dcm_dir, out_dir)
```

## 11. Pipeline: CT Feature Extractor

```python
from medcore.pipeline import CTFeatureExtractor

save_dir = '/path/to/save_dir_figure'
fpath = '/path/to/ct.nii.gz'
mpath = '/path/to/mask_folder/'
hutom_id = 'HUTOM0000'
study_uid = None
series_uid = None
```

`mpath` must contain mask files with the exact filenames below:

```text
vertebrae_L3.nii.gz
visceral_fat.nii.gz
subcutaneous_fat.nii.gz
muscle.nii.gz
iliopsoas_right.nii.gz
iliopsoas_left.nii.gz
pancreas.nii.gz
gallbladder.nii.gz
liver.nii.gz
spleen.nii.gz
stomach.nii.gz
```

`pancreas.nii.gz` is used for both pancreas-specific metrics and organ volume metrics.

```python
extractor = CTFeatureExtractor(
    save_dir=save_dir,
)

metric = extractor.run(
    fpath,
    mpath,
    metadata={
        "hutom_id": hutom_id,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "fpath": fpath,
    },
)

print(metric)
```

## 12. Notes

- `medcore.segment` import is supported directly:
  - `from medcore.segment import TorsoSegmenter`
- `compute_label_volumns` is still available as a backward-compatible alias of `compute_label_volumes`.
