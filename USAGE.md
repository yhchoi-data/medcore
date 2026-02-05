# Usage

This document describes how to use **h-reader**, a lightweight medical image reader built on top of **SimpleITK** for DICOM and NIfTI workflows.

---

## 1. Basic concepts

- All images are handled as `SimpleITK.Image`
- Both **DICOM series** and **NIfTI files** are supported
- Physical space consistency (origin / spacing / direction) is preserved
- Orientation can be standardized to **LPS** or **RAS**

---

## 2. Loading images

### Load a DICOM series

```python
from h_sitk_reader import ImageReader

vol = ImageReader(
    "/path/to/dicom_dir",
).read()

print(vol.GetSize())
```

- Input must be a directory containing a single / multiple DICOM series
- Slices are automatically sorted

---

### Load a NIfTI file

```python
from h_sitk_reader import ImageReader

vol = ImageReader(
    "/path/to/image.nii.gz",
).read()

print(vol.GetSize())
```

Supported formats:
- `.nii`
- `.nii.gz`

---

## 3. Orientation standardization

Medical images often come with inconsistent orientation conventions.  
`h-reader` allows explicit conversion.

### Convert to RAS (default: LPS)

```python
vol_ras = ImageReader(
    "/path/to/image.nii.gz",
    target_orientation="RAS"
).read()
```

Notes:
- Orientation conversion updates **direction**, **origin**, and **spacing** consistently
- Internally relies on SimpleITK physical space conventions

---

## 4. Get Array From Image

Converts a `SimpleITK.Image` into a NumPy array.  
Optionally, CT image intensities (HU) can be normalized to the `[0, 1]` range.

```python
from h_sitk_reader import ImageReader, sitk_get_array

vol = ImageReader(
    "/path/to/image.nii.gz",
).read()
img = sitk_get_array(
    vol
)
img_norm = sitk_get_array(
    vol,
    normalize = True,
    norm_min = -500,
    norm_max = 2000
)
print(img.shape, img.min(), img.max())
print(img_norm.shape, img_norm.min(), img_norm.max())
```

---

## 5. Create Euler3D transform (physical space)

Creates a Euler3DTransform centered at the **physical center of the reference image.**

```python
from h_sitk_reader import ImageReader, sitk_make_euler3dtransform

vol = ImageReader(
    "/path/to/image.nii.gz",
).read()
tfm = sitk_make_euler3dtransform(
    vol, 
    rotation_deg = 15, 
    axis='x'
)
```
---

## 5. Resampler

Images can be resampled to a target spacing. 
Or Arbitrary SimpleITK transforms can be applied.

```python
from h_reader import ImageReader, sitk_make_euler3dtransform, sitk_resampler

vol = ImageReader(
    "/path/to/image.nii.gz",
).read()
tfm = sitk_make_euler3dtransform(
    vol, 
    rotation_deg = 15, 
    axis='x'
)

vol_rsl_tfm = sitk_resampler(
    vol, 
    transform = tfm
)

vol_rsl = sitk_resampler(
    img,
    new_spacing=(1.0, 1.0, 1.0),
    interpolation="linear"
)

```

Available interpolators:
- `"nearest"`
- `"linear"`
- `"bspline"`

---

## 6. Convert Dicom to NIfTI

Dicoms can be converted to NIfTI file. 

```python
from h_sitk_reader import ImageReader

reader = ImageReader(
    "/path/to/dicom_dir",
)
reader.to_nifti("/path/to/dicom_dir/nifti.nii.gz")

```
- Output preserves physical metadata
- Suitable for downstream ML pipelines

---

## 7. Save NIfTI file
Images (SimpleITK or nd.array) can be saved as a NIfTI file.

```python
from h_sitk_reader import ImageReader, sitk_write_nii

vol = ImageReader(
    "/path/to/image.nii.gz",
).read()
img = sitk_get_array(
    vol
)

# processing ...... > out_vol or out_img
sitk_write_nii(out_vol, "/path/to/out.nii.gz")
sitk_write_nii(out_img, "/path/to/out.nii.gz", reference=vol)
```

---

## 8. Notes and limitations

- All operations assume a **single-volume image**
- DICOM RTSTRUCT is not handled directly (yet)
- Designed for research and preprocessing pipelines (Medical Imaging)

---

## 9. Installation

### From PyPI

```bash
pip install h-sitk-reader
```

### From source

```bash
git clone <repository-url>
cd h-sitk-reader
pip install .
```

---

## 10. Requirements

- Python >= 3.8
- SimpleITK

