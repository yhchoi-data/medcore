# medcore

Medical imaging utilities for DICOM / NIfTI workflows based on SimpleITK.

## Features
- IO
  - DICOM / NIfTI reading (`ImageReader`)
- Image utils
  - intensity array conversion / normalization
  - resampling / transform helpers
  - NIfTI write and label merge utilities
- Detection
  - coronal angle estimation
  - umbilicus detection helpers
- Segmentation
  - torso and abdomen segmentation classes
- Feature extraction
  - label area/volume summary
  - patch extraction around landmark points

## Installation
### From source
```bash
git clone https://github.com/yhchoi-data/medcore.git
cd medcore
pip install .
```

### From source (developer mode)
```bash
git clone https://github.com/yhchoi-data/medcore.git
cd medcore
pip install -e ".[dev]"
```

### From pypi [to be updated]
```bash
pip install medcore
```

### Optional: DICOM to NIfTI conversion with dcm2niix
`medcore.io.convert_dicom_to_nifti` uses the external `dcm2niix` executable.
Install it separately when you need dcm2niix-based DICOM to NIfTI conversion:

```bash
conda install -c conda-forge dcm2niix
```

After installation, make sure the executable is available in your shell or
notebook kernel:

```bash
dcm2niix -h
```

## Quick start

```python
from medcore.io import ImageReader, convert_dicom_to_nifti
from medcore.utils import (
    sitk_get_array,
    sitk_write_nii,
    sitk_read_labelfiles,
)

from medcore.detect import UmbilicusPredictor, UmbilicusDetector
from medcore.segment import TorsoSegmenter, AbdomenSegmenter
from medcore.feature import compute_label_volumes, extract_patches_from_image
```

## Package usage

```python
# IO
from medcore.io import ImageReader

# Utility functions
from medcore.utils import sitk_resampler, figure_overlay_label_on_slices

# Detection / segmentation
from medcore.detect import UmbilicusPredictor
from medcore.segment import TorsoSegmenter

# Feature extraction
from medcore.feature import compute_label_areas, compute_label_volumes
```

## Documentation
Detailed examples are in [USAGE.md](USAGE.md).

## Developer workflow

```bash
pre-commit run --all-files
```

On the first run, some hooks may show `Failed` because they automatically fix files such as trailing whitespace or formatting. That is normal. Run `pre-commit run --all-files` once more and it should pass after the auto-fixes are applied.
