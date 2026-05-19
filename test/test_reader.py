import numpy as np
import SimpleITK as sitk

from medcore.io import ImageReader


def _write_test_nrrd(path) -> np.ndarray:
    array = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.2, 1.3, 1.4))
    sitk.WriteImage(image, str(path))
    return array


def test_image_reader_reads_nrrd_file(tmp_path) -> None:
    nrrd_path = tmp_path / "image.nrrd"
    expected = _write_test_nrrd(nrrd_path)

    image = ImageReader(nrrd_path).read()

    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), expected)
    assert image.GetSpacing() == (1.2, 1.3, 1.4)


def test_image_reader_detects_single_nrrd_in_directory(tmp_path) -> None:
    nrrd_path = tmp_path / "image.nrrd"
    expected = _write_test_nrrd(nrrd_path)

    image = ImageReader(tmp_path).read()

    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), expected)
