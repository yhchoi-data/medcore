import numpy as np
import pandas as pd
import SimpleITK as sitk

from medcore.pipeline.umbilicus_detection import UmbilicusDetectionPipeline, _safe_mean_hu


def _image(array: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def test_safe_mean_hu_clips_patch_to_image_bounds() -> None:
    image = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

    expected = image[0:3, 0:5, 0:3].mean()

    assert _safe_mean_hu(image, si=0, ap=0, lr=0) == float(expected)
    assert _safe_mean_hu(image, si=0, ap=10, lr=0) == float("-inf")


def test_pipeline_selects_filtered_candidate_with_highest_mean_hu(monkeypatch) -> None:
    extractor = UmbilicusDetectionPipeline()
    image = np.zeros((20, 20, 20), dtype=np.int16)
    image[10:15, 12:17, 10:15] = 100
    image[3:8, 5:10, 3:8] = -100
    vol = _image(image)

    predictor_points = pd.DataFrame([[12, 12, 12, 1.0]], columns=["SI", "AP", "LR", "SCORE"])
    detector_points = pd.DataFrame(
        [[5, 5, 5, -1.0, 0.5]], columns=["SI", "AP", "LR", "MIN_CV", "MEAN_VAL"]
    )
    abdomen_info = {"height_start": 4}

    monkeypatch.setattr(extractor, "_load_volume", lambda ct_input: vol)
    monkeypatch.setattr(
        extractor,
        "_make_supine_transform",
        lambda ct_vol: (sitk.Transform(), sitk.Transform(), 0.0),
    )
    monkeypatch.setattr(extractor, "_resample_to_supine", lambda ct_vol, transform: vol)
    monkeypatch.setattr(extractor.predictor, "predict", lambda ct_vol: predictor_points)
    monkeypatch.setattr(
        extractor.abdomen_segmenter,
        "segment",
        lambda ct_vol: (
            np.zeros((1, 1, 1), dtype=np.float32),
            np.ones((1, 1, 1), dtype=np.uint8),
            abdomen_info,
            [np.zeros((1, 2), dtype=int)] * 5,
        ),
    )
    monkeypatch.setattr(
        extractor.detector,
        "detect",
        lambda **kwargs: detector_points,
    )

    result = extractor.run("ct.nii.gz")

    assert result.umbilicus_points.loc["RAW"].tolist() == [12.0, 12.0, 12.0]
    assert result.umbilicus_points.loc["ISO"].tolist() == [12.0, 12.0, 12.0]
    assert result.umbilicus_points.loc["VOX"].tolist() == [1.0, 1.0, 1.0]
    assert result.candidates["SOURCE"].tolist() == ["predictor", "detector"]
