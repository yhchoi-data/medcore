from pathlib import Path
import subprocess

import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.filewriter import dcmwrite
from pydicom.uid import CTImageStorage, ImplicitVRLittleEndian, generate_uid

from medcore.io import convert_dicom_to_nifti
from medcore.io import converter


def test_convert_dicom_to_nifti_raises_when_dcm2niix_missing(tmp_path, monkeypatch) -> None:
    dicom_dir = tmp_path / "dicom"
    dicom_dir.mkdir()

    monkeypatch.setattr(converter.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="dcm2niix executable not found"):
        convert_dicom_to_nifti(dicom_dir, tmp_path / "out", verbose=False)


def test_convert_dicom_to_nifti_runs_dcm2niix(tmp_path, monkeypatch) -> None:
    dicom_dir = tmp_path / "dicom"
    output_dir = tmp_path / "out"
    dicom_dir.mkdir()
    commands = []

    monkeypatch.setattr(converter.shutil, "which", lambda name: "/usr/bin/dcm2niix")

    def fake_run(cmd, check, text, capture_output):
        commands.append(cmd)
        output_dir.mkdir(exist_ok=True)
        (output_dir / "converted.nii.gz").write_text("nii")
        return subprocess.CompletedProcess(cmd, 0, stdout="converted\n", stderr="")

    monkeypatch.setattr(converter.subprocess, "run", fake_run)

    outputs = convert_dicom_to_nifti(
        dicom_dir,
        output_dir,
        compress=False,
        bids_sidecar=False,
        filename_format="case_%s",
        search_depth=9,
        verbose=False,
    )

    assert outputs == [output_dir / "converted.nii.gz"]
    assert commands == [
        [
            "/usr/bin/dcm2niix",
            "-z",
            "n",
            "-b",
            "n",
            "-d",
            "9",
            "-f",
            "case_%s",
            "-o",
            str(output_dir),
            str(dicom_dir),
        ]
    ]


def test_convert_dicom_to_nifti_falls_back_to_temporary_part10_folder(
    tmp_path,
    monkeypatch,
) -> None:
    dicom_dir = tmp_path / "EAP"
    output_dir = tmp_path / "out"
    dicom_dir.mkdir()
    commands = []
    fallback_inputs = []

    monkeypatch.setattr(converter.shutil, "which", lambda name: "/usr/bin/dcm2niix")

    def fake_write_part10_dicoms(src_dir: Path, dst_dir: Path) -> int:
        (dst_dir / "slice.dcm").write_text("dicom")
        return 1

    def fake_run(cmd, check, text, capture_output):
        commands.append(cmd)
        input_dir = Path(cmd[-1])
        if len(commands) == 1:
            raise subprocess.CalledProcessError(
                2,
                cmd,
                output="",
                stderr=f"Error: Unable to find any DICOM images in {input_dir}",
            )

        fallback_inputs.append(input_dir)
        assert input_dir.exists()
        assert (input_dir / "slice.dcm").exists()
        output_dir.mkdir(exist_ok=True)
        (output_dir / "EAP_ct.nii.gz").write_text("nii")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(converter, "_write_part10_dicoms", fake_write_part10_dicoms)
    monkeypatch.setattr(converter.subprocess, "run", fake_run)

    outputs = convert_dicom_to_nifti(dicom_dir, output_dir, verbose=False)

    assert outputs == [output_dir / "EAP_ct.nii.gz"]
    assert fallback_inputs
    assert not fallback_inputs[0].exists()
    assert "-f" not in commands[0]
    assert commands[1][commands[1].index("-f") + 1] == "EAP_%p_%t_%s"


def test_write_part10_dicoms_rewrites_force_readable_dicom(tmp_path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()

    _write_non_part10_dicom(src_dir / "slice")

    converted = converter._write_part10_dicoms(src_dir, dst_dir)

    assert converted == 1
    output_file = next(dst_dir.iterdir())
    assert output_file.read_bytes()[128:132] == b"DICM"
    ds = pydicom.dcmread(str(output_file), force=False)
    assert ds.SOPClassUID == CTImageStorage


def _write_non_part10_dicom(path: Path) -> None:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.Rows = 1
    ds.Columns = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = b"\0\0"
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    dcmwrite(str(path), ds, write_like_original=True)
