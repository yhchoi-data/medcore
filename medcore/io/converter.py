from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import (
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    generate_uid,
)


def convert_dicom_to_nifti(
    dicom_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    compress: bool = True,
    bids_sidecar: bool = True,
    filename_format: str | None = None,
    fallback_part10: bool = True,
    search_depth: int | None = None,
    verbose: bool = True,
) -> list[Path]:
    """
    Convert a DICOM folder to NIfTI using the dcm2niix executable.

    If dcm2niix cannot detect DICOM files because they are missing the Part 10
    preamble/header, the function can rewrite force-readable DICOM slices into a
    temporary Part 10 folder and retry conversion.
    """
    dicom_dir = Path(dicom_dir)
    output_dir = Path(output_dir)

    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    if not dicom_dir.is_dir():
        raise NotADirectoryError(f"DICOM path must be a directory: {dicom_dir}")

    dcm2niix = shutil.which("dcm2niix")
    if dcm2niix is None:
        raise RuntimeError(
            "dcm2niix executable not found. "
            "Install it with `conda install -c conda-forge dcm2niix` "
            "or make sure it is available in PATH."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = set(output_dir.iterdir())

    def run(
        input_dir: Path,
        *,
        force_filename_format: str | None = None,
    ) -> subprocess.CompletedProcess:
        cmd = _build_dcm2niix_command(
            dcm2niix,
            input_dir,
            output_dir,
            compress=compress,
            bids_sidecar=bids_sidecar,
            filename_format=force_filename_format or filename_format,
            search_depth=search_depth,
        )
        return subprocess.run(cmd, check=True, text=True, capture_output=True)

    try:
        result = run(dicom_dir)
    except subprocess.CalledProcessError as error:
        if not fallback_part10 or not _is_no_dicom_error(error):
            _print_process_output(error, verbose=verbose)
            raise

        with tempfile.TemporaryDirectory(prefix="dcm2niix_part10_") as temp_dir:
            fixed_dir = Path(temp_dir)
            converted = _write_part10_dicoms(dicom_dir, fixed_dir)
            if converted == 0:
                raise RuntimeError(
                    f"No force-readable DICOM images found under {dicom_dir}"
                ) from error

            fallback_filename_format = filename_format or f"{dicom_dir.name}_%p_%t_%s"
            result = run(fixed_dir, force_filename_format=fallback_filename_format)

    _print_process_output(result, verbose=verbose)
    generated_outputs = sorted(set(output_dir.iterdir()) - existing_outputs)
    return generated_outputs


def _build_dcm2niix_command(
    dcm2niix: str,
    input_dir: Path,
    output_dir: Path,
    *,
    compress: bool,
    bids_sidecar: bool,
    filename_format: str | None,
    search_depth: int | None,
) -> list[str]:
    cmd = [
        dcm2niix,
        "-z",
        "y" if compress else "n",
        "-b",
        "y" if bids_sidecar else "n",
    ]

    if search_depth is not None:
        cmd.extend(["-d", str(int(search_depth))])
    if filename_format is not None:
        cmd.extend(["-f", filename_format])

    cmd.extend(["-o", str(output_dir), str(input_dir)])
    return cmd


def _is_no_dicom_error(error: subprocess.CalledProcessError) -> bool:
    output = "\n".join(text for text in [error.stdout, error.stderr] if text)
    return "Unable to find any DICOM images" in output


def _print_process_output(
    result: subprocess.CompletedProcess | subprocess.CalledProcessError,
    *,
    verbose: bool,
) -> None:
    if not verbose:
        return
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")


def _write_part10_dicoms(src_dir: Path, dst_dir: Path) -> int:
    converted = 0
    for src_file in _iter_files(src_dir):
        try:
            ds = pydicom.dcmread(str(src_file), force=True)
        except Exception:
            continue

        if not hasattr(ds, "PixelData") or not hasattr(ds, "SOPClassUID"):
            continue

        if not hasattr(ds, "SOPInstanceUID"):
            ds.SOPInstanceUID = generate_uid()

        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            ds.file_meta = FileMetaDataset()

        transfer_syntax = getattr(ds.file_meta, "TransferSyntaxUID", None)
        if transfer_syntax is None:
            transfer_syntax = _infer_transfer_syntax(ds)

        ds.file_meta.TransferSyntaxUID = transfer_syntax
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.FileMetaInformationVersion = b"\x00\x01"

        if not getattr(ds.file_meta, "ImplementationClassUID", None):
            ds.file_meta.ImplementationClassUID = generate_uid()

        ds.preamble = b"\0" * 128
        dst_file = dst_dir / f"{converted:06d}_{src_file.name}.dcm"
        ds.save_as(str(dst_file), write_like_original=False)
        converted += 1

    return converted


def _iter_files(src_dir: Path) -> list[Path]:
    return sorted(path for path in src_dir.rglob("*") if path.is_file())


def _infer_transfer_syntax(ds: pydicom.Dataset) -> str:
    if not getattr(ds, "is_little_endian", True):
        return ExplicitVRBigEndian
    if getattr(ds, "is_implicit_VR", True):
        return ImplicitVRLittleEndian
    return ExplicitVRLittleEndian
