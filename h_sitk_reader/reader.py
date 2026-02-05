from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk


class ImageReader:
    """
    Medical image reader for NIfTI (.nii/.nii.gz) and DICOM folders.

    Features
    --------
    - NIfTI: read via SimpleITK
    - DICOM: prefer SimpleITK ImageSeriesReader (robust), fallback to pydicom stacking
    - Multi-series selection: filter + scoring by Modality / SeriesDescription / BodyPartExamined / SliceThickness
    - Orientation standardization: SimpleITK DICOMOrientImageFilter (default LPS)
    - Export: numpy array, normalized array, metadata dataframe
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        check_coord_flag: bool = True,
        verbose: bool = False,
        target_orientation: str = "LPS",  # "LPS" or "RAS"
        # --- DICOM series selection options ---
        prefer_modality: Optional[str] = None,                  # e.g. "CT", "MR"
        include_series_description: Optional[List[str]] = None,  # AND keywords, e.g. ["lung", "thorax"]
        exclude_series_description: Optional[List[str]] = None,  # e.g. ["localizer", "scout"]
        prefer_body_part: Optional[List[str]] = None,           # e.g. ["CHEST", "LUNG"]
        # --- Thin-slice preference ---
        prefer_thin_slice: bool = True,
        max_slice_thickness_mm: Optional[float] = None,          # hard filter if thickness known (e.g. 5.0)
    ) -> None:
        self.input_path = Path(input_path)
        self.check_coord_flag = check_coord_flag
        self.verbose = verbose
        self.target_orientation = target_orientation.upper()

        # selection options
        self.prefer_modality = prefer_modality.upper() if prefer_modality else None
        self.include_series_description = include_series_description or []
        self.exclude_series_description = exclude_series_description or []
        self.prefer_body_part = prefer_body_part or []
        self.prefer_thin_slice = prefer_thin_slice
        self.max_slice_thickness_mm = max_slice_thickness_mm

        self.sitk_volume = self.load_medical_image(self.input_path)

        if self.check_coord_flag:
            self.sitk_volume = self.standardize_orientation(
                self.sitk_volume, target_orientation=self.target_orientation
            )

    # -------------------------
    # Convenience API
    # -------------------------
    def read(self) -> sitk.Image:
        return self.sitk_volume

    def to_nifti(
        self,
        output_path: Union[str, Path],
        compress: bool = True,
        overwrite: bool = True,
        orientation: str | None = None,  # None / "LPS" / "RAS"
    ) -> None:
        """
        Save current image as NIfTI (.nii or .nii.gz).

        Parameters
        ----------
        output_path : str or Path
            Output file path (.nii or .nii.gz).
        compress : bool, default=True
            If True and suffix is .nii, save as .nii.gz.
        overwrite : bool, default=True
            Overwrite existing file if True.
        """
        output_path = Path(output_path)

        # suffix 처리
        if output_path.suffix not in {".nii", ".gz"}:
            raise ValueError("Output path must end with .nii or .nii.gz")

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {output_path}")

        # .nii + compress=True → .nii.gz
        if output_path.suffix == ".nii" and compress:
            output_path = output_path.with_suffix(".nii.gz")

        # 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # orientation 확인
        sitk_volume = self.sitk_volume
        if orientation:
            sitk_volume = sitk.DICOMOrient(sitk_volume, orientation)
        sitk.WriteImage(sitk_volume, str(output_path))

        if self.verbose:
            print(f"Saved NIfTI: {output_path}")

    def get_numpy_array(self) -> np.ndarray:
        """Return image as numpy array (z, y, x)."""
        return sitk.GetArrayFromImage(self.sitk_volume)

    def get_normalized_array(self, norm_min: float = -2000, norm_max: float = 500) -> np.ndarray:
        """
        Normalize CT intensity (HU) to [0, 1] by clipping to [norm_min, norm_max].
        """
        img = self.get_numpy_array().astype(np.float32)
        img = np.clip(img, norm_min, norm_max)
        denom = (norm_max - norm_min) if (norm_max - norm_min) != 0 else 1.0
        return (img - norm_min) / denom

    def get_metadata(self) -> pd.DataFrame:
        """
        Extract metadata keys stored in the SimpleITK image (best-effort).
        """
        rows = []
        for key in self.sitk_volume.GetMetaDataKeys():
            value = self.sitk_volume.GetMetaData(key)

            keyword = "Unknown"
            description = "Unknown Tag"
            try:
                group, element = [int(k, 16) for k in key.split("|")]
                tag = pydicom.tag.Tag(group, element)
                keyword = pydicom.datadict.keyword_for_tag(tag) or "Unknown"
                description = pydicom.datadict.dictionary_description(tag) or "Unknown Tag"
            except Exception:
                pass

            rows.append(
                {"Tag": key, "Keyword": keyword, "Description": description, "Value": value}
            )
        return pd.DataFrame(rows)

    # -------------------------
    # Load pipeline
    # -------------------------
    def load_medical_image(self, input_path: Path) -> sitk.Image:

        if input_path.is_file():
            has_nifti = self._is_nifti_file(input_path)
            if has_nifti:
                return self._load_nifti(input_path)
            else:
                raise ValueError(f"Check File: {input_path}")
            
        if input_path.is_dir():
            files = [p for p in input_path.iterdir() if p.is_file()]
            nifti_files = [p for p in files if self._is_nifti_file(p)]
            has_nifti = len(nifti_files) > 0
            # DICOM은 몇 개만 probe
            has_dicom = False
            for p in files[: min(10, len(files))]:
                if self._is_nifti_file(p):
                    continue
                if self._probe_is_dicom(p):
                    has_dicom = True
                    break

            # 1) 파일 없음
            if (not has_nifti) and (not has_dicom):
                # 실제론 "파일은 있는데 DICOM도 NIfTI도 아닌" 케이스일 수도 있으니 메시지 분리 가능
                files = [p for p in input_path.iterdir() if p.is_file()]
                if not files:
                    raise ValueError(f"Empty directory: {input_path}")
                raise ValueError(f"Directory has files but no NIfTI/DICOM detected: {input_path}")

            # 2) NIfTI만 있음
            if has_nifti and (not has_dicom):
                # 보통 nii/nii.gz가 1개일 때 기대하지만, 여러 개면 정책 필요
                if len(nifti_files) == 1:
                    if self.verbose:
                        print(f"Detected NIfTI in directory: {nifti_files[0].name}")
                    return self._load_nifti(nifti_files[0])

                # 여러 NIfTI 파일이면: 에러 or 첫 번째 선택 (여기서는 에러 권장)
                raise ValueError(
                    f"Multiple NIfTI files found in directory; please pass a file path.\n"
                    f"Found: {[p.name for p in nifti_files]}"
                )

            # 3) DICOM만 있음
            if has_dicom and (not has_nifti):
                if self.verbose:
                    print("Detected DICOM-only directory.")
                return self._load_dicom_folder(input_path)

            # 4) 섞여 있음 (NIfTI + DICOM) 기본은 error 권장. 필요시 정책으로 선택 
            if has_dicom and has_nifti:
                raise ValueError(
                    f"Mixed directory (NIfTI + DICOM) is not allowed: {input_path}\n"
                    f"NIfTI: {[p.name for p in nifti_files[:5]]}{'...' if len(nifti_files) > 5 else ''}"
                )

        raise ValueError(f"Unsupported input: {input_path}")

    def _is_nifti_file(self, folder: Path) -> bool:
        name = folder.name.lower()
        return name.endswith(".nii") or name.endswith(".nii.gz")

    def _probe_is_dicom(self, folder: Path) -> bool:
        # 확장자 없는 dicom이 많아서 header sniffing
        try:
            _ = pydicom.dcmread(str(folder), stop_before_pixels=True, force=True)
            # 파일이 DICOM인지 더 확실히 하고 싶으면 SOPClassUID 같은 태그 존재 체크 추가 가능
            return True
        except Exception:
            return False
        
    def _load_nifti(self, folder: Path) -> sitk.Image:
        name = folder.name.lower()
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            raise ValueError(f"Unsupported file format (expect .nii/.nii.gz): {folder}")

        if self.verbose:
            print("Detected NIfTI format.")
        return sitk.ReadImage(str(folder))

    def _load_dicom_folder(self, folder: Path) -> sitk.Image:
        if self.verbose:
            print("Detected DICOM folder. Trying SimpleITK ImageSeriesReader first...")

        # 1) Prefer SimpleITK series reader (more robust)
        try:
            img = self._read_dicom_series_sitk(folder)
            if self.verbose:
                print("Loaded DICOM via SimpleITK ImageSeriesReader.")
            return img
        except Exception as e:
            if self.verbose:
                print(f"[WARN] SimpleITK series read failed, fallback to pydicom stacking. Reason: {e}")

        # 2) Fallback to pydicom stacking (single best series by UID)
        try:
            volume, first_ds = self.dcmread_series(str(folder))
            img = self.array2sitk(volume, first_ds)
            self._attach_dicom_metadata(img, first_ds)
            if self.verbose:
                print("Loaded DICOM via pydicom stacking fallback.")
            return img
        except Exception as e:
            raise RuntimeError("No valid DICOM series found in the directory.") from e

    def _read_dicom_series_sitk(self, folder: Path) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(folder))
        if not series_ids:
            raise RuntimeError("No DICOM series IDs found.")

        best_files = self._select_best_series_files(folder, series_ids)
        if best_files is None:
            raise RuntimeError("Failed to select DICOM series by criteria.")

        reader.SetFileNames(best_files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOff()
        img = reader.Execute()

        # Optionally copy representative per-slice metadata into image metadata
        try:
            if reader.GetMetaDataDictionaryArraySize() > 0:
                md0 = reader.GetMetaDataDictionaryArray()[0]
                for k in md0.GetKeys():
                    try:
                        img.SetMetaData(k, md0[k])
                    except Exception:
                        pass
        except Exception:
            pass

        return img

    # -------------------------
    # DICOM series selection (filter + scoring)
    # -------------------------
    def _select_best_series_files(
        self, folder: Path, series_ids: List[str]
    ) -> Optional[List[str]]:
        candidates = []

        for sid in series_ids:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), sid)
            if not files:
                continue

            meta = self._read_dicom_meta(files[0])
            desc = (meta.get("SeriesDescription") or "").lower()
            mod = (meta.get("Modality") or "").upper()
            bpe = (meta.get("BodyPartExamined") or "").upper()

            # ---- Hard filters ----
            if self.exclude_series_description:
                if any(k.lower() in desc for k in self.exclude_series_description):
                    continue

            if self.prefer_modality:
                if mod and mod != self.prefer_modality:
                    continue

            if self.include_series_description:
                # AND: all keywords must appear
                if not all(k.lower() in desc for k in self.include_series_description):
                    continue

            # Slice thickness hard filter (if known)
            thk = meta.get("SpacingBetweenSlices") or meta.get("SliceThickness")
            if self.max_slice_thickness_mm is not None and thk is not None:
                if thk > float(self.max_slice_thickness_mm):
                    continue

            # ---- Soft scoring ----
            score = 0.0

            # modality match bonus
            if self.prefer_modality and mod == self.prefer_modality:
                score += 100.0

            # body part bonus
            if self.prefer_body_part and bpe:
                if any(bp.upper() in bpe for bp in self.prefer_body_part):
                    score += 30.0

            # description keyword bonus (soft)
            if self.include_series_description and desc:
                matches = sum(1 for k in self.include_series_description if k.lower() in desc)
                score += 10.0 * matches

            # thin slice preference (soft)
            if self.prefer_thin_slice and thk is not None and thk > 0:
                score += 10.0 / float(thk)  # 0.5mm=20, 1mm=10, 2mm=5...

            # number of files (soft, scaled)
            score += min(len(files), 500) / 5.0

            candidates.append((score, len(files), sid, files, meta))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = candidates[0]

        if self.verbose:
            score, nfiles, sid, _, meta = best
            print("[DICOM] Selected series:")
            print(f"  SeriesInstanceUID: {sid}")
            print(f"  Score: {score:.2f}, Files: {nfiles}")
            print(f"  Modality: {meta.get('Modality')}")
            print(f"  SeriesDescription: {meta.get('SeriesDescription')}")
            print(f"  BodyPartExamined: {meta.get('BodyPartExamined')}")
            print(f"  SliceThickness: {meta.get('SliceThickness')}")
            print(f"  SpacingBetweenSlices: {meta.get('SpacingBetweenSlices')}")

        return best[3]

    def _read_dicom_meta(self, dcm_path: str) -> Dict[str, Any]:
        """
        Read minimal DICOM tags (without pixels) for series selection.
        """
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
        except Exception:
            return {}

        def get_str(name: str) -> Optional[str]:
            v = getattr(ds, name, None)
            return None if v is None else str(v)

        def get_float(name: str) -> Optional[float]:
            v = getattr(ds, name, None)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        return {
            "Modality": get_str("Modality"),
            "SeriesDescription": get_str("SeriesDescription"),
            "BodyPartExamined": get_str("BodyPartExamined"),
            "SeriesNumber": get_str("SeriesNumber"),
            "ProtocolName": get_str("ProtocolName"),
            "SliceThickness": get_float("SliceThickness"),
            "SpacingBetweenSlices": get_float("SpacingBetweenSlices"),
        }

    # -------------------------
    # Orientation standardization
    # -------------------------
    @staticmethod
    def standardize_orientation(img: sitk.Image, target_orientation: str = "LPS") -> sitk.Image:
        """
        Standardize image orientation using SimpleITK DICOMOrientImageFilter.
        target_orientation: "LPS" or "RAS"
        """
        target_orientation = target_orientation.upper()
        if target_orientation not in {"LPS", "RAS"}:
            raise ValueError("target_orientation must be 'LPS' or 'RAS'.")

        f = sitk.DICOMOrientImageFilter()
        f.SetDesiredCoordinateOrientation(target_orientation)
        return f.Execute(img)

    # -------------------------
    # pydicom fallback helpers (kept)
    # -------------------------
    @staticmethod
    def dcmread_series(folder_path: str) -> Tuple[np.ndarray, pydicom.Dataset]:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        dicoms = []

        for f in files:
            try:
                ds = pydicom.dcmread(f, force=True)
                if hasattr(ds, "PixelData"):
                    dicoms.append(ds)
            except Exception:
                continue

        if len(dicoms) == 0:
            raise RuntimeError("No valid DICOM files found in folder.")

        series_uids = [ds.SeriesInstanceUID for ds in dicoms if hasattr(ds, "SeriesInstanceUID")]
        if not series_uids:
            raise RuntimeError("No SeriesInstanceUID found in DICOM files.")

        most_common_uid, _ = Counter(series_uids).most_common(1)[0]
        dicoms = [ds for ds in dicoms if getattr(ds, "SeriesInstanceUID", None) == most_common_uid]

        # Sort: InstanceNumber -> ImagePositionPatient[2]
        def sort_key(ds: pydicom.Dataset) -> float:
            if hasattr(ds, "InstanceNumber"):
                try:
                    return float(ds.InstanceNumber)
                except Exception:
                    pass
            if hasattr(ds, "ImagePositionPatient"):
                try:
                    return float(ds.ImagePositionPatient[2])
                except Exception:
                    pass
            return 0.0

        dicoms.sort(key=sort_key)

        slices = [d.pixel_array for d in dicoms]
        volume = np.stack(slices, axis=-1).astype(np.float32)  # (H, W, D)

        intercept = float(dicoms[0].get("RescaleIntercept", 0.0))
        slope = float(dicoms[0].get("RescaleSlope", 1.0))
        volume = volume * slope + intercept

        return volume, dicoms[0]

    @staticmethod
    def array2sitk(volume: np.ndarray, reference_ds: pydicom.Dataset) -> sitk.Image:
        # volume: (H, W, D) -> SITK expects array (Z, Y, X)
        img = sitk.GetImageFromArray(volume.transpose(2, 0, 1))

        # Spacing
        spacing_xy = [1.0, 1.0]
        if hasattr(reference_ds, "PixelSpacing"):
            try:
                spacing_xy = list(map(float, reference_ds.PixelSpacing))
            except Exception:
                pass

        spacing_z = float(getattr(reference_ds, "SliceThickness", 1.0))
        img.SetSpacing(spacing_xy + [spacing_z])

        # Origin
        if hasattr(reference_ds, "ImagePositionPatient"):
            try:
                origin = list(map(float, reference_ds.ImagePositionPatient))
                img.SetOrigin(origin)
            except Exception:
                pass

        # Direction
        if hasattr(reference_ds, "ImageOrientationPatient"):
            try:
                iop = list(map(float, reference_ds.ImageOrientationPatient))  # 6
                row = np.array(iop[:3], dtype=np.float64)
                col = np.array(iop[3:], dtype=np.float64)
                normal = np.cross(row, col)
                direction = np.concatenate([row, col, normal])
                img.SetDirection(direction.tolist())
            except Exception:
                pass

        return img

    def _attach_dicom_metadata(self, image: sitk.Image, first_ds: pydicom.Dataset) -> None:
        fail = 0
        total = 0
        for elem in first_ds:
            if elem.tag.is_private:
                continue
            total += 1
            tag_str = f"{elem.tag.group:04x}|{elem.tag.element:04x}"
            try:
                image.SetMetaData(tag_str, str(elem.value))
            except Exception:
                fail += 1

                

        if self.verbose:
            print(f"Metadata copied from pydicom first slice: {total - fail}/{total} succeeded.")

