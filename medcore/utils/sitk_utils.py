from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk

from ..io.reader import ImageReader


def sitk_write_nii(
    image: Union[sitk.Image, np.ndarray],
    output_path: str,
    reference: sitk.Image | None = None,
    compress: bool = True,
    overwrite: bool = True,
    verbose: bool = True,
):
    """
    Write image to NIfTI format using SimpleITK.

    Parameters
    ----------
    image : sitk.Image or np.ndarray
        Image to write.
        NumPy array must be in (z, y, x) order.
    out_path : str
        Output NIfTI file path (.nii or .nii.gz).
    reference : sitk.Image
        Reference image to copy physical metadata.
        REQUIRED when image is a NumPy array.
    compress : bool, default=True
        Write compressed NIfTI (.nii.gz) if True.

    Raises
    ------
    ValueError
        If image is np.ndarray and reference is None.
    TypeError
        If image type is unsupported.
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

    # --- case 1: SimpleITK image ---
    if isinstance(image, sitk.Image):
        sitk_img = image

    # --- case 2: NumPy array (reference REQUIRED) ---
    elif isinstance(image, np.ndarray):
        if reference is None:
            raise ValueError("`reference` must be provided when image is a NumPy array.")

        sitk_img = sitk.GetImageFromArray(image)
        sitk_img.CopyInformation(reference)

    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. Expected sitk.Image or np.ndarray."
        )

    sitk.WriteImage(sitk_img, str(output_path))

    if verbose:
        print(f"Saved NIfTI: {output_path}")


def sitk_get_array(
    volume: sitk.Image,
    norm_min: float = -1000,
    norm_max: float = 500,
    normalize: bool = False,
) -> np.ndarray:
    """
    Normalize CT image intensity (HU) to [0, 1] range.

    Parameters:
    -----------
    img : ndarray
        CT 이미지 (HU 값)
    NORM_MIN : int, default=-2000
        정규화 최소값 (공기/폐 영역)
    NORM_MAX : int, default=500
        정규화 최대값 (뼈/금속 영역)

    Returns:
    --------
    img_norm : ndarray
        [0, 1] 범위로 정규화된 이미지 [0~0.2 / 0.2~0.6 / 0.6~1]
    """

    image = sitk.GetArrayFromImage(volume)
    if normalize == True:
        image_norm = np.clip(image, norm_min, norm_max)
        image_norm = (image_norm - norm_min) / (norm_max - norm_min)

        return image_norm
    else:
        return image


def sitk_get_shape_features(mask_img: sitk.Image, label: int = 1) -> dict[str, float]:
    """
    mask_img: SimpleITK Image
        0 = background, label = pancreas
    """

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_img)

    if not stats.HasLabel(label):
        raise ValueError(f"Label {label} not found in mask.")

    physical_size = stats.GetPhysicalSize(label)
    return {
        "volume_mm3": physical_size,
        "volume_ml": physical_size / 1000.0,
        "elongation": stats.GetElongation(label),
        "flatness": stats.GetFlatness(label),
        "roundness": stats.GetRoundness(label),
    }


def sitk_make_euler3dtransform(
    sitk_vol: sitk.Image,
    rotation_deg: float,
    axis: str = "z",
    inverse: bool = False,
) -> sitk.Transform:
    """
    Create a physical-space Euler3DTransform centered at image center.

    Parameters
    ----------
    sitk_vol : sitk.Image
        Reference image (defines physical space).
    rotation_deg : float
        Rotation angle in degrees.
    axis : {"x","y","z"}
        Rotation axis.
    inverse : bool
        Return inverse transform if True.

    Returns
    -------
    sitk.Transform
    """
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unknown axis: {axis}")

    # --- robust physical center ---
    size = np.array(sitk_vol.GetSize(), dtype=np.float64)  # (x,y,z)
    spacing = np.array(sitk_vol.GetSpacing(), dtype=np.float64)
    origin = np.array(sitk_vol.GetOrigin(), dtype=np.float64)
    direction = np.array(sitk_vol.GetDirection(), dtype=np.float64).reshape(3, 3)

    center_offset = (size - 1.0) / 2.0 * spacing
    center = origin + direction @ center_offset

    transform = sitk.Euler3DTransform()
    transform.SetCenter(tuple(center.tolist()))

    rad = float(np.deg2rad(rotation_deg))
    if axis == "x":
        transform.SetRotation(rad, 0.0, 0.0)
    elif axis == "y":
        transform.SetRotation(0.0, rad, 0.0)
    else:  # z
        transform.SetRotation(0.0, 0.0, rad)

    return transform.GetInverse() if inverse else transform


def sitk_resampler(
    sitk_vol: sitk.Image,
    transform: Optional[sitk.Transform] = None,
    new_spacing: Optional[Tuple[float, float, float]] = None,
    interpolation: str = "linear",
    default_pixel: Optional[float] = None,
) -> sitk.Image:
    """
    Resample a sitk.Image using sitk.Resample().

    Parameters
    ----------
    sitk_vol : sitk.Image
        Input image.
    transform : sitk.Transform or None
        Physical-space transform to apply (identity if None).
    new_spacing : (sx, sy, sz) or None
        Output spacing. If None, keep original spacing and size.
    interpolation : {"linear","nn","spline"}
        Interpolation method.
    default_pixel : float or None
        Default pixel value for out-of-bound regions.

    Returns
    -------
    sitk.Image
    """

    # --- interpolator ---
    interpolation = interpolation.lower()
    if interpolation == "linear":
        interpolator = sitk.sitkLinear
    elif interpolation in {"nn", "nearest"}:
        interpolator = sitk.sitkNearestNeighbor
    elif interpolation in {"spline", "bspline"}:
        interpolator = sitk.sitkBSpline
    else:
        raise ValueError(f"Unknown interpolation: {interpolation}")

    if default_pixel is None:
        default_pixel = float(sitk.GetArrayViewFromImage(sitk_vol).min())

    if transform is None:
        transform = sitk.Transform()  # identity

    # --- output geometry ---
    ref = sitk_vol
    out_origin = ref.GetOrigin()
    out_direction = ref.GetDirection()

    if new_spacing is None:
        out_spacing = ref.GetSpacing()
        out_size = ref.GetSize()
    else:
        out_spacing = tuple(map(float, new_spacing))
        in_spacing = np.array(ref.GetSpacing(), dtype=np.float64)
        in_size = np.array(ref.GetSize(), dtype=np.int64)

        new_size = np.round(in_size * (in_spacing / np.array(out_spacing))).astype(int)
        new_size = np.maximum(new_size, 1)
        out_size = tuple(int(x) for x in new_size.tolist())

    out_pixel_id = ref.GetPixelID()

    # --- resample ---
    return sitk.Resample(
        sitk_vol,
        out_size,
        transform,
        interpolator,
        out_origin,
        out_spacing,
        out_direction,
        default_pixel,
        out_pixel_id,
    )


def sitk_resample_point_between_volumes(
    point_zyx: Sequence[float],
    source_volume: sitk.Image,
    target_volume: sitk.Image,
    transform: sitk.Transform,
    *,
    neighborhood_radius: int = 1,  # 1 -> 3x3x3
    allow_outside: bool = True,
    clip_output: bool = False,
) -> List[int]:
    """
    Map a voxel point from source_volume to target_volume by embedding it as a small
    binary mask and resampling that mask.

    Why this approach?
    ------------------
    - This is NOT a direct point transform (TransformPoint).
    - It is robust to discretization/rounding when volumes have different spacing
      and when your pipeline is defined in "resample space" (common in medical imaging).

    Parameters
    ----------
    point_zyx : (z, y, x) sequence
        Voxel index in source_volume array coordinate (same convention as sitk.GetArrayFromImage).
    source_volume : sitk.Image
        Reference image for the input point.
    target_volume : sitk.Image
        Target reference image defining output spacing/geometry.
    transform : sitk.Transform
        The transform used in resampling. We apply its inverse so that the resulting
        mask lands in the target_volume grid.
    neighborhood_radius : int, default=1
        Radius around the point to mark as 1 in the temporary mask.
        radius=1 => 3x3x3, radius=0 => single voxel.
    allow_outside : bool, default=True
        If True, points outside source/target grid are mapped by physical-space geometry
        instead of raising. The returned point may still be outside target_volume.
    clip_output : bool, default=False
        If True, clip the returned target index to the target_volume grid.
        Use this only when downstream code needs a valid array index.

    Returns
    -------
    List[int]
        Mapped voxel index in target grid as [z, y, x].

    Raises
    ------
    ValueError
        If mapped mask is empty and allow_outside=False.
    """
    point_arr_zyx = np.asarray(point_zyx, dtype=np.float64)
    if point_arr_zyx.shape != (3,):
        raise ValueError("`point_zyx` must be a 3-element (z, y, x) sequence.")
    if not np.all(np.isfinite(point_arr_zyx)):
        raise ValueError("`point_zyx` must contain finite values.")

    def _map_by_physical_geometry() -> List[int]:
        point_xyz = tuple(float(v) for v in point_arr_zyx[::-1])
        source_physical = source_volume.TransformContinuousIndexToPhysicalPoint(point_xyz)
        target_physical = transform.TransformPoint(source_physical)
        target_xyz = target_volume.TransformPhysicalPointToIndex(target_physical)
        mapped_zyx = np.array(
            [target_xyz[2], target_xyz[1], target_xyz[0]],
            dtype=np.int64,
        )

        if clip_output:
            max_zyx = np.array(target_volume.GetSize()[::-1], dtype=np.int64) - 1
            mapped_zyx = np.clip(mapped_zyx, 0, max_zyx)

        return mapped_zyx.astype(int).tolist()

    # 1) Build small binary mask in numpy (Z,Y,X)
    src_arr = sitk.GetArrayViewFromImage(source_volume)
    mask = np.zeros(src_arr.shape, dtype=np.uint8)

    z, y, x = (int(v) for v in point_arr_zyx)
    r = int(neighborhood_radius)
    if r < 0:
        raise ValueError("`neighborhood_radius` must be >= 0.")

    source_shape = np.array(mask.shape, dtype=np.float64)
    point_inside_source = bool(np.all((0.0 <= point_arr_zyx) & (point_arr_zyx < source_shape)))
    if not point_inside_source:
        if allow_outside:
            return _map_by_physical_geometry()
        raise ValueError(
            "Input point is outside source_volume grid. "
            "Set allow_outside=True to map it by physical-space geometry."
        )

    z0, z1 = max(0, z - r), min(mask.shape[0], z + r + 1)
    y0, y1 = max(0, y - r), min(mask.shape[1], y + r + 1)
    x0, x1 = max(0, x - r), min(mask.shape[2], x + r + 1)

    mask[z0:z1, y0:y1, x0:x1] = 1

    # 2) Convert mask -> sitk and copy geometry from source
    mask_img = sitk.GetImageFromArray(
        mask
    )  # creates (x,y,z) image internally, consistent with SITK
    mask_img.CopyInformation(source_volume)

    # 3) Resample into target grid using inverse transform (NN for mask)
    inv_t = transform.GetInverse()

    # Use sitk.Resample directly (reference image defines output origin/dir/size/spacing)
    out = sitk.Resample(
        mask_img,
        target_volume,  # reference image defines grid
        inv_t,
        sitk.sitkNearestNeighbor,
        0,  # default pixel for outside
        sitk.sitkUInt8,
    )

    out_mask = sitk.GetArrayViewFromImage(out)

    # 4) Recover mapped point as robust center (median of indices where mask==1)
    idx = np.where(out_mask == 1)
    if idx[0].size == 0:
        if allow_outside:
            return _map_by_physical_geometry()
        raise ValueError(
            "Mapped mask is empty. The point may be outside target FOV or transform/grid mismatch."
        )

    mapped_zyx = np.median(np.vstack(idx), axis=1).astype(int).tolist()
    if clip_output:
        max_zyx = np.array(target_volume.GetSize()[::-1], dtype=np.int64) - 1
        mapped_zyx = np.clip(np.array(mapped_zyx, dtype=np.int64), 0, max_zyx).tolist()

    return mapped_zyx


def sitk_read_labelfiles(labelfiles: Mapping[int, Union[str, Path]]) -> sitk.Image:
    """
    Read multiple label files and combine them into one UInt8 label volume.

    Parameters
    ----------
    labelfiles : mapping
        Mapping of {label_value: image_path}.

    Returns
    -------
    sitk.Image
        Combined label image (UInt8).
    """
    if not labelfiles:
        raise ValueError("`labelfiles` is empty.")

    reference_path = next(iter(labelfiles.values()))
    reference_img = ImageReader(str(reference_path)).read()

    combined = sitk.Image(reference_img.GetSize(), sitk.sitkUInt8)
    combined.CopyInformation(reference_img)

    for label, filepath in labelfiles.items():
        mask = ImageReader(filepath).read()
        mask = mask > 0
        mask = sitk.Cast(mask, sitk.sitkUInt8) * int(label)
        combined = sitk.Maximum(combined, mask)

    return combined


def sitk_create_shell_mask(
    mask: sitk.Image,
    inner_mm: float = 5.0,
    outer_mm: float = 10.0,
) -> sitk.Image:
    """
    Create an external shell mask around a binary mask.

    Parameters
    ----------
    mask : sitk.Image
        Input binary mask. Non-zero voxels are treated as foreground.
    inner_mm : float, default=5.0
        Inner shell distance from the foreground boundary in millimeters.
    outer_mm : float, default=10.0
        Outer shell distance from the foreground boundary in millimeters.

    Returns
    -------
    sitk.Image
        UInt8 shell mask with the same geometry as ``mask``.
    """
    inner_mm = float(inner_mm)
    outer_mm = float(outer_mm)
    if inner_mm < 0:
        raise ValueError("`inner_mm` must be >= 0.")
    if outer_mm <= inner_mm:
        raise ValueError("`outer_mm` must be greater than `inner_mm`.")

    binary_mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    mask_arr = sitk.GetArrayFromImage(binary_mask) > 0
    if not np.any(mask_arr):
        raise ValueError("`mask` must contain at least one foreground voxel.")

    dist_map = sitk.SignedMaurerDistanceMap(
        binary_mask,
        squaredDistance=False,
        useImageSpacing=True,
        insideIsPositive=False,
    )
    dist_arr = sitk.GetArrayFromImage(dist_map)
    shell = (~mask_arr) & (dist_arr >= inner_mm) & (dist_arr < outer_mm)

    shell_mask = sitk.GetImageFromArray(shell.astype(np.uint8))
    shell_mask.CopyInformation(mask)
    return shell_mask


def sitk_copy_metainfo(volume: sitk.Image, image: np.ndarray) -> sitk.Image:
    """
    sitk 이미지 → numpy 처리 → sitk 복원

    Parameters:
    -----------
    sitk_img : SimpleITK.Image
        입력 이미지
    image : np.ndarray
        processed image

    Returns:
    --------
    processed_img : SimpleITK.Image
        처리된 sitk 이미지
    """
    processed_volume = sitk.GetImageFromArray(image)
    processed_volume.CopyInformation(volume)  # 메타데이터 복사
    return processed_volume


def sitk_to_nib_affine(volume: sitk.Image, verbose: bool = False) -> np.ndarray:
    """
    Convert SimpleITK image geometry to a nibabel-compatible affine.

    SimpleITK uses LPS physical coordinates, so the affine built from spacing,
    origin, and direction is converted from LPS to RAS before returning.

    Parameters
    ----------
    volume : sitk.Image
        Input SimpleITK image. Its spacing, origin, and direction are used.
    verbose : bool, default=False
        If True, print the SimpleITK orientation string inferred from
        ``volume.GetDirection()``.

    Returns
    -------
    np.ndarray
        A 4x4 affine matrix compatible with nibabel, e.g. usable as input to
        ``nib.aff2axcodes(affine)`` or ``nib.Nifti1Image(array_xyz, affine)``.

    Notes
    -----
    ``sitk.GetArrayFromImage(volume)`` returns data in ``(z, y, x)`` order.
    If creating a nibabel image, transpose the array to ``(x, y, z)`` before
    pairing it with the returned affine.
    """

    spacing = np.array(volume.GetSpacing(), dtype=float)  # x, y, z
    origin = np.array(volume.GetOrigin(), dtype=float)  # LPS
    direction = np.array(volume.GetDirection(), dtype=float).reshape(3, 3)

    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin

    # Convert the affine's world coordinate convention from LPS to RAS
    # so nibabel.aff2axcodes interprets it correctly.
    lps_to_ras = np.diag([-1, -1, 1, 1])
    affine_ras = lps_to_ras @ affine_lps

    if verbose == True:
        sitk_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            volume.GetDirection()
        )
        print("Oritentation[sitk]:", sitk_orient)

    return affine_ras
