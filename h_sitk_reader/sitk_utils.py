from typing import Optional, Tuple, Sequence, List
import SimpleITK as sitk
import numpy as np

def sitk_get_array(
        volume: sitk.Image,
        norm_min: float = -2000,
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
        image_norm = (image_norm - norm_min) / (norm_max-norm_min)

        return image_norm
    else:
        return image

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
    size = np.array(sitk_vol.GetSize(), dtype=np.float64)        # (x,y,z)
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
    point_zyx: Sequence[int],
    source_volume: sitk.Image,
    target_volume: sitk.Image,
    transform: sitk.Transform,
    *,
    neighborhood_radius: int = 1,  # 1 -> 3x3x3
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
    point_zyx : (z, y, x) int sequence
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

    Returns
    -------
    List[int]
        Mapped voxel index in target grid as [z, y, x].

    Raises
    ------
    ValueError
        If mapped mask is empty (assumes point went out-of-FOV or transform mismatch).
    """
    # 1) Build small binary mask in numpy (Z,Y,X)
    src_arr = sitk.GetArrayViewFromImage(source_volume)
    mask = np.zeros(src_arr.shape, dtype=np.uint8)

    z, y, x = map(int, point_zyx)
    r = int(neighborhood_radius)

    z0, z1 = max(0, z - r), min(mask.shape[0], z + r + 1)
    y0, y1 = max(0, y - r), min(mask.shape[1], y + r + 1)
    x0, x1 = max(0, x - r), min(mask.shape[2], x + r + 1)

    mask[z0:z1, y0:y1, x0:x1] = 1

    # 2) Convert mask -> sitk and copy geometry from source
    mask_img = sitk.GetImageFromArray(mask)  # creates (x,y,z) image internally, consistent with SITK
    mask_img.CopyInformation(source_volume)

    # 3) Resample into target grid using inverse transform (NN for mask)
    inv_t = transform.GetInverse()

    # Use sitk.Resample directly (reference image defines output origin/dir/size/spacing)
    out = sitk.Resample(
        mask_img,
        target_volume,                # reference image defines grid
        inv_t,
        sitk.sitkNearestNeighbor,
        0,                            # default pixel for outside
        sitk.sitkUInt8,
    )

    out_mask = sitk.GetArrayViewFromImage(out)

    # 4) Recover mapped point as robust center (median of indices where mask==1)
    idx = np.where(out_mask == 1)
    if idx[0].size == 0:
        raise ValueError(
            "Mapped mask is empty. The point may be outside target FOV or transform/grid mismatch."
        )

    mapped_zyx = np.median(np.vstack(idx), axis=1).astype(int).tolist()
    return mapped_zyx

