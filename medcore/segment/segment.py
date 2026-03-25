from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes, gaussian_filter, label
from skimage import measure
from skimage.morphology import remove_small_objects

from ..utils.sitk_utils import sitk_copy_metainfo, sitk_get_array


class TorsoSegmenter:
    def __init__(
        self,
        threshold_low: float = 0.3,
        threshold_high: float = 1.5,
        reference_area_ratio: float = 0.6,
        min_object_size: int = 100,
        slice_margin: int = 5,
        area_threshold_ratio: float = 0.8,
        kernel_shape: int = cv2.MORPH_ELLIPSE,
        kernel_size: Tuple[int, int] = (5, 5),
        morph_iterations: int = 1,
        smooth_sigma: float = 3.0,
        smooth_threshold: float = 0.75,
    ):
        """
        Initialize the TorsoSegmenter.

        Parameters
        ----------
        config : SegmentationConfig, optional
            Configuration parameters. If None, uses default values.
        """
        # parameters
        self.threshold_low = threshold_low  # 0.6
        self.threshold_high = threshold_high  # 1.5
        self.reference_area_ratio = reference_area_ratio  # 0.6
        self.min_object_size = min_object_size  # 100
        self.slice_margin = slice_margin  # 5
        self.area_threshold_ratio = area_threshold_ratio  # 0.8

        # parameters: morphological kernel
        self.kernel_shape = kernel_shape  # cv2.MORPH_ELLIPSE
        self.kernel_size = kernel_size  # (5,5)
        self.morph_iterations = morph_iterations  # 1

        # parameters: smoothing kernel
        self.smooth_sigma = smooth_sigma  # 3
        self.smooth_threshold = smooth_threshold  # 0.75

        self.reference_area = 0.0
        self.processing_range = (0, 0)

    def segment(
        self,
        volume: sitk.Image,
        return_image: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Step 0
        image = sitk_get_array(volume, normalize=True)

        # Step 1: Create initial mask
        initial_mask = self._create_initial_mask(image)

        # Step 2: Calculate reference area
        self._calculate_reference_area(initial_mask)

        # Step 3: Find processing range
        self._find_processing_range(image)

        # Step 4: Process slices
        torso_mask, contour_mask = self._process_slices(initial_mask)
        torso_volume = sitk_copy_metainfo(volume, torso_mask)

        # Step 5: Post-processing
        smoothed_mask = self._smooth_mask(torso_mask)
        if return_image:
            return torso_volume, contour_mask, smoothed_mask
        return torso_volume

    def _create_initial_mask(self, image: np.ndarray) -> np.ndarray:
        """Create initial binary mask using intensity thresholding."""
        mask_condition = np.logical_and(image > self.threshold_low, image < self.threshold_high)
        initial_mask = mask_condition.astype(np.uint8) * 255
        return initial_mask

    def _calculate_reference_area(self, initial_mask: np.ndarray) -> None:
        """Calculate reference area from representative slice."""
        height = initial_mask.shape[0]
        reference_slice = initial_mask[max(0, height - 10), ...]

        contours, _ = cv2.findContours(reference_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Safe fallback for near-empty scans.
            self.reference_area = 0.0
            return

        largest_contour = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(largest_contour)
        self.reference_area = self.reference_area_ratio * max_area

    def _find_processing_range(self, image: np.ndarray) -> None:
        """Find optimal slice range for processing using MIP analysis."""
        # Maximum Intensity Projection along depth axis
        mip_ap = image.max(axis=1)
        mip_ap[mip_ap < 1] = 0

        # Remove small objects
        mip_cleaned = remove_small_objects(
            mip_ap.astype(bool), min_size=self.min_object_size
        ).astype(int)

        # Find slice indices with significant content
        slice_sums = mip_cleaned.sum(axis=1)
        if slice_sums.max() == 0:
            # Fallback to middle portion
            height = image.shape[0]
            start_idx = height // 4
            end_idx = height - self.slice_margin
        else:
            sum_threshold = np.quantile(slice_sums, 0.5)
            valid_indices = np.where(slice_sums > sum_threshold)[0]
            if len(valid_indices) == 0:
                # Another fallback
                height = image.shape[0]
                start_idx = height // 4
                end_idx = height - self.slice_margin
            else:
                start_idx = max(0, valid_indices[0] + self.slice_margin)
                end_idx = min(image.shape[0], valid_indices[-1] - self.slice_margin)

        if end_idx <= start_idx:
            height = image.shape[0]
            start_idx, end_idx = 0, height
        self.processing_range = (start_idx, end_idx)

    def _process_slices(self, initial_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process individual slices to create segmentation masks."""
        torso_mask = np.zeros(initial_mask.shape, dtype=np.uint8)
        contour_mask = np.zeros(initial_mask.shape, dtype=np.uint8)

        start_idx, end_idx = self.processing_range
        area_threshold = self.reference_area * self.area_threshold_ratio

        for i in range(start_idx, end_idx):
            current_slice = initial_mask[i, ...]

            # Skip empty slices
            if current_slice.max() == 0:
                continue

            # Apply morphological operations
            opened_slice = self._morphology_opening(current_slice, area_threshold)
            # Fill holes
            filled_slice = binary_fill_holes(opened_slice).astype(np.uint8)
            # Find and filter contours
            contours, _ = cv2.findContours(filled_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on output masks
            filtered_contours = [
                contour for contour in contours if cv2.contourArea(contour) > self.reference_area
            ]
            cv2.drawContours(torso_mask[i, ...], filtered_contours, -1, 1, -1)
            cv2.drawContours(contour_mask[i, ...], filtered_contours, -1, 1, 2)

        return torso_mask, contour_mask

    def _morphology_opening(self, mask_img: np.ndarray, area_threshold: float) -> np.ndarray:
        """Apply morphological opening with area-based filtering."""
        # Erosion step

        kernel = cv2.getStructuringElement(self.kernel_shape, self.kernel_size)
        eroded = cv2.erode(mask_img, kernel, iterations=self.morph_iterations)

        # Label connected components and filter by area
        labels = measure.label(eroded)
        unique_labels, counts = np.unique(labels, return_counts=True)

        cleaned_labels = labels.copy()
        for label_val, count in zip(unique_labels, counts):
            if count <= area_threshold:
                cleaned_labels[labels == label_val] = 0

        # Convert to binary
        binary_mask = (cleaned_labels > 0).astype(np.uint8)

        # Dilation step (complete opening)
        dilated = cv2.dilate(binary_mask, kernel, iterations=self.morph_iterations + 1)

        return dilated

    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to the mask."""
        smoothed = gaussian_filter(mask.astype(float), sigma=self.smooth_sigma)
        return (smoothed >= self.smooth_threshold).astype(np.uint8)


class AbdomenSegmenter:
    """
    Comprehensive abdominal region detection and segmentation class.

    This class provides a complete pipeline for:
    1. Automatic abdominal region detection
    2. Region extraction
    3. Abdominal structure segme
    """

    def __init__(
        self,
        margin: int = 20,
        height_margin: int = 50,
        threshold_low: float = 0.3,
        threshold_high: float = 1.5,
        kernel_shape: int = cv2.MORPH_ELLIPSE,
        kernel_size: Tuple[int, int] = (3, 3),
        min_object_size: int = 100,
        window_size: int = 50,
        n_clusters: int = 2,
        gaussian_sigma: float = 2.0,
        contour_level: float = 0.75,
    ) -> None:
        """
        Initialize the AbdominalSegmenter

        Parameters
        ----------
        margin : int, default=10
            Margin to apply to depth/width boundaries (in pixels)
        height_margin : int, default=50
            Margin for height range (in pixels)
        threshold_low : float, default=0.6
            Lower intensity threshold for initial segmentation
        threshold_high : float, default=1.5
            Upper intensity threshold for initial segmentation
        kernel_shape : int, default=cv2.MORPH_ELLIPSE
            Shape of morphological kernel (cv2.MORPH_ELLIPSE, cv2.MORPH_RECT, etc.)
        kernel_size : Tuple[int, int], default=(3, 3)
            Size of morphological kernel (height, width)
        min_object_size : int, default=100
            Minimum size for connected components (in pixels)
        window_size : int, default=50
            Size of sliding window for height feature analysis
        n_clusters : int, default=2
            Number of clusters for K-means clustering
        gaussian_sigma : float, default=2.0
            Standard deviation for Gaussian blur
        contour_level : float, default=0.75
            Level for contour detection (0.0 to 1.0)
        """

        self.margin = margin
        self.height_margin = height_margin
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.kernel_shape = kernel_shape
        self.kernel_size = kernel_size
        self.min_object_size = min_object_size
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.gaussian_sigma = gaussian_sigma
        self.contour_level = contour_level

    def segment(
        self,
        volume: sitk.Image,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[np.ndarray]]:
        """
        Complete abdominal processing pipeline.

        Parameters
        ----------
        image : np.ndarray
            3D normalized CT image
        return_all : bool, default=True
            Whether to return all intermediate results

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        # Step 0
        image = sitk_get_array(volume, normalize=True)

        # Step 1: Detect abdominal region
        abdomen_region = self.detect_abdomen_region(image)

        # Step 2: Extract abdominal region
        abdominal_image = self.extract_abdominal_region(image, abdomen_region)

        # Step 3: Segment abdominal structures
        segmentation_mask, contour_mask, contour_list = self.segment_abdominal_region(
            abdominal_image
        )
        self.contour_mask = contour_mask

        return abdominal_image, segmentation_mask, abdomen_region, contour_list

    def detect_abdomen_region(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect abdominal region from 3D medical image using anatomical landmarks.

        Parameters
        ----------
        image : np.ndarray
            3D medical image with shape (height, depth, width)

        Returns
        -------
        Dict[str, Any]
            Dictionary containing abdominal region boundaries
        """
        height_margin = min(self.height_margin, max(1, image.shape[0] // 4))
        abdominal_region: Dict[str, Any] = {}

        # Step 1: Superior-Inferior (SI) projection
        mip_si = image.max(axis=0)  # Max projection along height axis
        mip_si[mip_si < 1] = 0
        nonzero_coords = np.where(mip_si > 0)
        row_coords, col_coords = nonzero_coords
        if row_coords.size == 0 or col_coords.size == 0:
            # Fallback to broad center crop when thresholding fails.
            abdominal_region["depth_start"] = 0
            abdominal_region["depth_end"] = image.shape[1]
            abdominal_region["width_start"] = 0
            abdominal_region["width_end"] = image.shape[2]
            abdominal_region["height_start"] = max(0, image.shape[0] // 2 - 100)
            abdominal_region["height_end"] = min(image.shape[0], image.shape[0] // 2 + 100)
            return abdominal_region

        # Define depth boundaries
        depth_range = np.linspace(row_coords.min(), row_coords.max(), 4).astype(int)
        abdominal_region["depth_start"] = max(0, depth_range[0] - self.margin)
        abdominal_region["depth_end"] = min(image.shape[1], depth_range[1])
        # Define width boundaries with margins
        width_range = np.linspace(col_coords.min(), col_coords.max(), 5).astype(int)
        abdominal_region["width_start"] = width_range[1] + self.margin
        abdominal_region["width_end"] = width_range[3] - self.margin

        # Define height boundaries
        mip_ap = image.max(axis=1)  # Max projection along depth axis
        mip_ap[mip_ap < 1] = 0
        height_intervals = self._search_height_range(mip_ap, height_margin)
        abdominal_region["height_start"] = height_intervals[0]
        abdominal_region["height_end"] = height_intervals[1]

        return abdominal_region

    def _search_height_range(self, image_mip: np.ndarray, margin: int) -> List[int]:
        image_mip = binary_fill_holes(image_mip).astype(int)
        width = image_mip.sum(1)
        if width.size == 0:
            return [0, 0]

        margin = int(max(1, min(margin, max(1, width.size // 4))))
        search_band = width[margin:-margin]
        if search_band.size == 0:
            center_loc = int(width.size // 2)
        else:
            center_loc = int(np.argmin(search_band) + margin)

        mask = np.abs(width - width[center_loc]) < margin
        labeled_array, num_features = label(mask)  # 연속된 True 영역 라벨링
        # 구간 추출
        intervals = []
        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            if center_loc in indices:
                intervals.append(max(margin, indices[0] - margin))
                intervals.append(min(image_mip.shape[0] - margin, indices[-1] + margin))
                break
        if len(intervals) == 0:
            pos = int(image_mip.shape[0] / 2)
            intervals.append(max(0, pos - 100))
            intervals.append(min(image_mip.shape[0], pos + 100))

        return intervals

    # Backward-compatible alias for previous typo.
    def _search_height_ragne(self, image_mip: np.ndarray, margin: int) -> List[int]:
        return self._search_height_range(image_mip, margin)

    def extract_abdominal_region(self, image: np.ndarray, region: Dict[str, Any]) -> np.ndarray:
        """
        Extract abdominal region from the full image.

        Parameters
        ----------
        image : np.ndarray
            Full 3D image
        region : Dict[str, Any]
            Abdominal region boundaries

        Returns
        -------
        np.ndarray
            Extracted abdominal region
        """
        abdominal_image = image[
            region["height_start"] : region["height_end"],
            region["depth_start"] : region["depth_end"],
            region["width_start"] : region["width_end"],
        ]

        return abdominal_image

    def segment_abdominal_region(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Segment abdominal structures from extracted region.

        Parameters
        ----------
        image : np.ndarray
            3D abdominal region image

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - segmentation_mask: Binary segmentation mask
            - contour_mask: Contour visualization mask
        """

        # Step 1: Create initial binary mask
        initial_mask = np.logical_and(
            image > self.threshold_low, image < self.threshold_high
        ).astype(np.uint8)

        # Remove small objects
        initial_mask_cleaned = remove_small_objects(
            initial_mask.astype(bool), min_size=self.min_object_size
        ).astype(int)

        # Fill holes
        filled_mask = binary_fill_holes(initial_mask_cleaned).astype(int)

        # Step 2: Initialize output masks
        segmentation_mask = np.zeros(filled_mask.shape, dtype=np.uint8)
        contour_mask = np.zeros(filled_mask.shape, dtype=np.uint8)

        # Step 3: Process each slice
        valid_slice_indices = np.where(filled_mask.max(axis=(1, 2)))[0]
        contour_list = []
        for i in valid_slice_indices:
            slice_mask = filled_mask[i, ...]

            # Remove small objects from slice
            slice_cleaned = remove_small_objects(
                slice_mask.astype(bool), min_size=self.min_object_size
            ).astype(int)

            mask, contour = self._fill_below_contour(slice_cleaned)
            segmentation_mask[i, ...] = mask
            # Add contour points to contour mask
            contour_int = contour.astype(int)
            if contour_int.size > 0:
                contour_mask[i, contour_int[:, 0], contour_int[:, 1]] = 1
                contour_list.append(contour_int)

        return segmentation_mask, contour_mask, contour_list

    def _fill_below_contour(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fill region below the detected contour in a 2D image."""

        height, width = image.shape
        mask = np.zeros_like(image, dtype=np.uint8)

        # Smooth the image
        image_uint8 = (255 * image).astype(np.uint8)
        smoothed = cv2.GaussianBlur(image_uint8, (0, 0), self.gaussian_sigma)
        if smoothed.max() > 1.0:
            smoothed = smoothed / 255.0

        # Find contour
        contours = measure.find_contours(smoothed, level=self.contour_level)
        if not contours:
            return mask, np.empty((0, 2), dtype=np.float32)
        # Select top-most contour
        min_locations = [c[:, 0].min() for c in contours]
        selected_contour = contours[np.argmin(min_locations)]

        # Convert (x,y -> y,x) and close contour
        contour_points = np.fliplr(np.round(selected_contour)).astype(np.int32)
        closed_contour = self._create_closed_contour(contour_points, width, height)

        cv2.fillPoly(mask, [closed_contour], 1)
        return mask, selected_contour

    def _create_closed_contour(
        self, contour_points: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Create a closed contour by connecting to image edges."""
        max_y = height - 1
        edge_points = np.array([[width - 1, max_y], [0, max_y], contour_points[0]])
        return np.vstack([contour_points, edge_points])
