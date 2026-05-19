from __future__ import annotations

from typing import Sequence, Union

import networkx as nx
import numpy as np
import SimpleITK as sitk
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from ..utils.sitk_utils import sitk_create_shell_mask, sitk_resampler


class RegionCenterlinePartitioner:
    """
    Partition a binary region into subregions along its centerline.

    Workflow
    --------
    - Skeletonize the foreground mask.
    - Extract the longest endpoint-to-endpoint centerline path.
    - Smooth and resample the centerline in physical space.
    - Assign each foreground voxel to the nearest centerline sample.
    - Label voxels by cumulative centerline length cutoffs.
    """

    def __init__(
        self,
        mask_volume: sitk.Image,
        cutoffs: Union[float, Sequence[float]] = (0.25, 0.75),
        target_spacing_mm: float = 1.0,
        smoothing: float = 10.0,
        shell_inner_mm: float = 5.0,
        shell_outer_mm: float = 10.0,
    ) -> None:
        self.cutoffs = self._validate_cutoffs(cutoffs)
        self.target_spacing_mm = float(target_spacing_mm)
        self.smoothing = float(smoothing)
        self.shell_inner_mm = float(shell_inner_mm)
        self.shell_outer_mm = float(shell_outer_mm)

        if self.target_spacing_mm <= 0:
            raise ValueError("`target_spacing_mm` must be > 0.")
        if self.smoothing < 0:
            raise ValueError("`smoothing` must be >= 0.")

        self.mask_volume = mask_volume
        self.mask = sitk.GetArrayFromImage(mask_volume) > 0
        self.spacing_zyx = np.asarray(mask_volume.GetSpacing()[::-1], dtype=np.float64)
        self.skeleton_volume: sitk.Image | None = None
        self.skeleton_spacing_zyx = self.spacing_zyx
        self._skeleton_uses_resampled_grid = False

        self.skeleton: np.ndarray | None = None
        self.raw_centerline_skeleton: np.ndarray | None = None
        self.raw_centerline: np.ndarray | None = None
        self.smooth_centerline: np.ndarray | None = None
        self.region_label: np.ndarray | None = None
        self.cumulative_length: np.ndarray | None = None
        self.centerline_length: float | None = None
        self.shell_volume: sitk.Image | None = None
        self.shell_region_volume: sitk.Image | None = None

    @staticmethod
    def _validate_cutoffs(cutoffs: Union[float, Sequence[float]]) -> tuple[float, ...]:
        if isinstance(cutoffs, (float, int)):
            cutoffs = (float(cutoffs),)
        values = tuple(sorted(float(v) for v in cutoffs))

        if not values:
            return values
        if not all(np.isfinite(values)):
            raise ValueError("`cutoffs` must contain finite values.")
        if any(v <= 0.0 or v >= 1.0 for v in values):
            raise ValueError("`cutoffs` must be between 0 and 1.")
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("`cutoffs` must be unique.")
        return values

    def create_skeleton(self) -> np.ndarray:
        if not np.any(self.mask):
            raise ValueError("`mask_volume` must contain at least one foreground voxel.")

        spacing = np.array(self.mask_volume.GetSpacing(), dtype=float)
        if np.allclose(spacing, spacing[0]):
            self.skeleton = skeletonize(self.mask)
            self.skeleton_volume = self.mask_volume
            self.skeleton_spacing_zyx = self.spacing_zyx
            self._skeleton_uses_resampled_grid = False
        else:
            binary_mask_volume = sitk.Cast(self.mask_volume > 0, sitk.sitkUInt8)
            mask_volume_rsl = sitk_resampler(
                binary_mask_volume,
                new_spacing=(1.0, 1.0, 1.0),
                interpolation="nn",
                default_pixel=0,
            )
            mask_rsl = sitk.GetArrayFromImage(mask_volume_rsl) > 0
            self.skeleton = skeletonize(mask_rsl)
            self.skeleton_volume = mask_volume_rsl
            self.skeleton_spacing_zyx = np.asarray(
                mask_volume_rsl.GetSpacing()[::-1],
                dtype=np.float64,
            )
            self._skeleton_uses_resampled_grid = True

        return self.skeleton

    def skeleton_to_graph(self, skeleton: np.ndarray) -> nx.Graph:
        coords = np.argwhere(skeleton > 0)
        if coords.size == 0:
            raise ValueError("`skeleton` must contain at least one foreground voxel.")

        coord_list = sorted(tuple(int(v) for v in coord) for coord in coords)
        coord_set = set(coord_list)
        graph = nx.Graph()
        graph.add_nodes_from(coord_list)
        offsets = [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ]

        spacing_zyx = np.asarray(self.skeleton_spacing_zyx, dtype=np.float64)
        for coord in coord_list:
            for offset in offsets:
                neighbor = (
                    coord[0] + offset[0],
                    coord[1] + offset[1],
                    coord[2] + offset[2],
                )
                if neighbor in coord_set:
                    weight = float(np.linalg.norm(np.asarray(offset) * spacing_zyx))
                    graph.add_edge(coord, neighbor, weight=weight)

        return graph

    def extract_longest_centerline(self, skeleton: np.ndarray) -> np.ndarray:
        graph = self.skeleton_to_graph(skeleton)
        endpoints = [node for node in graph.nodes if graph.degree[node] == 1]
        if len(endpoints) < 2:
            raise ValueError("Not enough skeleton endpoints were found.")

        max_len = -1.0
        best_path: list[tuple[int, int, int]] | None = None
        for idx, source in enumerate(endpoints[:-1]):
            lengths, paths = nx.single_source_dijkstra(graph, source, weight="weight")
            for target in endpoints[idx + 1 :]:
                if target in lengths and lengths[target] > max_len:
                    max_len = float(lengths[target])
                    best_path = paths[target]

        if best_path is None:
            raise ValueError("No connected endpoint path was found in the skeleton.")

        self.raw_centerline_skeleton = np.asarray(best_path, dtype=np.float64)
        self.raw_centerline = self._map_centerline_to_original_indices(
            self.raw_centerline_skeleton
        )
        return self.raw_centerline

    def compute_length(self, centerline_zyx: np.ndarray) -> float:
        points = self._validate_centerline(centerline_zyx)
        points_mm = points * self.spacing_zyx
        diffs = np.diff(points_mm, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def smooth_spline(self, centerline_zyx: np.ndarray) -> np.ndarray:
        points = self._validate_centerline(centerline_zyx)
        points_mm = points * self.spacing_zyx
        diffs = np.diff(points_mm, axis=0)
        keep = np.concatenate([[True], np.linalg.norm(diffs, axis=1) > 0])
        points_mm = points_mm[keep]

        if len(points_mm) < 2:
            raise ValueError("Spline fitting requires at least 2 unique points.")

        diffs = np.diff(points_mm, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        u_raw = np.concatenate([[0.0], np.cumsum(distances)])
        length = float(u_raw[-1])
        if length <= 0:
            raise ValueError("Centerline length is zero.")

        self.centerline_length = length
        n_points = max(2, int(np.ceil(length / self.target_spacing_mm)) + 1)
        u = u_raw / length
        k = min(3, len(points_mm) - 1)

        tck, _ = splprep(
            [points_mm[:, 0], points_mm[:, 1], points_mm[:, 2]],
            u=u,
            s=self.smoothing,
            k=k,
        )
        z_s, y_s, x_s = splev(np.linspace(0.0, 1.0, n_points), tck)
        smooth_mm = np.vstack([z_s, y_s, x_s]).T
        self.smooth_centerline = smooth_mm / self.spacing_zyx
        return self.smooth_centerline

    def create_shell_mask(
        self,
        inner_mm: float | None = None,
        outer_mm: float | None = None,
    ) -> sitk.Image:
        inner_mm = self.shell_inner_mm if inner_mm is None else float(inner_mm)
        outer_mm = self.shell_outer_mm if outer_mm is None else float(outer_mm)
        self.shell_volume = sitk_create_shell_mask(
            self.mask_volume,
            inner_mm=inner_mm,
            outer_mm=outer_mm,
        )
        return self.shell_volume

    def create_region_label(
        self,
        centerline_zyx: np.ndarray | None = None,
        volume: sitk.Image | None = None,
    ) -> sitk.Image:
        if volume is None:
            volume = self.mask_volume
            mask = self.mask
        else:
            mask = sitk.GetArrayFromImage(volume) > 0

        if centerline_zyx is None:
            centerline_zyx = self.smooth_centerline
        if centerline_zyx is None:
            raise ValueError("No centerline. Run `smooth_spline()` first.")

        centerline = self._validate_centerline(centerline_zyx)
        label_dtype = np.uint16 if len(self.cutoffs) + 1 > np.iinfo(np.uint8).max else np.uint8
        region_label = np.zeros(mask.shape, dtype=label_dtype)
        voxels_zyx = np.argwhere(mask)
        if voxels_zyx.size == 0:
            return self._array_to_label_image(region_label, volume)

        voxels_mm = voxels_zyx * self.spacing_zyx
        centerline_mm = centerline * self.spacing_zyx
        diffs = np.diff(centerline_mm, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
        if cum_len[-1] <= 0:
            raise ValueError("Centerline length is zero.")

        s = cum_len / cum_len[-1]
        self.cumulative_length = s

        tree = cKDTree(centerline_mm)
        _, nearest_idx = tree.query(voxels_mm, k=1)
        voxel_s = s[nearest_idx]

        bins = (0.0,) + self.cutoffs + (1.0,)
        labels = np.zeros(len(voxel_s), dtype=label_dtype)
        for idx in range(len(bins) - 1):
            lo = bins[idx]
            hi = bins[idx + 1]
            if idx == len(bins) - 2:
                selected = (voxel_s >= lo) & (voxel_s <= hi)
            else:
                selected = (voxel_s >= lo) & (voxel_s < hi)
            labels[selected] = idx + 1

        region_label[tuple(voxels_zyx.T)] = labels
        self.region_label = region_label
        return self._array_to_label_image(region_label, volume)

    def run(self, skeleton: np.ndarray | None = None) -> sitk.Image:
        if skeleton is None:
            skeleton = self.create_skeleton()
        else:
            self.skeleton = skeleton
            self.skeleton_volume = self.mask_volume
            self.skeleton_spacing_zyx = self.spacing_zyx
            self._skeleton_uses_resampled_grid = False

        raw = self.extract_longest_centerline(skeleton)
        smooth = self.smooth_spline(raw)
        return self.create_region_label(centerline_zyx=smooth)

    @staticmethod
    def _validate_centerline(centerline_zyx: np.ndarray) -> np.ndarray:
        points = np.asarray(centerline_zyx, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("`centerline_zyx` must have shape (N, 3).")
        if len(points) < 2:
            raise ValueError("`centerline_zyx` must contain at least 2 points.")
        if not np.all(np.isfinite(points)):
            raise ValueError("`centerline_zyx` must contain finite values.")
        return points

    def _map_centerline_to_original_indices(self, centerline_zyx: np.ndarray) -> np.ndarray:
        points = self._validate_centerline(centerline_zyx)
        if not self._skeleton_uses_resampled_grid:
            return points
        if self.skeleton_volume is None:
            raise RuntimeError("`skeleton_volume` is not set.")

        mapped = np.zeros_like(points, dtype=np.float64)
        for idx, point_zyx in enumerate(points):
            skeleton_xyz = tuple(float(v) for v in point_zyx[::-1])
            physical = self.skeleton_volume.TransformContinuousIndexToPhysicalPoint(
                skeleton_xyz
            )
            original_xyz = self.mask_volume.TransformPhysicalPointToContinuousIndex(physical)
            mapped[idx] = np.asarray(original_xyz[::-1], dtype=np.float64)

        return mapped

    @staticmethod
    def _array_to_label_image(label: np.ndarray, reference: sitk.Image) -> sitk.Image:
        label_volume = sitk.GetImageFromArray(label)
        label_volume.CopyInformation(reference)
        return label_volume
