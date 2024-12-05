import os
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import scipy.ndimage
import torch
from PIL import Image
from pytorch_lightning.cli import LightningCLI
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import find_boundaries
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import Dataset
from utils.affinity_matrix import compute_affinity_matrix
from utils.instance_colors import COLORS
from utils.spectral_embedding import spectral_embedding

# Ignore seome torch warnings
warnings.filterwarnings('ignore', '.*The default behavior for interpolate/upsample with float*')
warnings.filterwarnings(
    'ignore', '.*Default upsampling behavior when mode=bicubic is changed to align_corners=False*')
warnings.filterwarnings('ignore', '.*Only one label was provided to `remove_small_objects`*')
warnings.filterwarnings('ignore', '.*does not have many workers which may be a bottleneck.*')
warnings.filterwarnings('ignore', '.*Graph is not fully connected.*')


class InstanceCluster(pl.LightningModule):
    """Panoptic fusion module that uses the semantic and boundary model to cluster semantic blobs
    into instances.

    Parameters
    ----------
    semantic_model : pl.LightningModule
        Semantic segmentation model.
    semantic_model_ckpt : str
        Path to the semantic segmentation model checkpoint.
    boundary_model : pl.LightningModule
        Boundary detection model.
    boundary_model_ckpt : str
        Path to the boundary detection model checkpoint.
    structure_connectivity : List[List[int]]
        Connectivity matrix for the CCA (scipy.ndimage.label) function.
    instance_min_pixel : int
        Minimum number of pixels for an instance to be considered.
    erosion_structure : List[List[int]]
        Structure for the binary erosion (scipy.ndimage.binary_erosion) function.
    erosion_iterations : int
        Number of iterations for the binary erosion (scipy.ndimage.binary_erosion) function.
    output_size : Tuple[int, int]
        Output size of panoptic segmentation.
    ignore_index : int
        Ignore index for the semantic segmentation.
    test_plot : bool
        Whether to plot the predictions during testing.
    test_save_dir : str
        Directory to save the predictions during testing.
    test_save_vis : bool
        Whether to save the prediction visualization as images during testing.
    debug_plot : bool
        Whether to plot the intermediate predictions during testing.
    """

    def __init__(self, semantic_model: pl.LightningModule, semantic_model_ckpt: str,
                 boundary_model: pl.LightningModule, boundary_model_ckpt: str,
                 do_post_processing: bool,
                 boundary_margin: int, boundary_min_pixel: int,
                 structure_connectivity: List[List[int]],
                 instance_min_pixel: List[int], mode: str,
                 output_size: Tuple[int, int],
                 upsample_factor_affinity_map: Optional[float], neighbor_radius_affinity_matrix: Optional[int],
                 beta: Optional[float], eigen_tol: Optional[float],
                 eigen_vec_hist_bins: Optional[int], ncut_threshold: Optional[float],
                 eigen_vec_hist_ratio: Optional[float], eigen_vec_thresholds: Optional[int],
                 threshold_boundary: Optional[float],
                 ignore_index: int = 255,
                 test_plot: bool = False, test_save_dir: str = None, test_save_vis: bool = False,
                 debug_plot: bool = False):
        super().__init__()
        self.semantic_model = semantic_model
        semantic_model_ckpt_dict = torch.load(semantic_model_ckpt, map_location='cpu')
        self.semantic_model.load_state_dict(semantic_model_ckpt_dict['state_dict'])
        self.semantic_model.on_load_checkpoint(semantic_model_ckpt_dict)

        self.boundary_model = boundary_model
        boundary_model_ckpt_dict = torch.load(boundary_model_ckpt, map_location='cpu')
        self.boundary_model.load_state_dict(boundary_model_ckpt_dict['state_dict'])
        self.boundary_model.on_load_checkpoint(boundary_model_ckpt_dict)

        # share encoder if the same vit model is used
        if self.semantic_model.vit_model == self.boundary_model.vit_model:
            self.boundary_model.encoder = self.semantic_model.encoder

        for param in self.semantic_model.parameters():  # freeze
            param.requires_grad = False
        for param in self.boundary_model.parameters():  # freeze
            param.requires_grad = False

        self.do_post_processing = do_post_processing
        self.boundary_margin = boundary_margin
        self.boundary_min_pixel = boundary_min_pixel
        self.structure_connectivity = np.array(structure_connectivity)
        self.instance_min_pixel = instance_min_pixel
        assert mode in ['cca', 'ncut']
        self.mode = mode
        self.output_size = output_size
        self.upsample_factor_affinity_map = upsample_factor_affinity_map
        self.neighbor_radius_affinity_matrix = neighbor_radius_affinity_matrix
        self.beta = beta
        self.eigen_tol = eigen_tol
        self.eigen_vec_hist_bins = eigen_vec_hist_bins
        self.ncut_threshold = ncut_threshold
        self.eigen_vec_hist_ratio = eigen_vec_hist_ratio
        self.eigen_vec_thresholds = eigen_vec_thresholds
        self.threshold_boundary = threshold_boundary
        self.ignore_index = ignore_index

        self.test_plot = test_plot
        self.test_save_dir = test_save_dir
        self.test_save_vis = test_save_vis
        self.debug_plot = debug_plot

        id_color_array = COLORS  # (CLASSES, 3)
        np.random.seed(0)
        id_color_array = np.random.permutation(id_color_array)
        id_color_array[0] = [0, 0, 0]  # background
        self.id_color_array = (id_color_array * 255).astype(np.uint8)

    def get_dataset(self) -> Dataset:
        dataset = self.trainer.test_dataloaders[0].dataset
        return dataset

    def add_instance(self, pred_sem_i, semantic_segment_mask, mask: np.array, pred_instances, number_of_instances: int,
                     semantic_class_id: int):
        mask_upsampled = torch.nn.functional.interpolate(
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            size=self.output_size,
            mode="nearest",
        ).squeeze().numpy().astype(bool)
        #semantic_class_mask = pred_sem_i == semantic_class_id

        # Add only connected masks
        mask_upsampled_segments = scipy.ndimage.label(mask_upsampled,
                                                      structure=self.structure_connectivity)[0]  # (H, W)
        mask_upsampled_segment_id, mask_upsampled_segment_size = np.unique(
            mask_upsampled_segments, return_counts=True)
        for s_id, s_size in zip(mask_upsampled_segment_id, mask_upsampled_segment_size):
            if s_id == 0:  # skip background
                continue
            if s_size >= self.instance_min_pixel[semantic_class_id]:  # Do not add if too small
                number_of_instances += 1
                pred_instances[np.logical_and(mask_upsampled_segments == s_id,
                                              semantic_segment_mask)] = number_of_instances
            else:
                pred_sem_i[np.logical_and(mask_upsampled_segments == s_id,
                                          semantic_segment_mask)] = self.ignore_index
                semantic_segment_mask[np.logical_and(mask_upsampled_segments == s_id,
                                          semantic_segment_mask)] = 0
        return pred_instances, pred_sem_i, semantic_segment_mask, number_of_instances

    def predict(self, rgb: torch.Tensor, rgb_original: torch.Tensor,
                ego_car_mask: Optional[torch.Tensor] = None) \
            -> Tuple[np.array, np.array, np.array]:
        pred_sem = self.semantic_model.predict(rgb, ego_car_mask)  # (B, H, W)
        device = pred_sem.device
        pred_sem = pred_sem.cpu().numpy()  # (B, H, W)

        pred_boundary = self.boundary_model.predict(rgb)  # (B, H, W)
        pred_boundary = pred_boundary.cpu().numpy()  # (B, H, W)

        if self.do_post_processing:
            # Semantic post-processing to remove some noise
            # 1) Remove visual artifacts at the image border
            pred_sem[:, :self.boundary_margin, :] = self.ignore_index
            pred_sem[:, -self.boundary_margin:, :] = self.ignore_index
            pred_sem[:, :, :self.boundary_margin] = self.ignore_index
            pred_sem[:, :, -self.boundary_margin:] = self.ignore_index

            # Boundary post-processing to remove some noise
            # 1) Remove visual artifacts at the image border
            pred_boundary[:, :self.boundary_margin, :] = 1
            pred_boundary[:, -self.boundary_margin:, :] = 1
            pred_boundary[:, :, :self.boundary_margin] = 1
            pred_boundary[:, :, -self.boundary_margin:] = 1
            # 2) Remove small boundaries that are "randomly" predicted
            pred_boundary_thresh = (pred_boundary > self.threshold_boundary)  # (B, H, W)
            pred_boundary_thresh_inv = 1 - pred_boundary_thresh
            for i, pred_boundary_thresh_inv_i in enumerate(pred_boundary_thresh_inv):
                pred_boundary_segments = scipy.ndimage.label(
                    pred_boundary_thresh_inv_i, structure=self.structure_connectivity)[0]
                boundary_id, boundary_size = np.unique(pred_boundary_segments, return_counts=True)
                for b_id, b_size in zip(boundary_id, boundary_size):
                    if b_size < self.boundary_min_pixel:
                        pred_boundary[i][pred_boundary_segments == b_id] = 1

        if ego_car_mask is None:
            ego_car_mask = torch.zeros_like(torch.Tensor(rgb_original[:, 0, :, :]))

        if self.mode == "ncut":
            rgb_h, rgb_w = rgb_original.shape[2:]
            affinity_map_size = [int(rgb_h // self.boundary_model.patch_size * self.upsample_factor_affinity_map),
                                 int(rgb_w // self.boundary_model.patch_size * self.upsample_factor_affinity_map)]

        pred_instances_batch = []
        pred_sem_batch = []
        for rgb_i, rgb_original_i, pred_sem_i, pred_boundary_i, ego_car_mask_i in \
                zip(rgb, rgb_original, pred_sem, pred_boundary, ego_car_mask):

            # rgb_original_i_t = rgb_original_i.transpose((1, 2, 0))  # (H, W, 3)
            # img_rgb = Image.fromarray(rgb_original_i_t.astype(np.uint8))
            # img_rgb.save("rgb.png")

            if self.debug_plot:
                self.semantic_model.plot(rgb_original_i, pred_sem_i)
                self.boundary_model.plot(rgb_original_i, pred_boundary_i)

            if self.mode == "ncut":
                pred_boundary_i_downsampled = nn.functional.interpolate(
                    torch.Tensor(pred_boundary_i).unsqueeze(0).unsqueeze(0), size=affinity_map_size,
                    mode='area').squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

                affinity_matrix = compute_affinity_matrix(pred_boundary_i_downsampled,
                                                          beta=self.beta,
                                                          neighbor_radius=self.neighbor_radius_affinity_matrix)  # (H * W, H * W)

            assert pred_sem_i.shape == tuple(self.output_size)
            pred_instances = np.zeros(self.output_size,
                                      dtype=int)  # to store the instance IDs (H, W)
            number_of_instances = 0

            if self.do_post_processing:
                # Threshold boundary map for post-processing
                pred_boundary_i = (pred_boundary_i > self.threshold_boundary).astype(float)

                # Create boundary map between thing and stuff classes and add to predicted boundary
                thing_map = np.isin(pred_sem_i, self.get_dataset().thing_classes)
                thing_map = remove_small_holes(thing_map, 1000)
                thing_map = remove_small_objects(thing_map, 1000)
                thing_stuff_boundary = find_boundaries(thing_map, connectivity=1, mode='thick')
                pred_boundary_i = 1 - np.clip(-pred_boundary_i + 1 + thing_stuff_boundary, 0, 1)
                pred_boundary[0] = pred_boundary_i

                if self.debug_plot:
                    self.boundary_model.plot(rgb_original_i, pred_boundary_i)

                pred_boundary_segments = \
                scipy.ndimage.label(pred_boundary_i, structure=self.structure_connectivity)[0]
                segment_id, segment_size = np.unique(pred_boundary_segments, return_counts=True)
                for s_id, s_size in zip(segment_id, segment_size):
                    # Skip the background and the boundary
                    if s_id not in [0, 1] and s_size < pred_boundary_i.size / 8:
                        semantic_classes = np.unique(pred_sem_i[pred_boundary_segments == s_id],
                                                     return_counts=True)
                        not_ignore_index = semantic_classes[0] != self.ignore_index
                        semantic_classes = (semantic_classes[0][not_ignore_index], semantic_classes[1][not_ignore_index])
                        if len(semantic_classes[0]) > 1:
                            max_class = semantic_classes[0][np.argmax(semantic_classes[1])]
                            # if np.max(semantic_classes[1]) > semantic_classes[1].sum() * 2 / 3:
                            mask = np.logical_and(pred_boundary_segments == s_id, pred_boundary_i == 1)
                            pred_sem_i[mask] = max_class
                            # for s_class, s_class_size in zip(semantic_classes[0], semantic_classes[1]):
                            #     if s_class_size < semantic_classes[1].sum() * .4:
                            #         mask = np.logical_and(pred_boundary_segments == s_id, pred_sem_i == s_class)
                            #         pred_sem_i[mask] = max_class

                if self.debug_plot:
                    self.semantic_model.plot(rgb_original_i, pred_sem_i)

            thing_classes = self.get_dataset().thing_classes
            for semantic_class_id in thing_classes:
                semantic_class_mask = pred_sem_i == semantic_class_id  # (H, W)
                if np.sum(semantic_class_mask) == 0:  # skip if thing class is not present
                    continue

                # Cluster into semantic segments/blobs with CCA
                semantic_segments_mask = scipy.ndimage.label(semantic_class_mask,
                                                             structure=self.structure_connectivity)[
                    0]  # (H, W)

                # if self.debug_plot:
                #     self.plot_instances(rgb_original_i, semantic_segments_mask)

                # Remove small instances from the semantic and the instance masks
                segment_id, segment_size = np.unique(semantic_segments_mask, return_counts=True)
                for s_id, s_size in zip(segment_id, segment_size):
                    if s_id == 0:  # skip background
                        continue
                    if s_size < self.instance_min_pixel[semantic_class_id]:
                        pred_sem_i[
                            semantic_segments_mask == s_id] = self.ignore_index
                        semantic_segments_mask[semantic_segments_mask == s_id] = 0

                # if self.debug_plot:
                #     self.plot_instances(rgb_original_i, semantic_segments_mask)
                #     self.semantic_model.plot(rgb_original_i, pred_sem_i)

                if self.do_post_processing:
                    # If a blob has no overlay with a boundary, remove it
                    for s_id in np.unique(semantic_segments_mask):
                        if s_id == 0:  # skip background
                            continue
                        if not np.logical_and(pred_boundary_i == 0,
                                              semantic_segments_mask == s_id).any():
                            pred_sem_i[
                                semantic_segments_mask == s_id] = self.ignore_index
                            semantic_segments_mask[semantic_segments_mask == s_id] = 0

                    # if self.debug_plot:
                    #     self.plot_instances(rgb_original_i, semantic_segments_mask)
                    #     self.semantic_model.plot(rgb_original_i, pred_sem_i)

                    # If a blob is fully surrounded by another thing class, remove it
                    for s_id in np.unique(semantic_segments_mask):
                        if s_id == 0:  # skip background
                            continue
                        segment_mask = semantic_segments_mask == s_id  # (H, W)
                        # Extend the segment mask by one pixel
                        segment_mask = binary_dilation(segment_mask)
                        # Semantic classes in this extended mask. Remove own and ignore classes
                        semantic_classes = np.unique(pred_sem_i[segment_mask])
                        semantic_classes = np.delete(semantic_classes, np.where(
                            semantic_classes == semantic_class_id))
                        semantic_classes = np.delete(semantic_classes, np.where(
                            semantic_classes == self.ignore_index))
                        # If all the surrounding pixel belong to the same thing class, remove the
                        # entire blob (or thing-like stuff classes)
                        if len(semantic_classes) == 1 and semantic_classes[0] in thing_classes:
                            pred_sem_i[segment_mask] = self.ignore_index
                            semantic_segments_mask[segment_mask] = 0

                    # if self.debug_plot:
                    #     self.plot_instances(rgb_original_i, semantic_segments_mask)
                    #     self.semantic_model.plot(rgb_original_i, pred_sem_i)

                for semantic_segment_id in np.unique(semantic_segments_mask):
                    if semantic_segment_id == 0:  # skip background
                        continue
                    semantic_segment_mask = (
                            semantic_segments_mask == semantic_segment_id)  # (H, W)

                    if self.mode == 'ncut':
                        semantic_segment_mask_original = semantic_segment_mask.copy()  # (H, W)
                        semantic_segment_mask_downsampled = torch.nn.functional.interpolate(
                            torch.from_numpy(semantic_segment_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                            size=affinity_map_size,
                            mode="nearest",
                        ).squeeze().numpy().astype(bool)  # (H, W)

                        ncut_queue = deque()
                        ncut_queue.append(semantic_segment_mask_downsampled)

                        instances_mask = np.zeros(semantic_segment_mask.shape, dtype=int)
                        number_of_instances_in_segment = 0

                        while len(ncut_queue) > 0:  # iterative Ncut
                            mask = ncut_queue.pop()
                            # If the mask is too small, do not add it
                            if np.sum(mask) < (self.instance_min_pixel[semantic_class_id] * (self.upsample_factor_affinity_map / self.semantic_model.encoder.patch_size)**2):
                                instances_mask, pred_sem_i, semantic_segment_mask,\
                                    number_of_instances_in_segment = self.add_instance(pred_sem_i,
                                                                                       semantic_segment_mask,
                                                                                       mask,
                                                                                       instances_mask,
                                                                                       number_of_instances_in_segment,
                                                                                       semantic_class_id)
                                continue

                            # Mask affinity matrix to only include pixels of the mask
                            affinity_matrix_semantic_segment = affinity_matrix[mask.reshape(-1), :]  # (K, W)
                            affinity_matrix_semantic_segment = affinity_matrix_semantic_segment[:,
                                                               mask.reshape(-1)]  # (K, K)

                            with cp.cuda.Device(device.index):
                                eigenvalues, eigen_maps = spectral_embedding(  # (n_components), (K * K, n_components)
                                    affinity_matrix_semantic_segment,
                                    n_components=1,
                                    eigen_solver="lobpcg",  # lobpcg, arpack
                                    random_state=0,
                                    eigen_tol=self.eigen_tol,
                                    drop_first=True,
                                )
                            cut_cost = eigenvalues[0]
                            # print(f"Cut cost: {cut_cost}")

                            eigen_vec = eigen_maps[:, 0]  # (K * K)
                            eigen_vec_sorted = np.sort(eigen_vec)
                            # plt.plot(eigen_vec_sorted)
                            # plt.show()

                            # Stability criterion from Ncut paper
                            eigen_vec_hist, eigen_vec_bins = np.histogram(eigen_vec[np.isfinite(eigen_vec)],
                                                                          bins=self.eigen_vec_hist_bins)
                            eigen_vec_hist_ratio = np.min(eigen_vec_hist) / np.max(eigen_vec_hist)
                            # print("Min / Max eigen vec hist: ", eigen_vec_hist_ratio)
                            # yes: 0.024 no: 0.36 => threshold between 0.024 and 0.36

                            if cut_cost > self.ncut_threshold or eigen_vec_hist_ratio > self.eigen_vec_hist_ratio:  # Do not cut if the cut cost is too high
                                instances_mask, pred_sem_i, semantic_segment_mask,\
                                    number_of_instances_in_segment = self.add_instance(pred_sem_i,
                                                                                       semantic_segment_mask,
                                                                                       mask,
                                                                                       instances_mask,
                                                                                       number_of_instances_in_segment,
                                                                                       semantic_class_id)
                                continue

                            eigen_map = np.zeros(mask.shape, dtype=np.float32)  # (H, W)
                            eigen_map[mask] = eigen_vec  # (H, W)

                            # Search for the threshold that minimizes the Ncut cost
                            eigen_vec_thresholds = np.linspace(np.min(eigen_vec), np.max(eigen_vec),
                                                               self.eigen_vec_thresholds)
                            eigen_vec_thresholds = eigen_vec_thresholds[1:-1]  # remove min and max
                            ncut_cost = np.inf
                            ncut_cost_threshold = None
                            for eigen_vec_threshold in eigen_vec_thresholds:
                                eigen_vec_cut = eigen_vec > eigen_vec_threshold  # (K * K)
                                cut_cost = affinity_matrix_semantic_segment[eigen_vec_cut, :] \
                                    [:, np.logical_not(eigen_vec_cut)].sum()
                                if cut_cost < ncut_cost:
                                    ncut_cost = cut_cost
                                    ncut_cost_threshold = eigen_vec_threshold

                            # eigen_vec_mean = np.mean(eigen_vec)  # (K * K)
                            eigen_vec_cut_0 = eigen_vec > ncut_cost_threshold  # (K * K)
                            eigen_vec_cut_1 = eigen_vec < ncut_cost_threshold  # (K * K)

                            eigen_map_cut_0 = np.zeros(mask.shape, dtype=bool)  # (H, W)
                            eigen_map_cut_1 = np.zeros(mask.shape, dtype=bool)  # (H, W)
                            eigen_map_cut_0[mask] = eigen_vec_cut_0  # (H, W)
                            eigen_map_cut_1[mask] = eigen_vec_cut_1  # (H, W)

                            ncut_queue.append(eigen_map_cut_0)
                            ncut_queue.append(eigen_map_cut_1)

                            # plt.imshow(eigen_map)
                            # plt.show()
                            # plt.imshow(eigen_map_cut_0)
                            # plt.show()
                            # plt.imshow(eigen_map_cut_1)
                            # plt.show()

                        pred_sem_i[np.logical_and(semantic_segment_mask_original,
                                   np.logical_not(instances_mask > 0))] = self.ignore_index

                    elif self.mode == 'cca':
                        # Subtract boundary from semantic segment
                        instances_mask = np.logical_and(semantic_segment_mask,
                                                        pred_boundary_i)  # (H, W)

                        # Cluster into instances with CCA
                        instances_mask = \
                            scipy.ndimage.label(instances_mask, structure=self.structure_connectivity)[
                                0]  # (H, W)
                        # ToDo: Look into min pixel size. This removes some detections.
                        # Remove small instances from the semantic and the instance masks
                        segment_id, segment_size = np.unique(instances_mask, return_counts=True)
                        if len(segment_id) > 1:
                            # instance_min_pixel = int(segment_size[1:].max() * .05)
                            instance_min_pixel = self.instance_min_pixel[semantic_class_id]
                            for s_id, s_size in zip(segment_id, segment_size):
                                if s_id == 0:  # skip background
                                    continue
                                if s_size < instance_min_pixel:
                                    pred_sem_i[instances_mask == s_id] = self.ignore_index
                                    semantic_segment_mask[instances_mask == s_id] = 0
                                    instances_mask[instances_mask == s_id] = 0

                        # if self.debug_plot:
                        #     self.plot_instances(rgb_original_i, instances_mask)
                        #     self.semantic_model.plot(rgb_original_i, pred_sem_i)

                        # if no large enough instance is found through the boundary estimation,
                        # use the whole semantic segment as one instance
                        if np.sum(instances_mask) == 0:  # TODO: remove this?
                             instances_mask[semantic_segment_mask] = 1

                        instances_ids = np.unique(instances_mask)
                        for i in range(1, len(instances_ids)):  # renumber instances to be consecutive
                            instances_mask[instances_mask == instances_ids[i]] = i

                        # if semantic has no instance (on boundary), add them to the nearest instance with 1-NN
                        assert semantic_segment_mask.shape == instances_mask.shape
                        coordinates = np.indices(
                            (self.output_size[0], self.output_size[1])).reshape(2, -1).T  # (H * W, 2)
                        coordinates_sem_seg = coordinates[
                            semantic_segment_mask.reshape(-1) == 1]  # (M, 2)
                        coordinates_instances = coordinates[instances_mask.reshape(-1) != 0]  # (N, 2)

                        knn = KNeighborsClassifier(n_neighbors=1)
                        knn.fit(coordinates_instances,
                                instances_mask.reshape(-1)[instances_mask.reshape(-1) != 0])
                        instances_mask_shape = instances_mask.shape
                        instances_mask = instances_mask.reshape(-1)
                        instances_mask[semantic_segment_mask.reshape(-1) == 1] = \
                            knn.predict(coordinates_sem_seg)
                        instances_mask = instances_mask.reshape(instances_mask_shape)  # (H, W)

                    instances_mask += number_of_instances
                    instances_mask[instances_mask == number_of_instances] = 0

                    pred_instances += instances_mask
                    number_of_instances = np.max(pred_instances)

            if self.debug_plot:
                self.plot_instances(rgb_original_i, pred_instances)
                self.semantic_model.plot(rgb_original_i, pred_sem_i)

            if self.do_post_processing:
                stuff_classes = self.get_dataset().stuff_classes_without_thing_like_classes
                background_classes = self.get_dataset().background_classes
                # ToDo: How to preserve semantic areas of small stuff classes?
                for semantic_class_id in stuff_classes:
                    semantic_class_mask = pred_sem_i == semantic_class_id  # (H, W)
                    if np.sum(semantic_class_mask) == 0:  # skip if stuff class is not present
                        continue

                    # Cluster into semantic segments/blobs with CCA
                    semantic_segments_mask = scipy.ndimage.label(semantic_class_mask,
                                                                 structure=self.structure_connectivity)[
                        0]  # (H, W)

                    # if self.debug_plot:
                    #     self.plot_instances(rgb_original_i, semantic_segments_mask)

                    # Remove small instances from the semantic and the instance masks
                    segment_id, segment_size = np.unique(semantic_segments_mask, return_counts=True)
                    for s_id, s_size in zip(segment_id, segment_size):
                        if s_id == 0:  # skip background
                            continue
                        if s_size < self.instance_min_pixel[semantic_class_id]:
                            pred_sem_i[
                                semantic_segments_mask == s_id] = self.ignore_index
                            semantic_segments_mask[semantic_segments_mask == s_id] = 0

                    # if self.debug_plot:
                    #     self.plot_instances(rgb_original_i, semantic_segments_mask)
                    #     self.semantic_model.plot(rgb_original_i, pred_sem_i)

                    # Remove blobs that are surrounded by thing classes
                    for s_id in np.unique(semantic_segments_mask):
                        if s_id == 0:  # skip background
                            continue
                        segment_mask = semantic_segments_mask == s_id  # (H, W)

                        # Skip if the segment mask touches the image border and is background class
                        border_mask = np.ones_like(segment_mask)
                        border_mask[self.boundary_margin + 1:-self.boundary_margin + 1,
                                    self.boundary_margin + 1:-self.boundary_margin + 1] = 0
                        if np.sum(segment_mask * border_mask) > 0 and semantic_class_id in background_classes:
                            continue

                        # Extend the segment mask by one pixel
                        segment_mask = binary_dilation(segment_mask)
                        # Semantic classes in this extended mask. Remove own and ignore classes
                        semantic_classes = np.unique(pred_sem_i[segment_mask])
                        semantic_classes = np.delete(semantic_classes, np.where(
                            semantic_classes == semantic_class_id))
                        semantic_classes = np.delete(semantic_classes, np.where(
                            semantic_classes == self.ignore_index))
                        # If all the surrounding pixel belong to the same thing class, remove the
                        # entire blob (or thing-like stuff classes)
                        if len(semantic_classes) == 1 and semantic_classes[0] in thing_classes:
                            pred_sem_i[segment_mask] = self.ignore_index
                            semantic_segments_mask[segment_mask] = 0

            if self.debug_plot:
                self.semantic_model.plot(rgb_original_i, pred_sem_i)

            # Fill the resulting holes in both the semantic and the instance predictions
            unknown_mask = np.logical_and(pred_sem_i == self.ignore_index,
                                          np.logical_not(ego_car_mask_i.cpu().numpy()))
            known_mask = np.logical_and(pred_sem_i != self.ignore_index,
                                        np.logical_not(ego_car_mask_i.cpu().numpy()))
            coordinates = np.indices((self.output_size[0], self.output_size[1])).reshape(2, -1).T
            coordinates_known = coordinates[known_mask.reshape(-1) == 1]
            coordinates_unknown = coordinates[unknown_mask.reshape(-1) == 1]
            knn = KNeighborsClassifier(n_neighbors=1)
            fit_data = np.stack((pred_sem_i.reshape(-1)[known_mask.reshape(-1) == 1],
                                 pred_instances.reshape(-1)[known_mask.reshape(-1) == 1])).T
            knn.fit(coordinates_known, fit_data)
            if len(coordinates_unknown) > 0:
                predict_data = knn.predict(coordinates_unknown).T
                pred_sem_i.reshape(-1)[unknown_mask.reshape(-1)], pred_instances.reshape(-1)[
                    unknown_mask.reshape(-1)] = predict_data

            if self.debug_plot:
                self.plot_instances(rgb_original_i, pred_instances)
                self.semantic_model.plot(rgb_original_i, pred_sem_i)

            # Make sure that only thing classes have an instance ids
            stuff_classes_mask = np.isin(pred_sem_i, self.get_dataset().stuff_classes)
            pred_instances[stuff_classes_mask] = 0

            thing_classes_mask = np.isin(pred_sem_i, self.get_dataset().thing_classes)
            assert np.all(thing_classes_mask == (pred_instances != 0))

            pred_instances_batch.append(pred_instances)
            pred_sem_batch.append(pred_sem_i)

        pred_instances = np.stack(pred_instances_batch, axis=0)  # (B, H, W)
        pred_sem = np.stack(pred_sem_batch, axis=0)  # (B, H, W)

        return pred_instances, pred_sem, pred_boundary

    def plot_instances(self, rgb: np.array, instances: np.array):
        fig = plt.figure(figsize=(10, 6))
        # plt.figure(figsize=(10, 6))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(10, 10)

        rgb = rgb.transpose((1, 2, 0))  # (H, W, 3)
        instances_color = self.id_color_array[instances, :]  # (H, W, 3)

        # plt.subplot(1, 1, 1)
        # plt.axis('off')
        # plt.grid(False)
        # plt.imshow(rgb)
        #
        # plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(rgb)
        plt.imshow(instances_color, cmap='jet', alpha=0.5, interpolation='nearest')
        fig.tight_layout()
        plt.show()

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        rgb = batch['rgb']  # (B, 3, H, W)
        rgb_original = batch['rgb_original']  # (B, 3, H, W)
        rgb_original = rgb_original.cpu().numpy()  # (B, 3, H, W)
        ego_car_mask = batch.get('ego_car_mask', None)  # (B, H, W)

        # Skip if mask already exists
        if self.test_save_dir is not None:
            semantic_path_i = batch['semantic_path'][0]
            dataset_path_base = str(self.get_dataset().path_base)
            pred_sem_i_path = semantic_path_i.replace(dataset_path_base, self.test_save_dir)
            pred_sem_i_path = pred_sem_i_path.replace('.mat', '.png')
            if os.path.exists(pred_sem_i_path):
                return
            os.makedirs(os.path.dirname(pred_sem_i_path), exist_ok=True)
            open(pred_sem_i_path, 'w').close()  # Dummy file to indicate that the file is being prepared

        pred_instances, pred_sem, pred_boundary = self.predict(rgb, rgb_original, ego_car_mask)
        # pred_sem = batch['semantic'].cpu().numpy()
        # pred_boundary = [None]
        # a = batch['instance'].cpu().numpy()
        # pred_instances = scipy.stats.rankdata(a, 'dense').reshape(a.shape) - 1
        # import cv2
        # pred_sem = np.expand_dims(cv2.resize(pred_sem[0], (2048, 1024), interpolation=cv2.INTER_NEAREST), 0)
        # pred_instances = np.expand_dims(cv2.resize(pred_instances[0], (2048, 1024), interpolation=cv2.INTER_NEAREST), 0)

        # Assert that all pixel of the thing classes are assigned to an instance
        for pred_instances_i, pred_sem_i in zip(pred_instances, pred_sem):
            assert pred_sem_i.shape == pred_instances_i.shape
            thing_classes_mask = np.isin(pred_sem_i, self.get_dataset().thing_classes)
            assert np.all(thing_classes_mask == (pred_instances_i != 0))

        if self.test_plot:
            for rgb_original_i, pred_sem_i, pred_instances_i in zip(rgb_original, pred_sem,
                                                                    pred_instances):
                self.plot_instances(rgb_original_i, pred_instances_i)
                self.semantic_model.plot(rgb_original_i, pred_sem_i)

        if self.test_save_dir is not None:
            semantic_path = batch['semantic_path']
            instance_path = batch['instance_path']
            dataset = self.get_dataset()
            dataset_path_base = str(dataset.path_base)

            for pred_sem_i, pred_instances_i, pred_boundary_i, semantic_path_i, instance_path_i in \
                    zip(pred_sem, pred_instances, pred_boundary, semantic_path, instance_path):

                pred_sem_i_gt_format, pred_panoptic_i_gt_format = \
                    dataset.compute_panoptic_label_in_gt_format(pred_sem_i, pred_instances_i)

                pred_sem_i_path = semantic_path_i.replace(dataset_path_base, self.test_save_dir)
                pred_sem_i_path = pred_sem_i_path.replace('.mat', '.png')
                if not os.path.exists(os.path.dirname(pred_sem_i_path)):
                    os.makedirs(os.path.dirname(pred_sem_i_path))
                pred_img = Image.fromarray(pred_sem_i_gt_format.astype(np.uint8))
                pred_img.save(pred_sem_i_path)

                pred_panoptic_i_path = instance_path_i.replace(dataset_path_base,
                                                               self.test_save_dir)
                pred_panoptic_i_path = pred_panoptic_i_path.replace('.mat', '.png')
                if not os.path.exists(os.path.dirname(pred_panoptic_i_path)):
                    os.makedirs(os.path.dirname(pred_panoptic_i_path))
                pred_img = Image.fromarray(pred_panoptic_i_gt_format.astype(np.uint16))
                pred_img.save(pred_panoptic_i_path)

                if self.test_save_vis:
                    pred_sem_i_color = self.get_dataset().class_id_to_color()[pred_sem_i,
                                       :]  # (H, W, 3)
                    pred_ins_i_color = self.id_color_array[pred_instances_i, :]  # (H, W, 3)
                    pred_panop_i_color = np.zeros_like(pred_sem_i_color)
                    pred_panop_i_color[pred_instances_i == 0, :] = pred_sem_i_color[
                                                                   pred_instances_i == 0, :]
                    pred_panop_i_color[pred_instances_i != 0, :] = pred_ins_i_color[
                                                                   pred_instances_i != 0, :]

                    pred_img = Image.fromarray(pred_sem_i_color)
                    pred_sem_i_color_path = pred_sem_i_path.replace('.png', '_color.png')
                    pred_sem_i_color_path = pred_sem_i_color_path.replace('.npy', '_color.png')
                    pred_img.save(pred_sem_i_color_path)

                    pred_img = Image.fromarray(pred_panop_i_color)
                    pred_panop_i_color_path = pred_panoptic_i_path.replace('.png', '_color.png')
                    pred_panop_i_color_path = pred_panop_i_color_path.replace('.npy', '_color.png')
                    pred_img.save(pred_panop_i_color_path)

                    if pred_boundary_i is not None:
                        pred_boundary_i_path = pred_panoptic_i_path.replace('.png', '_boundary.png')
                        pred_boundary_i_path = pred_boundary_i_path.replace('.npy', '_boundary.png')
                        pred_boundary_i_path = pred_boundary_i_path.replace('.mat', '_boundary.png')
                        pred_boundary_img = Image.fromarray(
                            pred_boundary_i.astype(np.uint8) * 255).convert(
                            'RGB')
                        pred_boundary_img.save(pred_boundary_i_path)

                        pred_boundary_img = pred_boundary_img.convert('RGBA')
                        width, height = pred_boundary_img.size
                        pixdata = pred_boundary_img.load()
                        for y in range(height):
                            for x in range(width):
                                if pixdata[x, y] == (255, 255, 255, 255):
                                    pixdata[x, y] = (255, 255, 255, 0)
                                else:
                                    pixdata[x, y] = (0, 0, 0, 255)
                        pred_img.paste(pred_boundary_img, (0, 0), pred_boundary_img)
                        pred_boundary_i_path = pred_boundary_i_path.replace('_boundary.png',
                                                                            '_boundary_blend.png')
                        pred_img.save(pred_boundary_i_path)


class InstanceClusterCLI(LightningCLI):

    def __init__(self):
        super().__init__(
            model_class=InstanceCluster,
            seed_everything_default=0,
            parser_kwargs={'parser_mode': 'omegaconf'},
            save_config_callback=None,
        )

    def add_arguments_to_parser(self, parser):
        # Dataset
        parser.add_argument('--data_params', type=Dict)


if __name__ == '__main__':
    cli = InstanceClusterCLI()
