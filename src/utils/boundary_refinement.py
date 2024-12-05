from typing import Tuple

import numpy as np
import skimage
import torch
from numba import njit
from scipy.sparse import csr_matrix
from torch import nn

# from .spectral_embedding import spectral_embedding


@njit
def intensity_profile(boundary_map: np.array, p1: np.array, p2: np.array):
    """ Compute the intensity profile between two points on a boundary map with the Bresenham
    algorithm.

    Parameters
    ----------
    boundary_map
    p1
    p2

    Returns
    -------

    """
    mirrorX = 1
    mirrorY = 1

    # transform p2 in relation to p1 that p1 is the center(0, 0)
    p2[0] -= p1[0]
    p2[1] -= p1[1]

    if p2[0] < 0:  # p2 is to the left of p1 (Octant 3 - 6) = > mirror on y-Axis
        p2[0] *= -1
        mirrorY = -1

    if p2[1] < 0:  # Octant 7 or 8 = > mirror on x-Axis
        p2[1] *= -1
        mirrorX = -1

    mirrorXY = False
    if p2[0] < p2[1]:  # Octant 2 (slope > 1) = > swap x and y
        tmp = p2[0]
        p2[0] = p2[1]
        p2[1] = tmp
        mirrorXY = True

    x = 0
    y = 0
    dx = p2[0]
    dy = p2[1]
    dNE = 2 * (dy - dx)
    dE = 2 * dy
    d = 2 * dy - dx

    intensity_profile = []
    intensity_profile.append(boundary_map[x, y])

    while x < p2[0]:
        if d >= 0:
            d += dNE
            x += 1
            y += 1
        else:
            d += dE
            x += 1
        if mirrorXY:  # swap x and y back, add offset p1 and mirror back
            intensity_profile.append(boundary_map[y * mirrorY + p1[0], x * mirrorX + p1[1]])
        else:
            intensity_profile.append(boundary_map[x * mirrorY + p1[0], y * mirrorX + p1[1]])

    return np.array(intensity_profile)


@njit
def compute_distance_matrix(boundary_map: np.array):
    height, width = boundary_map.shape
    neighbor_radius = 5

    row_col_data = []
    for i in range(height):
        for j in range(width):
            for dx in range(-neighbor_radius, neighbor_radius + 1):
                for dy in range(-neighbor_radius, neighbor_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    x1, y1 = i, j
                    x2, y2 = i + dx, j + dy

                    if 0 <= x2 < height and 0 <= y2 < width:
                        # Check if the line between the two pixels crosses an edge
                        ip = intensity_profile(boundary_map, np.array([x1, y1]),
                                               np.array([x2, y2]))
                        distance = np.max(ip)  # take maximum intensity along the line
                        pixel_index = x1 * width + y1
                        neighbor_index = x2 * width + y2
                        row_col_data.append((pixel_index, neighbor_index, distance))
                        row_col_data.append(
                            (neighbor_index, pixel_index, distance))  # make symmetric
                        # affinity_matrix[pixel_index, neighbor_index] = affinity
    return np.array([row for row, _, _ in row_col_data]), \
        np.array([col for _, col, _ in row_col_data]), \
        np.array([data for _, _, data in row_col_data])


def refine_boundary_map(boundary_map: torch.Tensor,
                        refine_boundary_size: Tuple[int, int]) -> torch.Tensor:
    beta = 1.0
    n_components = 8  # like in original paper

    # invert to get the actual boundary map where 1 is boundary and 0 is non-boundary
    boundary_map = 1 - boundary_map  # (B, H, W)

    # downsample to speed up computation
    boundary_map_original = boundary_map.detach().cpu().numpy()  # (B, H, W)
    boundary_map_original_size = boundary_map_original.shape[1:]
    boundary_map = nn.functional.interpolate(boundary_map.unsqueeze(1), size=refine_boundary_size,
                                             mode='area').squeeze(1)  # (B, H, W)

    device = boundary_map.device
    boundary_map = boundary_map.detach().cpu().numpy()  # (B, H, W)
    height, width = boundary_map.shape[1:]

    boundary_maps_refined = []
    for boundary_map_i, boundary_map_original_i in zip(boundary_map, boundary_map_original):
        row, col, data = compute_distance_matrix(boundary_map_i)
        distance_matrix = csr_matrix((data, (row, col)),
                                     shape=(height * width, height * width))  # (H * W, H * W)
        affinity_matrix = distance_matrix.copy()
        # Exponentiate to convert distances to similarities
        affinity_matrix.data = np.exp(
            -beta * affinity_matrix.data / affinity_matrix.data.std())  # (H * W, H * W)
        # affinity_matrix.data = np.exp(-beta * affinity_matrix.data)  # (H * W, H * W)
        eigenvalues, eig_maps = spectral_embedding(  # (n_components), (H * W, n_components)
            affinity_matrix,
            n_components=n_components,
            eigen_solver='arpack',
            random_state=0,
            eigen_tol=0.0,
            drop_first=True,
        )

        eig_maps = np.transpose(eig_maps, (1, 0))  # (n_components, H * W)
        eig_maps = eig_maps.reshape((n_components, height, width))  # (n_components, H, W)
        # Upsample to original size
        eig_maps = nn.functional.interpolate(torch.Tensor(eig_maps).unsqueeze(1),
                                             size=boundary_map_original_size,
                                             mode='bilinear').squeeze(
            1).cpu().numpy()  # (n_components, H, W)

        # for eig_map in eig_maps:
        #    plt.imshow(eig_map)
        #    plt.show()

        combined_eig_maps = np.zeros(boundary_map_original_size)  # (H, W)
        for eigenvalue, eig_map in zip(eigenvalues, eig_maps):
            eig_map_sobel = skimage.filters.sobel(eig_map)  # (H, W)
            combined_eig_maps += 1 / np.sqrt(eigenvalue) * eig_map_sobel

        # plt.imshow(combined_eig_maps)
        # plt.show()

        # combined_eig_maps_thresholded = combined_eig_maps > (0.05 * np.max(combined_eig_maps))
        # (H, W)
        # plt.imshow(combined_eig_maps_thresholded)
        # plt.show()

        # plt.imshow(boundary_map_original_i)
        # plt.show()

        boundary_map_i_refined = np.maximum(boundary_map_original_i,
                                            200 * combined_eig_maps)  # (H, W)
        # print(np.min(boundary_map_i_refined), np.max(boundary_map_i_refined))
        boundary_map_i_refined = np.clip(boundary_map_i_refined, 0, 1)  # (H, W)
        # plt.imshow(boundary_map_i_refined)
        # plt.show()

        boundary_map_i_refined = 1 - boundary_map_i_refined  # (H, W)
        boundary_maps_refined.append(boundary_map_i_refined)

    boundary_map = torch.Tensor(np.stack(boundary_maps_refined, axis=0)).to(device)  # (B, H, W)

    return boundary_map
