from numba import njit
import numpy as np
from scipy.sparse import csr_matrix


@njit
def intensity_profile(boundary_map: np.array, p1: np.array, p2: np.array):
    """ Compute the intensity profile between two points on a boundary map with the Bresenham algorithm.

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
def compute_distance_matrix(boundary_map: np.array, neighbor_radius: int):
    height, width = boundary_map.shape

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
                        row_col_data.append((neighbor_index, pixel_index, distance))  # make symmetric
                        # affinity_matrix[pixel_index, neighbor_index] = affinity
    return np.array([row for row, _, _ in row_col_data]),\
        np.array([col for _, col, _ in row_col_data]),\
        np.array([data for _, _, data in row_col_data])


def compute_affinity_matrix(boundary_map: np.array, beta: float, neighbor_radius: int) -> csr_matrix:
    # invert to get the actual boundary map where 1 is boundary and 0 is non-boundary
    boundary_map = 1 - boundary_map  # (H, W)

    height, width = boundary_map.shape

    row, col, data = compute_distance_matrix(boundary_map, neighbor_radius)

    """
    # Randomly sample x % of the connections to speed up computation
    random_subset = np.random.choice(num_connections, size=, replace=False)
    row = row[random_subset]
    col = col[random_subset]
    data = data[random_subset]
    """

    distance_matrix = csr_matrix((data, (row, col)), shape=(height * width, height * width))  # (H * W, H * W)
    affinity_matrix = distance_matrix.copy()
    # Exponentiate to convert distances to similarities
    affinity_matrix.data = np.exp(-beta * affinity_matrix.data)  # (H * W, H * W)
    return affinity_matrix