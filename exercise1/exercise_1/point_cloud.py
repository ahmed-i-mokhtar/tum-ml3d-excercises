"""Triangle Meshes to Point Clouds"""
import numpy as np


def compute_all_triangle_areas(v1, v2, v3):
    """
    Compute the area of multiple triangles.
    :param v1, v2, v3: (N,3) numpy arrays,
        Each row of v_i contains (x, y, z) coordinates of the vertices
    :return: areas of all triangles of the provided shape, numpy array of size N
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


def sample_all_triangles(v1, v2, v3):
    """
    Sample all triangles using barycentric coordinates
    :param v1, v2, v3: (N,3) numpy arrays,
        Each row of v_i contains (x, y, z) coordinates of the vertices
    :return: point P for each provided triangle computed by using barycentric coordinates,  
        numpy array of size (N,3)
    """
    N = v1.shape[0]
    r = np.random.rand(N,2)
    r_1 = r[:,0]
    r_2 = r[:,1]

    u = 1 - np.sqrt(r_1)
    u = np.stack([u, u, u], axis=1)
    v = np.sqrt(r_1) * (1 - r_2)
    v = np.stack([v, v, v], axis=1)
    w = np.sqrt(r_1) * r_2 
    w = np.stack([w, w, w], axis=1)

    P = u*v1 + v*v2 + w*v3
    
    return P

def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    # ###############
    # TODO: Implement
    # ###############
    # compute triangle areas
    # compute probabilities for each triangle
    # sample triangles according to their probability
    # sample points on each sampled triangle
    # return sampled points
    triangle_areas = compute_all_triangle_areas(vertices[faces[:,0]], vertices[faces[:,1]], vertices[faces[:,2]])
    triangle_probabilities = triangle_areas / np.sum(triangle_areas)
    triangle_indices = np.random.choice(faces.shape[0], n_points, p=triangle_probabilities)
    sampled_points = sample_all_triangles(vertices[faces[triangle_indices,0]], vertices[faces[triangle_indices,1]], vertices[faces[triangle_indices,2]])
    return sampled_points
    
