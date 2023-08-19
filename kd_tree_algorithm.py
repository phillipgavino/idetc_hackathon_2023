import pandas as pd
import trimesh
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
import time
from scipy.spatial import KDTree
from save_vti import save_vti
from vti_to_numpy import vti_to_numpy
from rescale_microstructure import rescale_microstructure


def kd_tree_spin_mapping(spin_array, voxel_coords, micro_coords, shape):
    """
    Map the spin values from `spin_array` to the `voxel_coords` using a KDTree for nearest-neighbor lookup.

    Parameters:
    - spin_array (np.array): Array containing spin values.
    - voxel_coords (np.array): 3D array of voxel coordinates.
    - micro_coords (np.array): List of micro-structure coordinates.
    - dims (tuple): Dimensions to reshape the final result.

    Returns:
    - voxel_spin (np.array): 3D array of spin values mapped to voxel coordinates.
    """

    # Convert spin_array to float and flatten it
    spin_array = spin_array.astype(float).ravel()

    # Create KDTree for efficient nearest-neighbor lookup
    tree = KDTree(micro_coords)

    # Find the filled voxel indices
    filled_voxel_indices = np.array(np.where(voxel_coords > 0)).T
    unique_vox_ind = np.unique(filled_voxel_indices[:, 0])

    # Initialize an array to store mapped spin values
    voxel_spin = np.zeros_like(spin_array)

    # For each filled voxel, find the nearest micro coordinate and assign the corresponding spin value
    for idx, voxel_index in enumerate(unique_vox_ind):

        voxel_coordinate = voxel_coords[voxel_index]

        _, ind = tree.query(voxel_coordinate)

        voxel_spin[voxel_index] = spin_array[ind]

    # Reshape the result to the given dimensions
    voxel_spin = voxel_spin.reshape(shape)

    return voxel_spin


def voxelize_stl(filename, voxel_size):
    """
    Voxelize an STL file.

    Parameters:
    - filename: Path to the STL file
    - voxel_size: Desired size of the voxels

    Returns:
    - voxelized mesh (trimesh.VoxelGrid instance)
    """
    # Load the STL file as a trimesh mesh
    mesh = trimesh.load_mesh(filename)

    # Voxelize the mesh
    voxel_grid = mesh.voxelized(pitch=voxel_size)

    voxel_grid = voxel_grid.fill()

    points = voxel_grid.points / 0.001

    min_points = np.min(points, axis=0)
    trans_points = points - min_points

    filled_voxel_indices = np.array(np.where(voxel_grid.matrix > 0)).T

    x_matrix = np.zeros(voxel_grid.matrix.shape)
    y_matrix = np.zeros(voxel_grid.matrix.shape)
    z_matrix = np.zeros(voxel_grid.matrix.shape)

    for idx, voxel_index in enumerate(filled_voxel_indices):
        x_matrix[tuple(voxel_index)] = trans_points[idx][0]
        y_matrix[tuple(voxel_index)] = trans_points[idx][1]
        z_matrix[tuple(voxel_index)] = trans_points[idx][2]

    x_matrix = x_matrix.ravel()
    y_matrix = y_matrix.ravel()
    z_matrix = z_matrix.ravel()
    filled_voxel_matrix = np.column_stack((x_matrix, y_matrix, z_matrix))

    return filled_voxel_matrix, voxel_grid.shape


def voxel_to_vti(voxels):
    # Convert voxel data to vtkImageData
    image_data = vtk.vtkImageData()

    # Set the dimensions of the vtkImageData structure
    image_data.SetDimensions(*voxels.shape)
    image_data.AllocateScalars(vtk.VTK_FLOAT, 1)  # Assuming your voxels data type is float

    # Convert the voxel data to a flat array
    flat_voxels = voxels.ravel(order='F')  # 'F' order means Fortran-style order which is used by vtk

    # Convert flat numpy array to vtk array and set it as scalars
    vtk_array = numpy_to_vtk(flat_voxels)
    image_data.GetPointData().SetScalars(vtk_array)

    return image_data


def main():
    filename = "cad/cube.stl"
    voxel_size = 0.01  # 10 microns

    start = time.time()

    voxel_coords, bbox = voxelize_stl(filename, voxel_size)

    print("Voxelize Time:", time.time() - start)

    bbox = np.array(bbox) * voxel_size

    vti_filename = 'hackathon-dataset/seed-001-potts_3d.50.vti'

    start = time.time()

    spin_array = vti_to_numpy(vti_filename)
    spin_array, X, Y, Z = rescale_microstructure(spin_array, bbox)

    print("Rescale Microstructure:", time.time() - start)

    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()
    micro_coords = np.column_stack((X, Y, Z))

    shape = (bbox / voxel_size).astype(int)

    start = time.time()

    mapped_array = kd_tree_spin_mapping(spin_array, voxel_coords, micro_coords, shape)

    print("Mapping Time:", time.time() - start)

    image_data = voxel_to_vti(mapped_array)

    save_vti(image_data, r"voxelized_stl\cube.vti")


if __name__ == "__main__":
    main()
