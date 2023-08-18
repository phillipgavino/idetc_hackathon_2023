import pandas as pd
import trimesh
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
import time
from scipy.spatial import KDTree
from save_vti import *


def kd_tree(voxel_grid, micro_structure):
    """
    Maps a given voxel grid to a randomized numpy array using KDTree.

    Parameters:
    - voxel_grid: 3D numpy array representing the voxel grid.
    - micro_structure: Not used in this function, but retained for consistency.

    Returns:
    - mapped_array: 3D numpy array with values from the randomized array mapped onto the voxel grid.
    """

    print(voxel_grid.shape)
    random_array_shape = voxel_grid.shape

    # Create a randomized numpy array
    random_array = np.random.randint(0, 1001, random_array_shape)
    print(random_array)
    random_array = random_array.astype(float)

    # Ensure the voxel grid and random array are of the same shape
    if voxel_grid.shape != random_array.shape:
        raise ValueError("Voxel grid and random array shapes do not match!")

    # Create KDTree for the random array
    indices = np.array(np.where(np.ones_like(random_array))).T
    print(indices)
    df = pd.DataFrame(indices)
    df.to_csv("temp.csv", index=False)
    tree = KDTree(indices)

    # Get all the indices of the voxel_grid where voxel value is greater than 0
    voxel_indices = np.array(np.where(voxel_grid.matrix > 0)).T

    # Query the KDTree for all voxel indices at once
    _, nearest_indices = tree.query(voxel_indices)

    # Create an array of zeros to map the values from random_array using the nearest_indices
    mapped_array = np.zeros_like(random_array, dtype=random_array.dtype)
    for voxel_idx, nearest_idx in zip(voxel_indices, nearest_indices):
        mapped_array[tuple(voxel_idx)] = random_array[tuple(indices[nearest_idx])]

    return mapped_array


# def voxel_to_vti(input_filename, output_filename):
#
#     # Load the mesh
#     mesh = trimesh.load_mesh(input_filename)
#
#     # Voxelization of the mesh
#     voxel = mesh.voxelized(0.01)
#     voxel_matrix = voxel.matrix
#     """
#     Converts a 3D voxel matrix to a .vti file.
#
#     Parameters:
#     - voxel_matrix: 3D numpy array.
#     - filename: Name of the file to save to (should end with .vti).
#     """
#
#     # Convert the numpy array to a VTK array
#     vtk_data = numpy_to_vtk(num_array=voxel_matrix.ravel(order='F'), deep=True, array_type=vtk.VTK_INT)
#
#     # Create an ImageData object and assign the vtk array to it
#     image_data = vtk.vtkImageData()
#     image_data.SetDimensions(voxel_matrix.shape)
#     image_data.GetPointData().SetScalars(vtk_data)
#
#     # Write the ImageData object to a .vti file
#     writer = vtk.vtkXMLImageDataWriter()
#     writer.SetFileName(output_filename)
#     writer.SetInputData(image_data)
#     writer.Write()

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

    return voxel_grid, mesh


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
    # Example usage:
    filename = "cad/tube.stl"
    voxel_size = 0.01  # adjust as needed
    start = time.time()
    voxels, mesh = voxelize_stl(filename, voxel_size)
    # filled_voxels = fill_interior_voxels(voxels, mesh)
    end = time.time()
    print(end - start)
    start = time.time()

    mapped_array = kd_tree(voxels, [])

    image_data = voxel_to_vti(mapped_array)

    end = time.time()
    print(end - start)
    save_vti(image_data, r"voxelized_stl\tube_3.vti")


if __name__ == "__main__":
    main()

# def main():
#     # Example usage:
#
#     filename = r"cad\tube.stl"
#     voxel_size = 0.01  # 10 micrometers
#
#     dims = (100, 100, 100)
#
#     voxel_to_vti(filename, "output_2.vti")


# result_array = map_voxelized_stl_to_random_array(filename, voxel_size, dims)

# start = time.time()
# zero_array = np.zeros(dims, dtype=int)
# random_array = np.random.randint(0, 1001, dims)
#
# indices = np.array(np.where(np.ones_like(random_array))).T
#
# tree = KDTree(indices)
#
# # Get all the indices of the zero_array in a vectorized manner
# all_indices = np.array(list(np.ndindex(zero_array.shape)))
#
# # Query the KDTree for all indices at once
# _, nearest_indices = tree.query(all_indices)
#
# # Map the values from random_array to zero_array using the nearest_indices
# for idx, nearest_idx in zip(all_indices, nearest_indices):
#     zero_array[tuple(idx)] = random_array[tuple(indices[nearest_idx])]
# #
# print("Random Array:")
# print(random_array)
# print("\nMapped Zero Array:")
# print(zero_array)
# print(time.time() - start)

# filename = "output_2.vti"
# numpy_to_vtk_image(result_array, filename)

# print(voxels)

# end = time.time()
# print(end - start)
# start = time.time()
# image_data = voxel_to_vti(voxels.matrix.astype(float))
# end = time.time()
# print(end - start)
# save.save_vti(image_data, r"voxelized_stl\cube.vti")

#
# if __name__ == "__main__":
#     main()
