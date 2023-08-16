import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from save_vti import save_vti
from scipy.ndimage import label
from load_vti import load_vti

from scipy.ndimage import zoom
from scipy.spatial import KDTree


def upscale_source_to_target(source_array, target_shape):
    """
    Upscale the source_array to the given target_shape using interpolation.

    Parameters:
    - source_array (np.ndarray): The 3D array you want to upscale.
    - target_shape (tuple): The desired shape (z, y, x) to upscale to.

    Returns:
    - np.ndarray: The upscaled 3D array.
    """

    # Calculate the scaling factors for each dimension
    z_scale = target_shape[0] / source_array.shape[0]
    y_scale = target_shape[1] / source_array.shape[1]
    x_scale = target_shape[2] / source_array.shape[2]

    # Use zoom to upscale
    upscaled_array = zoom(source_array, (z_scale, y_scale, x_scale))

    return upscaled_array


def map_source_to_target_using_kdtree(source_array, target_array):
    # Convert voxel data to points (non-zero voxels can be considered as points).
    source_points = np.argwhere(source_array)
    target_points = np.argwhere(target_array)

    # Create a KDTree from target points.
    tree = KDTree(target_points)

    # For each point in the source, find its nearest neighbor in the target.
    distances, indices = tree.query(source_points)

    # The above gives the nearest target point for each source point.
    # Now you can map or manipulate the source based on this information.

    return distances, indices


def main():
    # Sample usage:
    source_filename = "hackathon-dataset\seed-001-potts_3d.50.vti"
    target_filename = "voxelized_stl\cube.vti"

    source_vti = load_vti(source_filename)
    target_vti = load_vti(target_filename)

    dims = target_vti.GetDimensions()

    print(target_vti)

    source_array = vtk_to_numpy(source_vti.GetCellData().GetScalars())  # Your 3D source array
    source_array = source_array.reshape(100, 100, 100)
    target_shape = (dims[2], dims[1], dims[0])  # Replace with the shape of your target VTK
    target_array = vtk_to_numpy(target_vti.GetPointData().GetScalars())


    upscaled_source = upscale_source_to_target(source_array, target_shape)
    distances, indices = map_source_to_target_using_kdtree(upscaled_source, target_array)


if __name__ == "__main__":
    main()
