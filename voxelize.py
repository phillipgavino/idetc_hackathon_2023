import trimesh
import vtk
import numpy as np
import save_vti as save
from vtk.util.numpy_support import numpy_to_vtk
import time



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

    return voxel_grid


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
    filename = "cube.STL"
    voxel_size = 0.01  # adjust as needed
    start = time.time()
    voxels = voxelize_stl(filename, voxel_size)
    end = time.time()
    print(end - start)
    start = time.time()
    image_data = voxel_to_vti(voxels.matrix.astype(float))
    end = time.time()
    print(end - start)
    save.save_vti(image_data, r"voxelized_stl\cube.vti")


if __name__ == "__main__":
    main()
