import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from save_vti import save_vti
from scipy.ndimage import label
from load_vti import load_vti

from scipy.spatial import KDTree


def main():
    # Sample usage:
    source_filename = "hackathon-dataset\seed-001-potts_3d.50.vti"
    target_filename = "voxelized_stl\cube.vti"

    source_vti = load_vti(source_filename)
    target_vti = load_vti(target_filename)

    dims = target_vti.GetOutput().GetDimensions()

    print("Source:\n", source_vti.GetOutput())

    print("Target:\n",  target_vti.GetOutput())


if __name__ == "__main__":
    main()
