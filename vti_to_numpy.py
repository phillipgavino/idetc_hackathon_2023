import vtk
from vtkmodules.util import numpy_support


def vti_to_numpy(vti_filename):
    # Read the VTI file using VTK
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_filename)
    reader.Update()

    # Extract the data from the VTI file
    image_data = reader.GetOutput()
    point_data = image_data.GetCellData()
    array_data = point_data.GetArray(0)

    # Convert the VTK array to a numpy array
    numpy_array = numpy_support.vtk_to_numpy(array_data)

    # Reshape the numpy array based on the dimensions of the VTI data
    dims = image_data.GetDimensions()
    reshaped_array = numpy_array.reshape(dims[2]-1, dims[1]-1, dims[0]-1)

    return reshaped_array

if __name__ == "__main__":
    vti_filename = 'seed-001-potts_3d.50.vti'
    numpy_array = vti_to_numpy(vti_filename)
    print(numpy_array)
