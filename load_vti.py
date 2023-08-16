import vtk


def load_vti(filename):
    """Load a VTI file into a vtkImageData object."""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()
