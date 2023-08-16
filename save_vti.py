import vtk

def save_vti(image_data, output_filename):
    # Save the vtkImageData structure to a .vti file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(image_data)
    writer.Write()