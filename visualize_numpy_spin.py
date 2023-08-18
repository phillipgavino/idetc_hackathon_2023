import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vti_to_numpy import vti_to_numpy
from rescale_microstructure import rescale_microstructure
import numpy as np


def visualize_numpy_spin(spin_data):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = np.where(spin_data >= 0)
    values = spin_data[x, y, z]
    
    sc = ax.scatter3D(x, y, z, c=values, cmap='viridis', marker='o', s=5)

    cbar = fig.colorbar(sc)
    cbar.set_label('Spin')

    plt.show()

if __name__ == "__main__":
    vti_filename = 'seed-001-potts_3d.50.vti'
    numpy_array = vti_to_numpy(vti_filename)
    bbox = [1.1,2.2,1.3]
    spin_data = rescale_microstructure(numpy_array, bbox)
    print(spin_data)
    visualize_numpy_spin(spin_data)