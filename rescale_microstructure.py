import numpy as np
from vti_to_numpy import vti_to_numpy


def rescale_microstructure(spin_array, dims, origin=[0, 0, 0]):
    # Implicit assumption here is that all STL files will be larger than the microstrucure.

    max_rows = 100

    xdim = dims[0]
    ydim = dims[1]
    zdim = dims[2]

    xdiff = xdim - 1
    ydiff = ydim - 1
    zdiff = zdim - 1

    n_xrows = int(xdiff * 100)
    n_yrows = int(ydiff * 100)
    n_zrows = int(zdiff * 100)

    iter_x = int(n_xrows / max_rows)
    rem_x = int(n_xrows % max_rows)
    iter_y = int(n_yrows / max_rows)
    rem_y = int(n_yrows % max_rows)
    iter_z = int(n_zrows / max_rows)
    rem_z = int(n_zrows % max_rows)

    new_spin_array = spin_array

    for i in range(iter_x):
        extra_xrows = spin_array[:, :, :]
        new_spin_array = np.append(new_spin_array, extra_xrows, axis=0)
    remaining_xrows = spin_array[0:rem_x, :, :]
    new_spin_array = np.append(new_spin_array, remaining_xrows, axis=0)

    for i in range(iter_y):
        extra_yrows = new_spin_array[:, :, :]
        new_spin_array = np.append(new_spin_array, extra_yrows, axis=1)
    remaining_yrows = new_spin_array[:, 0:rem_y, :]
    new_spin_array = np.append(new_spin_array, remaining_yrows, axis=1)

    for i in range(iter_z):
        extra_zrows = new_spin_array[:, :, :]
        new_spin_array = np.append(new_spin_array, extra_zrows, axis=2)
    remaining_zrows = new_spin_array[:, :, 0:rem_z]
    new_spin_array = np.append(new_spin_array, remaining_zrows, axis=2)

    print("Final spin array shape: " + str(new_spin_array.shape))

    # Make corresponding meshgrid for spin matrix:

    m, n, o = new_spin_array.shape
    ox, oy, oz = origin[0], origin[1], origin[2]

    xs = np.arange(ox + 5, ox + m * 10, 10)
    ys = np.arange(oy + 5, oy + n * 10, 10)
    zs = np.arange(oz + 5, oz + o * 10, 10)

    X, Y, Z = np.meshgrid(xs, ys, zs)

    return new_spin_array, X, Y, Z


if __name__ == "__main__":
    vti_filename = 'seed-001-potts_3d.50.vti'
    spin_array = vti_to_numpy(vti_filename)
    bbox = [1.1, 2.2, 1.3]
    spin_array, X, Y, Z = rescale_microstructure(spin_array, bbox, origin=[-100, -100, 0])
    print(X)
