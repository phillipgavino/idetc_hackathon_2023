import numpy as np
from vti_to_numpy import vti_to_numpy


def optimized_rescale_microstructure(spin_array, dims, origin=[0, 0, 0]):
    """Rescale the given microstructure based on specified dimensions."""

    max_rows = 100

    # Compute the number of rows required in each dimension
    n_rows = np.array([(dim - 1) * 100 for dim in dims])

    # Compute the iterations and remainders for each dimension
    iters = n_rows // max_rows
    rems = n_rows % max_rows

    # Allocate space for new spin array
    new_dims = np.array(spin_array.shape) + n_rows
    new_spin_array = np.zeros(new_dims, dtype=spin_array.dtype)

    # Fill the allocated array using slices
    slices = [slice(None)] * 3
    for i, (iter_dim, rem_dim) in enumerate(zip(iters, rems)):
        new_spin_array[slices] = spin_array
        for _ in range(iter_dim):
            slices[i] = slice(None, -max_rows)
            new_spin_array[slices] += spin_array
        slices[i] = slice(None, rem_dim)
        new_spin_array[slices] += spin_array[:rem_dim]

    print("Final spin array shape: " + str(new_spin_array.shape))

    # Make corresponding meshgrid for spin matrix
    m, n, o = new_spin_array.shape
    ox, oy, oz = origin

    xs = np.arange(ox + 5, ox + m * 10, 10)
    ys = np.arange(oy + 5, oy + n * 10, 10)
    zs = np.arange(oz + 5, oz + o * 10, 10)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    return new_spin_array, X, Y, Z


# Sample usage
if __name__ == "__main__":
    vti_filename = 'seed-001-potts_3d.50.vti'
    spin_array = vti_to_numpy(vti_filename)
    bbox = [1.1, 2.2, 1.3]
    spin_array, X, Y, Z = rescale_microstructure(spin_array, bbox, origin=[-100, -100, 0])
    print(X)
