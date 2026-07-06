import numpy as np

from mdmaia.pbc import Box, nearest_image_vector


def test_nearest_image_vector_orthorhombic():
    box = Box.from_dimensions([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
    target = np.array([1.0, 1.0, 1.0])
    mobile = np.array([9.0, 1.0, 1.0])
    vec = nearest_image_vector(target, mobile, box)
    assert np.allclose(vec, [-2.0, 0.0, 0.0])


def test_nearest_image_vector_multiple_points():
    box = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
    target = np.array([5.0, 5.0, 5.0])
    mobile = np.array([[6.0, 5.0, 5.0], [0.5, 5.0, 5.0]])
    vec = nearest_image_vector(target, mobile, box)
    assert np.allclose(vec, [[1.0, 0.0, 0.0], [-4.5, 0.0, 0.0]])
