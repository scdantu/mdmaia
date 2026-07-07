import numpy as np

from mdmaia.collect import _align_coordinates
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


def test_align_coordinates_maps_mobile_to_reference():
    fixed = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mobile = fixed + np.array([5.0, -3.0, 2.0])
    coords = np.array([[5.2, -2.8, 2.0]])
    aligned = _align_coordinates(coords, mobile, fixed)
    assert np.allclose(aligned, [[0.2, 0.2, 0.0]])
