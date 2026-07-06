"""PBC-aware nearest-image geometry utilities.

The core design is local: for each frame we calculate the nearest periodic
image of a ligand/cofactor/ion relative to a target site. This avoids writing a
globally PBC-corrected trajectory for many site-centred analyses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """Simulation unit cell.

    Parameters
    ----------
    dimensions
        MDAnalysis-style dimensions ``[lx, ly, lz, alpha, beta, gamma]`` or a
        3x3 box matrix. Coordinates and lengths must use the same units.
    """

    matrix: np.ndarray

    @classmethod
    def from_dimensions(cls, dimensions: np.ndarray | list[float]) -> "Box":
        arr = np.asarray(dimensions, dtype=float)
        if arr.shape == (3, 3):
            return cls(arr)
        if arr.shape == (6,):
            lx, ly, lz, alpha, beta, gamma = arr
            if not np.allclose([alpha, beta, gamma], [90.0, 90.0, 90.0]):
                return cls(_triclinic_vectors(lx, ly, lz, alpha, beta, gamma))
            return cls(np.diag([lx, ly, lz]))
        if arr.shape == (3,):
            return cls(np.diag(arr))
        raise ValueError("Box must be length-3, length-6, or 3x3 array.")

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)


def _triclinic_vectors(
    lx: float, ly: float, lz: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Return a row-vector triclinic box matrix from lengths and angles."""

    a = np.deg2rad(alpha)
    b = np.deg2rad(beta)
    g = np.deg2rad(gamma)

    ax = lx
    by = ly * np.sin(g)
    bx = ly * np.cos(g)
    cx = lz * np.cos(b)
    cy = lz * (np.cos(a) - np.cos(b) * np.cos(g)) / np.sin(g)
    cz_sq = lz**2 - cx**2 - cy**2
    cz = np.sqrt(max(cz_sq, 0.0))
    return np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]], dtype=float)


def nearest_image_vector(
    target_xyz: np.ndarray, mobile_xyz: np.ndarray, box: Box | np.ndarray | list[float]
) -> np.ndarray:
    """Vector from ``target_xyz`` to nearest periodic image of ``mobile_xyz``.

    Returns
    -------
    numpy.ndarray
        Vector(s) in Cartesian coordinates. If ``mobile_xyz`` is an ``N x 3``
        array, returns ``N x 3``.
    """

    box_obj = box if isinstance(box, Box) else Box.from_dimensions(box)
    target = np.asarray(target_xyz, dtype=float)
    mobile = np.asarray(mobile_xyz, dtype=float)
    delta = mobile - target

    fractional = delta @ box_obj.inverse
    fractional -= np.rint(fractional)
    return fractional @ box_obj.matrix


def nearest_image_position(
    target_xyz: np.ndarray, mobile_xyz: np.ndarray, box: Box | np.ndarray | list[float]
) -> np.ndarray:
    """Nearest-image Cartesian position of mobile coordinates around target."""

    return np.asarray(target_xyz, dtype=float) + nearest_image_vector(target_xyz, mobile_xyz, box)


def distances(vectors: np.ndarray) -> np.ndarray:
    """Euclidean norm for one or more vectors."""

    return np.linalg.norm(np.asarray(vectors, dtype=float), axis=-1)
