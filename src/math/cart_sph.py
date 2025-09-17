"""
    src/maths/coords.py
    -------------------

    Mathematical script, fully vetorized spherical/cartesian coordinate 
    conversions and angle combinations (rotation) utilities for efficiently
    adding and subtracting angles. Designed to work seamlessly with scalars
    and NumPy Arrays. 
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from typing import Final, Tuple, Union

# Typing Declarations
ArrayF = npt.NDArray[np.floating]
ArrayF64 = npt.NDArray[np.float64]

# Constants
DEG2RAD: Final[float] = np.pi / 180.0
RAD2DEG: Final[float] = 180.0 / np.pi
EPS: Final[float] = 1e-12


# --------------- Utility Helpers --------------- #

# If adding overhead, equivalent to np.atleast_1d(np.asarray(...))
def _as_1d_array(x: Union[ArrayF, float]) -> ArrayF64:
    """
        Convert input to a 1D NumPy array of dtype float64.

        Scalars are wrapped into a 1-element array. Lists, 
        tuples, or arrays are flattened to 1D where possible.

        Args:
        -----
            x: Input scalar or array-like values.

        Returns:
        --------
            ArrayF64: 1D array of dtype float64 with shape (N,).
        --------
    """
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 0: array = array[None]
    return array


# ----------===== Coordinate Conversions =====----------- #

def cartesian_to_spherical(dvec: ArrayF) -> Tuple[ArrayF64, ArrayF64, ArrayF64]:
    """
        Convert Cartesian vectors to spherical coordinates `(r, phi, theta)`.

        - `r` is the vector magnitude.
        - `phi` is the azimuth angle in degrees, measured from +X in the XY-plane.
        - `theta` is the polar angle in degrees, measured from +Z axis.

        Definitions:
        ------------
            phi   = atan2(y, x) ∈ [-180, 180]
            theta = arccos(z / r) ∈ [0, 180]

        Args:
        -----
            dvec: Array-like with shape (3,) or (N, 3). Each row is (x, y, z).

        Returns:
        --------
            Tuple of arrays (radius, phi, theta), each with shape (N,).
        --------
    """
    array = np.asarray(dvec, dtype=np.float64)
    if array.ndim == 1 and array.size == 3: array = array[None, :]
    if array.ndim != 2 and array.shape[1] != 3:
        raise ValueError("dvec must have shape (N, 3) or (3,)")
        
    x, y, z = array[:, 0], array[:, 1], array[:, 2]
    radius = np.linalg.norm(array, axis=1)

    # Avoid divide-by-zero ...
    np.maximum(radius, EPS, out=radius)
    
    phi = np.arctan2(y, x) * RAD2DEG
    theta = np.arccos(np.clip(z / radius, -1.0, 1.0)) * RAD2DEG
    return radius, phi, theta



def spherical_to_cartesian(radius: ArrayF, phi: ArrayF, theta: ArrayF) -> ArrayF64:
    """
        Convert spherical coordinates to Cartesian `(x, y, z)`.

        Angles are expected in degrees:
        - `phi` is the azimuth (rotation around Z).
        - `theta` is the polar angle from +Z.

        Args:
        -----
            radius: Radius or array of radii.
            phi: Azimuth angle(s) in degrees.
            theta: Polar angle(s) in degrees.

        Returns:
        --------
            ArrayF64: Cartesian coordinates with shape (N, 3).
        --------
    """
    radius = _as_1d_array(radius)
    phi_array, theta_array = _as_1d_array(phi), _as_1d_array(theta)

    # Broadcast to common shape
    radius, phi_array, theta_array = np.broadcast_arrays(radius, 
                                                         phi_array, theta_array)
    phi_rad, theta_rad = phi_array * DEG2RAD, theta_array * DEG2RAD
    sin_theta = np.sin(theta_rad)

    return np.column_stack((radius * np.cos(phi_rad) * sin_theta,
                            radius * np.sin(phi_rad) * sin_theta,
                            radius * np.cos(theta_rad)))



# ----------===== Angle combination (rotation) =====----------- #

def _angle_rotation_kernel(phi0: ArrayF, theta0: ArrayF,
                           phi1: ArrayF, theta1: ArrayF,
                           inverse: bool=False) -> Tuple[ArrayF64, ArrayF64]:
    """
        Rotate a spherical angle `(phi0, theta0)` by another 
        angle `(phi1, theta1)`.

        Works internally in radians and operates on arrays 
        elementwise. Implements rotation via spherical-to-Cartesian
        conversion, applying a rotation matrix, then converting back.

        Args:
            phi0: Azimuth(s) of original direction, in radians.
            theta0: Polar angle(s) of original direction, in radians.
            phi1: Azimuth(s) of rotation axis, in radians.
            theta1: Polar angle(s) of rotation axis, in radians.
            inverse: If True, apply the inverse rotation.

        Returns:
            Tuple (phi, theta):
                - phi: New azimuth(s) in radians.
                - theta: New polar angle(s) in radians.
    """
    phi0 = np.asarray(phi0, dtype=np.float64)
    phi1 = np.asarray(phi1, dtype=np.float64)
    theta0 = np.asarray(theta0, dtype=np.float64)
    theta1 = np.asarray(theta1, dtype=np.float64)

    phi0, theta0, phi1, theta1 = np.broadcast_arrays(phi0, theta0, phi1, theta1)

    # Precompute the Trigonometric functions, compute each only once
    st0, ct0, sp0, cp0 = np.sin(theta0), np.cos(theta0), np.sin(phi0), np.cos(phi0)
    st1, ct1, sp1, cp1 = np.sin(theta1), np.cos(theta1), np.sin(phi1), np.cos(phi1)

    # original vector components (Cartesian)
    x0, y0, z0 = st0 * cp0, st0 * sp0, ct0
    if not inverse:
        # R = Rz(phi1) @ Ry(theta1)
        m00, m01, m02 = cp1 * ct1, -sp1, cp1 * st1
        m10, m11, m12 = sp1 * ct1, cp1, sp1 * st1
        m20, m21, m22 = -st1, 0.0, ct1

        x = m00 * x0 + m01 * y0 + m02 * z0
        y = m10 * x0 + m11 * y0 + m12 * z0
        z = m20 * x0 + m21 * y0 + m22 * z0
    else:
        # inverse rotation: apply R.T (transpose) to the vector.
        # Instead of forming R and transposing, write R.T element formulas directly.
        # R.T rows are R columns.
        # R.T[0] = [m00, m10, m20], R.T[1] = [m01, m11, m21], R.T[2] = [m02, m12, m22]
        m00, m10, m20 = cp1 * ct1, sp1 * ct1, -st1
        m01, m11, m21 = -sp1, cp1, 0.0
        m02, m12, m22 = cp1 * st1, sp1 * st1, ct1

        x = m00 * x0 + m10 * y0 + m20 * z0
        y = m01 * x0 + m11 * y0 + m21 * z0
        z = m02 * x0 + m12 * y0 + m22 * z0
    
    np.clip(z, -1.0, 1.0, out=z)
    return np.arctan2(y, x), np.arccos(z)



def _combine_angles(phi0: ArrayF64, theta0: ArrayF64,
                    phi1: ArrayF64, theta1: ArrayF64,
                    inverse: bool=False) -> Tuple[ArrayF64, ArrayF64]:
    """
        Combine two spherical angles `(phi0, theta0)` and `(phi1, theta1)`.

        Equivalent to rotating the first angle by the second one.
        Wrapper around `_angle_rotation_kernel`, handling degree/radian
        conversions.

        Args:
            phi0: Azimuth(s) of first angle in degrees.
            theta0: Polar angle(s) of first angle in degrees.
            phi1: Azimuth(s) of second angle in degrees.
            theta1: Polar angle(s) of second angle in degrees.
            inverse: If True, apply the inverse rotation.

        Returns:
            Tuple (phi, theta) in degrees.
    """
    p0, t0 = _as_1d_array(phi0) * DEG2RAD, _as_1d_array(theta0) * DEG2RAD
    p1, t1 = _as_1d_array(phi1) * DEG2RAD, _as_1d_array(theta1) * DEG2RAD

    phi, theta = _angle_rotation_kernel(p0, t0, p1, t1, inverse=inverse)
    return phi * RAD2DEG, theta * RAD2DEG





def add_angles(phi0: ArrayF, theta0: ArrayF,
               phi1: ArrayF, theta1: ArrayF) -> Tuple[ArrayF64, ArrayF64]:
    """
        Add two spherical angles `(phi0, theta0)` and `(phi1, theta1)`.
        Equivalent to rotating the first direction by the second direction.

        Args:
        -----
            phi0: Azimuth(s) of first angle in degrees.
            theta0: Polar angle(s) of first angle in degrees.
            phi1: Azimuth(s) of second angle in degrees.
            theta1: Polar angle(s) of second angle in degrees.

        Returns:
        --------
            Tuple (phi, theta) in degrees representing the rotated angle.
    """
    return _combine_angles(phi0, theta0, phi1, theta1, inverse=False)


def sub_angles(phi0: ArrayF, theta0: ArrayF,
               phi1: ArrayF, theta1: ArrayF) -> Tuple[ArrayF64, ArrayF64]:
    """
        Subtract spherical angle `(phi1, theta1)` from `(phi0, theta0)`.
        Equivalent to applying the inverse rotation of `(phi1, theta1)`.

        Args:
        -----
            phi0: Azimuth(s) of first angle in degrees.
            theta0: Polar angle(s) of first angle in degrees.
            phi1: Azimuth(s) of angle to subtract in degrees.
            theta1: Polar angle(s) of angle to subtract in degrees.

        Returns:
        --------
            Tuple (phi, theta) in degrees representing the rotated angle.
    """
    return _combine_angles(phi0, theta0, phi1, theta1, inverse=True)
