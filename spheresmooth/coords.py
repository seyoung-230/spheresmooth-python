import numpy as np


def cartesian_to_spherical(x: np.ndarray, byrow: bool = True) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates (theta, phi).

    Parameters
    ----------
    x : array-like, shape (n, 3) or (3, n)
        Points in Cartesian coordinates (x, y, z).
    byrow : bool, default=True
        If True, rows are points. If False, columns are points.

    Returns
    -------
    theta_phi : ndarray, shape (n, 2)
        Spherical coordinates (theta, phi) with:
        - theta in [0, pi]   (inclination from +z)
        - phi   in [0, 2*pi) (azimuth)
    """
    arr = np.asarray(x, dtype=float)
    if not byrow:
        arr = arr.T

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.shape[1] != 3:
        raise ValueError("Input must have shape (n, 3) in Cartesian coordinates.")

    xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
    r = np.linalg.norm(arr, axis=1)
    r = np.where(r == 0, 1.0, r)  # avoid division by zero

    # theta: inclination
    theta = np.arccos(np.clip(zs / r, -1.0, 1.0))
    # phi: azimuth
    phi = np.arctan2(ys, xs)
    phi = np.mod(phi, 2 * np.pi)

    return np.column_stack((theta, phi))


def spherical_to_cartesian(theta_phi: np.ndarray, byrow: bool = True) -> np.ndarray:
    """
    Convert spherical coordinates (theta, phi) to Cartesian coordinates (x, y, z).

    Parameters
    ----------
    theta_phi : array-like, shape (n, 2) or (2, n)
        Each row: (theta, phi), with theta in [0, pi], phi in [0, 2*pi).
    byrow : bool, default=True
        If True, rows are points. If False, columns are points.

    Returns
    -------
    xyz : ndarray, shape (n, 3)
        Cartesian coordinates on the unit sphere.
    """
    arr = np.asarray(theta_phi, dtype=float)
    if not byrow:
        arr = arr.T

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.shape[1] != 2:
        raise ValueError("Input must have shape (n, 2) in spherical coordinates.")

    theta, phi = arr[:, 0], arr[:, 1]
    sin_theta = np.sin(theta)

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)

    return np.column_stack((x, y, z))


def edp(p: np.ndarray) -> np.ndarray:
    """
    Equal-distance projection of a point on the unit sphere onto the xy-plane.

    NOTE
    ----
    The exact projection formula used in the R implementation can be
    matched later. For now, this function returns a simple orthographic
    projection onto the xy-plane, i.e. (x, y).

    Parameters
    ----------
    p : array-like, shape (3,)
        Point on (or near) the unit sphere.

    Returns
    -------
    projected : ndarray, shape (2,)
        Projected point in the plane.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != (3,):
        raise ValueError("p must be a 3-dimensional vector.")

    # TODO: Replace with the exact equal-distance projection if needed.
    return p[:2].copy()
