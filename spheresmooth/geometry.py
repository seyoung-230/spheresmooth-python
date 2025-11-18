import numpy as np


def dot(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the dot product of two vectors."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return float(np.dot(u, v))


def cross(u: np.ndarray, v: np.ndarray, normalize_vec: bool = False) -> np.ndarray:
    """
    Compute the cross product of two vectors.

    Parameters
    ----------
    u, v : array-like, shape (3,)
        Input vectors.
    normalize_vec : bool, default=False
        If True, return the normalized cross product.

    Returns
    -------
    w : ndarray, shape (3,)
        Cross product vector (possibly normalized).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    w = np.cross(u, v)
    if normalize_vec:
        n = np.linalg.norm(w)
        if n > 0:
            w = w / n
    return w


def norm2(u: np.ndarray) -> float:
    """Compute the L2 (Euclidean) norm of a vector u."""
    u = np.asarray(u, dtype=float)
    return float(np.linalg.norm(u))


import numpy as np

def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize a vector or a matrix row-wise using the L2 norm.

    Parameters
    ----------
    x : array-like
        Vector (shape (d,)) or matrix (shape (n, d)).

    Returns
    -------
    ndarray
        If input is a vector -> returns a normalized vector (d,).
        If input is a matrix -> returns row-normalized matrix (n, d).
        Zero vectors/rows remain unchanged.
    """
    arr = np.asarray(x, dtype=float)

    # Case 1: vector input (1D)
    if arr.ndim == 1:
        n = np.linalg.norm(arr)
        return arr if n == 0 else arr / n

    # Case 2: matrix input (2D)
    elif arr.ndim == 2:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    else:
        raise ValueError("Input must be a 1D vector or 2D matrix.")


def spherical_dist(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate spherical distance between two vectors on the unit sphere.

    Parameters
    ----------
    x, y : array-like, shape (3,)

    Returns
    -------
    float
        Geodesic distance on S^2, i.e. arccos(<x, y>).
    """
    x = normalize(x)
    y = normalize(y)
    cos_val = np.clip(dot(x, y), -1.0, 1.0)
    return float(np.arccos(cos_val))


def exp_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Exponential map on the unit sphere S^2.

    Given a base point x on S^2 and a tangent vector v in R^3,
    returns exp_x(v) on the sphere.

    Parameters
    ----------
    x : array-like, shape (3,)
        Base point (should be on the unit sphere).
    v : array-like, shape (3,)
        Tangent vector at x (approximately orthogonal to x).

    Returns
    -------
    ndarray, shape (3,)
        Point on the unit sphere.
    """
    x = normalize(x)
    v = np.asarray(v, dtype=float)

    # Remove radial component to keep v tangent to the sphere
    v = v - dot(v, x) * x
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return x

    v_unit = v / norm_v
    return np.cos(norm_v) * x + np.sin(norm_v) * v_unit


import numpy as np

def geodesic(t,
             p: np.ndarray,
             q: np.ndarray,
             a: float,
             b: float) -> np.ndarray:
    """
    Compute points along the geodesic on S^2 connecting p and q
    at time points t in [a, b].  Accepts scalar or array input.

    Parameters
    ----------
    t : float or array-like
        Time point(s). Can be scalar or array-like.
    p, q : array-like, shape (3,)
        Endpoints on the sphere.
    a, b : float
        Start and end time of the geodesic segment.

    Returns
    -------
    gamma : ndarray
        If t is scalar, returns shape (3,).
        If t is array-like, returns shape (len(t), 3).
    """
    # detect whether input is scalar
    scalar_input = np.isscalar(t)

    # convert to array for unified processing
    t = np.atleast_1d(t).astype(float)

    p = normalize(p)
    q = normalize(q)

    if a == b:
        # degenerate: return p
        gamma = np.tile(p, (t.size, 1))
        return gamma[0] if scalar_input else gamma

    # parameter s ∈ [0, 1]
    s = (t - a) / (b - a)
    s = np.clip(s, 0.0, 1.0)

    cos_omega = np.clip(dot(p, q), -1.0, 1.0)
    omega = np.arccos(cos_omega)

    # p ≈ q → linear interpolation + normalization
    if omega < 1e-8:
        gamma = (1 - s)[:, None] * p + s[:, None] * q
        gamma = normalize(gamma)
        return gamma[0] if scalar_input else gamma

    sin_omega = np.sin(omega)

    coef_p = np.sin((1 - s) * omega) / sin_omega
    coef_q = np.sin(s * omega) / sin_omega

    gamma = coef_p[:, None] * p + coef_q[:, None] * q
    gamma = normalize(gamma)

    return gamma[0] if scalar_input else gamma



def piecewise_geodesic(t: np.ndarray,
                       control_points: np.ndarray,
                       knots: np.ndarray) -> np.ndarray:
    """
    Compute a piecewise geodesic path between control points.

    Parameters
    ----------
    t : array-like, shape (n,)
        Time points.
    control_points : ndarray, shape (m, 3)
        Control points on the sphere. Each row is a control point.
    knots : ndarray, shape (m,)
        Knot values defining the segments. Typically increasing.

    Returns
    -------
    gamma : ndarray, shape (n, 3)
        Piecewise geodesic path evaluated at t.
    """
    t = np.asarray(t, dtype=float)
    cp = np.asarray(control_points, dtype=float)
    knots = np.asarray(knots, dtype=float)

    if cp.ndim != 2 or cp.shape[1] != 3:
        raise ValueError("control_points must have shape (m, 3).")
    if knots.ndim != 1 or knots.size != cp.shape[0]:
        raise ValueError("knots must be a 1D array of length equal to the number of control points.")
    if np.any(np.diff(knots) < 0):
        raise ValueError("knots must be non-decreasing.")

    m = cp.shape[0]
    n = t.size
    gamma = np.zeros((n, 3), dtype=float)

    # For each segment [knots[i], knots[i+1]] between cp[i] and cp[i+1]
    for i in range(m - 1):
        a = knots[i]
        b = knots[i + 1]
        mask = (t >= a) & (t <= b) if i < m - 2 else (t >= a) & (t <= b + 1e-12)
        if not np.any(mask):
            continue

        gamma[mask, :] = geodesic(t[mask], cp[i, :], cp[i + 1, :], a, b)

    # For t outside the knot range, extend using end control points
    min_k, max_k = knots[0], knots[-1]
    before_mask = t < min_k
    after_mask = t > max_k
    if np.any(before_mask):
        gamma[before_mask, :] = cp[0, :]
    if np.any(after_mask):
        gamma[after_mask, :] = cp[-1, :]

    return normalize(gamma)
