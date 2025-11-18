"""
geometry.py — Public spherical geometry utilities for spheresmooth

이 모듈은 R 패키지 `spheresmooth`에서 export된 함수들과 동일한 기능을 제공한다.
함수들은 사용자에게 노출되는 API이며, 내부 gradient/penalty 등 복잡한 계산은
_internal.py 에 숨겨져 있다.

제공 기능:
- 기본 벡터 연산(dot, cross, norm2)
- 벡터/행렬 정규화(normalize_lower, normalize)
- 구면 거리(spherical_dist)
- 지수 사상(exp_map, R 버전과 동일)
- 지오데식 경로 계산(geodesic_lower, geodesic)
- 조각별 지오데식(piecewise_geodesic)

이 파일은 smoothing.py 와 직접 연동되며,
R spheresmooth 의 piecewise geodesic 구조를 그대로 따른다.
"""

import numpy as np

from ._internal import (
    _Acos,
)


# ================================================================
# Basic operations
# ================================================================

def dot(u, v):
    """
    Compute the dot product of two vectors.

    Parameters
    ----------
    u, v : array-like (3,)
        Input vectors.

    Returns
    -------
    float
        Dot product <u, v>.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    return float(np.dot(u, v))


def cross(u: np.ndarray, v: np.ndarray, normalize_vec: bool = False) -> np.ndarray:
    """
    Compute the cross product u × v.

    Parameters
    ----------
    u, v : array-like (3,)
        Input vectors.
    normalize : bool, default=False
        If True, return the normalized cross product.

    Returns
    -------
    ndarray (3,)
        Cross product vector.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    w = np.cross(u, v)
    if normalize_vec:
        n = np.linalg.norm(w)
        if n > 0:
            w = w / n
    return w


def norm2(u):
    """
    Compute the L2 norm of a vector.

    Returns
    -------
    float
    """
    u = np.asarray(u, float)
    return float(np.linalg.norm(u))


# ================================================================
# Normalization
# ================================================================

def normalize_lower(v):
    """
    Normalize a single vector using the L2 norm.

    This corresponds exactly to R's normalize_lower().

    Zero vector remains unchanged.

    Parameters
    ----------
    v : array-like (3,)

    Returns
    -------
    ndarray (3,)
    """
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def normalize(X):
    """
    Normalize a vector OR each row of a matrix.

    Matches R's normalize() which uses apply(x, 1, normalize_lower).

    Parameters
    ----------
    X : array-like
        Vector (d,) or matrix (n, d)

    Returns
    -------
    ndarray
        Normalized vector or row-normalized matrix.
    """
    X = np.asarray(X, float)

    if X.ndim == 1:
        return normalize_lower(X)

    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        out[i] = normalize_lower(X[i])
    return out


# ================================================================
# Spherical distance (export)
# ================================================================

def spherical_dist(x, y):
    """
    Spherical distance between x and y on the unit sphere:
        dist(x, y) = Acos(<x, y>)

    Parameters
    ----------
    x, y : array-like (3,)

    Returns
    -------
    float
        Geodesic distance in radians.
    """
    x = normalize_lower(x)
    y = normalize_lower(y)
    return float(_Acos(dot(x, y)))


# ================================================================
# Exponential map (export)
# ================================================================

def exp_map(x, v):
    """
    Exponential map on S^2.

    R version:
        if sum(v^2) == 0: return(x)
        Exp = cos(norm_v) * x + sin(norm_v) * v / norm_v

    R 패키지 특성을 고려해 tangent-plane projection은 수행하지 않는다.

    Parameters
    ----------
    x : array-like (3,)
    v : array-like (3,)

    Returns
    -------
    ndarray (3,)
    """
    x = normalize_lower(x)
    v = np.asarray(v, float)

    if np.sum(v * v) == 0:
        return x

    nv = np.linalg.norm(v)
    return np.cos(nv) * x + np.sin(nv) * v / nv


# ================================================================
# Geodesic (export, uses internal)
# ================================================================

def geodesic_lower(t, p, q, a, b):
    """
    Compute a single geodesic point on S^2 between points p and q
    at a specific time t ∈ [a, b].

    This follows the R implementation exactly:
        n = cross(p, q, normalize=TRUE)
        w = cross(n, p, normalize=TRUE)
        theta(t) = dist(p,q) * (t - a) / (b - a)
        gamma = p * cos(theta) + w * sin(theta)

    Parameters
    ----------
    t : float
        Time scalar.
    p, q : array-like (3,)
        Endpoints on the sphere.
    a, b : float
        Start and end times.

    Returns
    -------
    ndarray (3,)
        A single point on S^2.
    """

    p = normalize(p)
    q = normalize(q)

    omega = spherical_dist(p, q)
    if omega < 1e-12:
        return p.copy()

    s = (t - a) / (b - a)
    s = np.clip(s, 0.0, 1.0)

    coef_p = np.sin((1 - s) * omega) / np.sin(omega)
    coef_q = np.sin(s * omega) / np.sin(omega)

    gamma = coef_p * p + coef_q * q
    return normalize(gamma)


def geodesic(t, p, q, a, b):
    """
    Compute geodesic on S2 between p and q at times t ∈ [a,b].
    """
    # scalar input → return (3,)
    if np.isscalar(t):
        t = float(t)
        p = normalize(p)
        q = normalize(q)

        dist_pq = spherical_dist(p, q)
        if dist_pq < 1e-12:
            return p.copy()

        theta = dist_pq * (t - a) / (b - a)

        # compute w = (n × p)
        n = np.cross(p, q)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            return p.copy()
        n = n / n_norm
        w = np.cross(n, p)
        w = w / np.linalg.norm(w)

        gamma = np.cos(theta) * p + np.sin(theta) * w
        return normalize(gamma)

    # vector input → apply geodesic_lower to each t
    t = np.asarray(t, float)
    out = np.zeros((len(t), 3))
    for i, ti in enumerate(t):
        out[i] = geodesic_lower(ti, p, q, a, b)
    return out



# ================================================================
# Piecewise geodesic
# ================================================================

def piecewise_geodesic(t, control_points, knots):
    """
    Replicates EXACT R behavior of piecewise_geodesic():
    - Iterate segments j = 1..K-1
    - For each segment, take t satisfying knots[j] <= t < knots[j+1]
    - Compute geodesic_lower for each t_sub in order
    - Append in the same order (rbind in R)
    - Does NOT reorder gamma to match global t order
    """

    t = np.asarray(t, float)
    cp = np.asarray(control_points, float)
    knots = np.asarray(knots, float)

    K = cp.shape[0]
    gamma_list = []

    for j in range(K - 1):
        a = knots[j]
        b = knots[j + 1]

        mask = (t >= a) & (t < b)
        t_sub = t[mask]

        if t_sub.size == 0:
            continue

        # R: geodesic(t_sub, ...)
        piece_gamma = geodesic(
            t_sub,
            cp[j],
            cp[j+1],
            a,
            b
        )

        gamma_list.append(piece_gamma)

    if len(gamma_list) == 0:
        return np.zeros((0,3))

    return np.vstack(gamma_list)
