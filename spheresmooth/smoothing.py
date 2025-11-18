import numpy as np

from .geometry import spherical_dist, piecewise_geodesic, normalize
from .coords import spherical_to_cartesian, cartesian_to_spherical


def calculate_loss(y: np.ndarray, gamma: np.ndarray) -> float:
    """
    Calculate loss based on squared spherical distances
    between observed values and predicted values on the curve.

    Parameters
    ----------
    y : ndarray, shape (n, 3)
        Observed points on the sphere (Cartesian).
    gamma : ndarray, shape (n, 3)
        Predicted points on the sphere (Cartesian).

    Returns
    -------
    float
        Loss value (sum of squared spherical distances).
    """
    y = np.asarray(y, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    if y.shape != gamma.shape:
        raise ValueError("y and gamma must have the same shape.")

    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("y and gamma must have shape (n, 3).")

    y = normalize(y)
    gamma = normalize(gamma)

    dists = np.array([spherical_dist(y[i], gamma[i]) for i in range(y.shape[0])])
    return float(np.sum(dists ** 2))


def knots_quantile(x: np.ndarray,
                   dimension: int,
                   tiny: float = 1e-5) -> np.ndarray:
    """
    Generate knots for the piecewise geodesic curve based on quantiles.

    Parameters
    ----------
    x : array-like
        Numeric vector representing time points.
    dimension : int
        Number of knots.
    tiny : float, default=1e-5
        Small constant to slightly expand the boundary.

    Returns
    -------
    knots : ndarray, shape (dimension,)
        Knot positions in the time domain.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array of time points.")
    if dimension < 2:
        raise ValueError("dimension must be at least 2.")

    x_min = x.min()
    x_max = x.max()
    # Slightly expand boundaries
    lower = x_min - tiny
    upper = x_max + tiny
    probs = np.linspace(0, 1, num=dimension)
    knots = lower + (upper - lower) * probs
    return knots


def penalized_linear_spherical_spline(
    t: np.ndarray,
    y: np.ndarray,
    initial_control_points: np.ndarray | None = None,
    dimension: int | None = None,
    initial_knots: np.ndarray | None = None,
    lambdas: np.ndarray | None = None,
    step_size: float = 1.0,
    maxiter: int = 1000,
    epsilon_iter: float = 1e-3,
    jump_eps: float = 1e-4,
    verbose: bool = False,
):
    """
    Penalized linear spherical spline (scaffolding).

    This function is intended to fit a penalized piecewise geodesic curve
    to spherical data, similarly to the R function
    `penalized_linear_spherical_spline` in the spheresmooth package.

    For now, this implementation only sets up the interface and basic checks.
    The full Riemannian optimization algorithm should be implemented here.

    Parameters
    ----------
    t : ndarray, shape (n,)
        Time or location.
    y : ndarray, shape (n, 3)
        Data points on the sphere (Cartesian).
    initial_control_points : ndarray or None
        Optional initial control points.
    dimension : int or None
        Dimension of the spline (number of control points).
        Required if `initial_control_points` is None.
    initial_knots : ndarray or None
        Optional initial knot vector. If None, it can be generated
        from `knots_quantile`.
    lambdas : ndarray or None
        Penalization parameters.
    step_size, maxiter, epsilon_iter, jump_eps, verbose :
        Optimization controls (not yet used).

    Returns
    -------
    dict
        Placeholder structure, expected to contain:
        - "fits": list of fit objects (one per lambda)
        - "bic_list": list/array of BIC values

    Notes
    -----
    TODO: Implement the full coordinate-wise gradient descent /
    Riemannian block coordinate descent algorithm described in the
    associated paper.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError("y must have shape (n, 3).")
    if t.size != y.shape[0]:
        raise ValueError("t and y must have compatible sizes (len(t) == y.shape[0]).")

    if lambdas is None:
        raise ValueError("lambdas (vector of penalty parameters) must be provided.")

    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.ndim != 1:
        raise ValueError("lambdas must be a 1D array.")

    n = t.size

    # Initialize control points if not provided
    if initial_control_points is None:
        if dimension is None:
            raise ValueError(
                "dimension must be provided when initial_control_points is None."
            )
        # Simple initialization: pick 'dimension' equally spaced points from y
        idx = np.linspace(0, n - 1, num=dimension, dtype=int)
        cp_init = y[idx, :]
    else:
        cp_init = np.asarray(initial_control_points, dtype=float)
        if cp_init.ndim != 2 or cp_init.shape[1] != 3:
            raise ValueError("initial_control_points must have shape (m, 3).")
        if dimension is None:
            dimension = cp_init.shape[0]

    # Initialize knots if not provided
    if initial_knots is None:
        knots = knots_quantile(t, dimension=dimension)
    else:
        knots = np.asarray(initial_knots, dtype=float)

    # Placeholder: for each lambda, we simply keep the same control points / knots.
    fits = []
    bic_list = []

    for lam in lambdas:
        # TODO: Replace with real optimization procedure.

        # For now, just construct the piecewise geodesic curve with cp_init & knots.
        gamma_hat = piecewise_geodesic(t, cp_init, knots)
        loss = calculate_loss(y, gamma_hat)
        # Very naive "BIC" placeholder: loss + lambda * (#control_points)
        bic = loss + float(lam) * cp_init.shape[0]

        fit_obj = {
            "control_points": cp_init.copy(),
            "knots": knots.copy(),
            "lambda": float(lam),
            "loss": loss,
        }
        fits.append(fit_obj)
        bic_list.append(bic)

        if verbose:
            print(f"[lambda={lam:.3e}] loss={loss:.5f}, BIC~={bic:.5f}")

    return {
        "fits": fits,
        "bic_list": np.array(bic_list, dtype=float),
    }
