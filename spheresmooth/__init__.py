
"""
spheresmooth: basic spherical geometry and piecewise geodesic paths.

This is a Python scaffolding for methods similar to the R package `spheresmooth`.
"""

from .coords import (
    cartesian_to_spherical,
    spherical_to_cartesian,
    edp,
)
from .geometry import (
    dot,
    cross,
    norm2,
    normalize,
    spherical_dist,
    exp_map,
    geodesic,
    piecewise_geodesic,
)
from .smoothing import (
    calculate_loss,
    knots_quantile,
    penalized_linear_spherical_spline,
)

__all__ = [
    # coords
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "edp",
    # geometry
    "dot",
    "cross",
    "norm2",
    "normalize",
    "normalize_lower",
    "spherical_dist",
    "exp_map",
    "geodesic",
    "geodesic_lower",
    "piecewise_geodesic",
    # smoothing
    "calculate_loss",
    "knots_quantile",
    "penalized_linear_spherical_spline",
]
