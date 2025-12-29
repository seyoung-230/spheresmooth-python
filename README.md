# spheresmooth (Python)

A Python port of the **spheresmooth** R package for spherical geometry and spline smoothing on the unit sphere.  
The library provides utilities for coordinate transforms, geodesics, and penalized spherical smoothing using piecewise geodesic splines.

This project is a work-in-progress port of the original R implementation, with many components rewritten in clean, modular Python.

---

## License
GPL-3
---

## Installation

```
pip install spheresmooth
```

---

## Requirements

`spheresmooth` requires the following dependencies:

### Core dependencies

| Package | Version (recommended) | Description |
|--------|------------------------|-------------|
| Python | 3.9+                   | Core language requirement |
| NumPy  | >= 1.24                | Numerical computations |
| pandas | >= 1.5                 | Internal data loading and handling (`import pandas as pd`) |
| importlib.resources | stdlib (Python ≥ 3.9) | Access to packaged data files |

### Optional dependencies (examples and visualization)

| Package     | Version (recommended) | Description |
|-------------|------------------------|-------------|
| matplotlib  | >= 3.7                 | Plotting and visualization in example scripts (`import matplotlib.pyplot as plt`) |
| geopandas   | >= 0.13                | Geographic data handling and map-based examples (`import geopandas as gpd`) |

The core library uses `pandas` and `importlib.resources` solely for internal
data loading.  
`importlib.resources` is part of the Python standard library (Python ≥ 3.9) and
does not need to be installed separately.

Visualization-related dependencies (`matplotlib`, `geopandas`) are required
**only** for running example scripts and generating figures. They are not
needed for using the core functionality of the library.

If you install `spheresmooth` via **pip**, the core dependencies will be
installed automatically:

```bash
pip install spheresmooth
```

To install the optional dependencies for examples and visualization, use:

```bash
pip install spheresmooth[viz]
```

---

## Features

### ✔ Coordinate Transformations
- Convert Cartesian ↔ Spherical coordinates  
- Batch processing with NumPy  
- Consistent handling of row-wise/column-wise inputs  

### ✔ Geometry Utilities
- Compute geodesics on the sphere  
- Normalize vectors  
- Spherical distance functions  
- Projection and gradient operators  

### ✔ Smoothing Functions (In Progress)
- Penalized piecewise geodesic spline smoothing  
- R function `penalized_linear_spherical_spline()`의 Python 버전 구조 구현 완료  
- Full Riemannian optimization is currently under development  


## Example

```python
import numpy as np
from spheresmooth import cartesian_to_spherical

points = np.array([
    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
    [-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],
])

theta_phi = cartesian_to_spherical(points)
print(theta_phi)
```

---
