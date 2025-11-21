# spheresmooth (Python)

A Python port of the **spheresmooth** R package for spherical geometry and spline smoothing on the unit sphere.  
The library provides utilities for coordinate transforms, geodesics, and penalized spherical smoothing using piecewise geodesic splines.

This project is a work-in-progress port of the original R implementation, with many components rewritten in clean, modular Python.

---

## Installation

```
pip install -e .
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

---

## Project Structure

```
spheresmooth/
    __init__.py
    coords.py
    geometry.py
    smoothing.py
    data.py
docs/
    source/
        api/
        gallery/
        generated/
```

---

## Documentation

Documentation is generated using **Sphinx + Sphinx-Gallery**.

```
cd docs
python -m sphinx -b html source build/html
```

---

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

## Example Gallery

Examples live in:

```
docs/source/gallery/
```

---

