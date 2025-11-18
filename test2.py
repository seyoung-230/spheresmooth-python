import numpy as np
import plotly.graph_objects as go
from spheresmooth import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    spherical_dist,
    piecewise_geodesic,
    penalized_linear_spherical_spline,
    geodesic
)

c = 1/np.sqrt(3)
control_points = np.array([
    [ c,  c,  c],
    [ c,  c, -c],
    [-c,  c,  c],
    [-c,  c, -c]
])
knots = np.array([1, 2, 3, 3.5]) 

# --- compute curve ---
t = np.linspace(0, 4, 1000)
curve = piecewise_geodesic(t, control_points, knots)

# --- sphere mesh ---
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))

# --- plot ---
fig = go.Figure()

fig.add_surface(x=x, y=y, z=z, opacity=0.05, colorscale="Greys")

fig.add_trace(go.Scatter3d(
    x=control_points[:,0],
    y=control_points[:,1],
    z=control_points[:,2],
    mode='markers',
    marker=dict(size=4, color='blue')
))

fig.add_trace(go.Scatter3d(
    x=curve[:,0],
    y=curve[:,1],
    z=curve[:,2],
    mode='lines',
    line=dict(color='red', width=5)
))

fig.update_layout(
    scene=dict(aspectmode='data'),
    title="Piecewise Geodesic on Sphere"
)

fig.show()