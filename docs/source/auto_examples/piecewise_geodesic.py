"""
Piecewise Geodesic Example
==========================

Compute and visualize a piecewise geodesic curve on the unit sphere.
"""
import plotly.io as pio
pio.renderers.default = "sphinx_gallery"
import numpy as np
import plotly.graph_objects as go
from spheresmooth import piecewise_geodesic

# -------------------------------------------------------------
# Define control points and knots
# -------------------------------------------------------------
c = 1/np.sqrt(3)
control_points = np.array([
    [ c,  c,  c],
    [ c,  c, -c],
    [-c,  c,  c],
    [-c,  c, -c]
])

knots = np.array([1, 2, 3, 3.5])

# Time grid
t = np.linspace(0, 4, 300)

# Compute piecewise geodesic
curve = piecewise_geodesic(t, control_points, knots)

# -------------------------------------------------------------
# Sphere mesh grid
# -------------------------------------------------------------
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))

# -------------------------------------------------------------
# Plot
# -------------------------------------------------------------
fig = go.Figure()

# Sphere
fig.add_surface(
    x=x, y=y, z=z,
    opacity=0.1,
    colorscale="Greys",
    showscale=False
)

# Control points
fig.add_trace(go.Scatter3d(
    x=control_points[:,0],
    y=control_points[:,1],
    z=control_points[:,2],
    mode='markers',
    marker=dict(size=5, color='blue')
))

# Piecewise geodesic path
fig.add_trace(go.Scatter3d(
    x=curve[:,0],
    y=curve[:,1],
    z=curve[:,2],
    mode='lines',
    line=dict(color='red', width=6)
))

# -------------------------------------------------------------
# Optional: latitude/longitude grid
# -------------------------------------------------------------
lat_values = np.linspace(-60, 60, 8)
for lat in lat_values:
    phi = np.radians(lat)
    u = np.linspace(0, 2*np.pi, 200)
    x_lat = np.cos(u) * np.cos(phi)
    y_lat = np.sin(u) * np.cos(phi)
    z_lat = np.full_like(u, np.sin(phi))
    fig.add_trace(go.Scatter3d(
        x=x_lat, y=y_lat, z=z_lat,
        mode='lines',
        line=dict(color='gray', width=1),
        showlegend=False
    ))

lon_values = np.linspace(0, 330, 12)
for lon in lon_values:
    lam = np.radians(lon)
    v = np.linspace(-np.pi/2, np.pi/2, 200)
    x_lon = np.cos(v) * np.cos(lam)
    y_lon = np.cos(v) * np.sin(lam)
    z_lon = np.sin(v)
    fig.add_trace(go.Scatter3d(
        x=x_lon, y=y_lon, z=z_lon,
        mode='lines',
        line=dict(color='gray', width=1),
        showlegend=False
    ))

# Layout
fig.update_layout(
    title="Piecewise Geodesic on the Sphere",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data"
    )
)

fig.show()

html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
print(html)