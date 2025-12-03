"""
Piecewise Geodesic Example
==========================

Compute and visualize a piecewise geodesic curve on the unit sphere.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from spheresmooth import piecewise_geodesic

# -------------------------------------------------------------
# Example setup
# -------------------------------------------------------------
c = 1/np.sqrt(3)
control_points = np.array([
    [ c,  c,  c],
    [ c,  c, -c],
    [-c,  c,  c],
    [-c,  c, -c]
])
knots = np.array([1, 2, 3, 3.5])

t = np.linspace(0, 4, 300)
curve = piecewise_geodesic(t, control_points, knots)

# -------------------------------------------------------------
# Sphere mesh
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

# Sphere surface
fig.add_surface(
    x=x, y=y, z=z,
    opacity=0.1,
    colorscale="Greys",
    showscale=False
)

# Control points
fig.add_trace(go.Scatter3d(
    x=control_points[:, 0],
    y=control_points[:, 1],
    z=control_points[:, 2],
    mode='markers',
    marker=dict(size=5, color='blue', opacity=1),
    name='Control Points' # 범례 추가
))

# Geodesic path
fig.add_trace(go.Scatter3d(
    x=curve[:, 0],
    y=curve[:, 1],
    z=curve[:, 2],
    mode='lines',
    line=dict(color='red', width=6),
    name='Piecewise Geodesic' # 범례 추가
))

# Optional: lat/long grid (간결성을 위해 생략 가능하나 원본 유지)
lat_values = np.linspace(-60, 60, 8)
for lat in lat_values:
    phi = np.radians(lat)
    ugrid = np.linspace(0, 2*np.pi, 200)
    fig.add_trace(go.Scatter3d(
        x=np.cos(ugrid) * np.cos(phi),
        y=np.sin(ugrid) * np.cos(phi),
        z=np.full_like(ugrid, np.sin(phi)),
        mode='lines',
        line=dict(color='gray', width=1),
        showlegend=False
    ))

lon_values = np.linspace(0, 330, 12)
for lon in lon_values:
    lam = np.radians(lon)
    vgrid = np.linspace(-np.pi/2, np.pi/2, 200)
    fig.add_trace(go.Scatter3d(
        x=np.cos(vgrid) * np.cos(lam),
        y=np.cos(vgrid) * np.sin(lam),
        z=np.sin(vgrid),
        mode='lines',
        line=dict(color='gray', width=1),
        showlegend=False
    ))

fig.update_layout(
    title="Piecewise Geodesic on the Sphere",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data"
    )
)

# -------------------------------------------------------------
# Sphinx Gallery/독립 실행 환경을 위해 Plotly 그래프를 이미지로 저장
# 이 부분이 Sphinx Gallery에서 이미지를 캡처하도록 합니다.
# pio.write_image를 사용하려면 'kaleido' 패키지가 설치되어 있어야 합니다.
# -------------------------------------------------------------
try:
    pio.write_image(fig, "piecewise_geodesic_example.png", scale=2)
    # 이미지 저장이 성공하면, Sphinx Gallery가 이 이미지를 캡처합니다.
except ValueError:
    print("Warning: 'kaleido' not installed. Cannot save static image.")
    # 'kaleido'가 설치되어 있지 않으면 정적 이미지 저장에 실패할 수 있습니다.
    
# IPython/Notebook 환경이 아니라면 아래 HTML 출력 코드는 필요하지 않습니다.
# from IPython.display import HTML
# HTML(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))