import numpy as np
import open3d as o3d
import trimesh
from math import radians, cos, sin

# -----------------------------
# Utility: build rigid transform
# -----------------------------
def make_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# -----------------------------
# Load bunny & sample points
# -----------------------------
bunny = o3d.data.BunnyMesh()
mesh_o3d = o3d.io.read_triangle_mesh(bunny.path)

pcd = mesh_o3d.sample_points_uniformly(5000)
points1 = np.asarray(pcd.points)

# -----------------------------
# Create transformed bunny
# -----------------------------
theta = radians(30)

R = np.array([
    [ cos(theta), 0, sin(theta)],
    [ 0,          1, 0         ],
    [-sin(theta), 0, cos(theta)]
])

t = np.array([0.03, 0.155, 0.02])  # small translation

T_bunny = make_transform(R, t)

# Apply transform
points2 = (points1 @ R.T) + t


# -----------------------------
# Compute OBBs
# -----------------------------
def compute_obb(points, color):
    T, extents = trimesh.bounds.oriented_bounds(points)
    T = np.linalg.inv(T)  # 🔴 critical

    obb = trimesh.creation.box(extents=extents, transform=T)
    obb.visual.face_colors = color
    return obb, extents


obb1, extents1 = compute_obb(points1, [0, 150, 255, 80])
obb2, extents2 = compute_obb(points2, [255, 100, 0, 80])

# -----------------------------
# Compute intersection volume
# -----------------------------
inter = obb1.intersection(obb2)

if inter.is_volume:
    vol_inter = inter.volume
else:
    vol_inter = 0.0

vol1 = np.prod(extents1)
vol2 = np.prod(extents2)

ratio_1 = vol_inter / vol1
ratio_2 = vol_inter / vol2

print(f"OBB1 volume: {vol1:.6f}")
print(f"OBB2 volume: {vol2:.6f}")
print(f"Intersection volume: {vol_inter:.6f}")
print(f"Intersection / OBB1: {ratio_1:.3f}")
print(f"Intersection / OBB2: {ratio_2:.3f}")


# -----------------------------
# Visualization
# -----------------------------
pc1 = trimesh.points.PointCloud(
    points1, colors=[200, 200, 200, 255]
)

pc2 = trimesh.points.PointCloud(
    points2, colors=[120, 120, 120, 255]
)

axes = trimesh.creation.axis(
    origin_size=0.01,
    axis_length=0.1
)

scene = trimesh.Scene([pc1, pc2, obb1, obb2, axes])
scene.show()