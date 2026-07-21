import numpy as np
from scipy.spatial import ConvexHull

OUT_DIR = "../assets/urdf/objects/meshes/set2"


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for face in faces:
            idx = " ".join(str(i + 1) for i in face)  # OBJ is 1-indexed
            f.write("f %s\n" % idx)


def oriented_convex_hull(points):
    """Convex hull with consistently outward-facing triangle winding."""
    points = np.asarray(points, dtype=float)
    hull = ConvexHull(points)
    verts = hull.points
    centroid = verts.mean(axis=0)
    faces = []
    for simplex in hull.simplices:
        a, b, c = verts[simplex[0]], verts[simplex[1]], verts[simplex[2]]
        normal = np.cross(b - a, c - a)
        face_center = (a + b + c) / 3.0
        if np.dot(normal, face_center - centroid) < 0:
            simplex = simplex[::-1]
        faces.append(list(simplex))
    return verts, faces


def fibonacci_sphere_points(n, radius=1.0):
    i = np.arange(0, n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden = np.pi * (1 + 5 ** 0.5)
    theta = golden * i
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return radius * np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# ball_1: sphere, radius 1.0 (matches existing objects' ~2.0-unit bounding box)
# ---------------------------------------------------------------------------
def make_ball_1():
    pts = fibonacci_sphere_points(200, radius=1.0)
    return oriented_convex_hull(pts)


# ---------------------------------------------------------------------------
# ball_4: ellipsoid, sphere squashed on y/z (x stays full radius 1.0, y/z at 0.8)
# same "one axis compressed to 0.8x" pattern already used by thin_block /
# cylinder_axis in the existing object set.
# ---------------------------------------------------------------------------
def make_ball_4():
    pts = fibonacci_sphere_points(200, radius=1.0)
    pts = pts * np.array([1.0, 0.8, 0.8])
    return oriented_convex_hull(pts)


# ---------------------------------------------------------------------------
# ball_2: regular dodecahedron (12 pentagonal faces -> triangulated by hull),
# circumradius normalized to 1.0
# ---------------------------------------------------------------------------
def make_ball_2():
    phi = (1 + 5 ** 0.5) / 2.0
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append((sx * 1, sy * 1, sz * 1))
    for sy in (-1, 1):
        for sz in (-1, 1):
            pts.append((0, sy * (1 / phi), sz * phi))
    for sx in (-1, 1):
        for sy in (-1, 1):
            pts.append((sx * (1 / phi), sy * phi, 0))
    for sx in (-1, 1):
        for sz in (-1, 1):
            pts.append((sx * phi, 0, sz * (1 / phi)))
    pts = np.array(pts, dtype=float)
    # Normalize by the max single-axis coordinate (not circumradius) so the
    # axis-aligned bounding box matches the ~2.0-unit envelope used by the
    # rest of the object set, instead of just the circumscribed-sphere radius.
    half_extent = np.abs(pts).max()
    pts = pts / half_extent
    return oriented_convex_hull(pts)


# ---------------------------------------------------------------------------
# ball_3: regular icosahedron (20 triangular faces), bounding-box normalized to 2.0
# ---------------------------------------------------------------------------
def make_ball_3():
    phi = (1 + 5 ** 0.5) / 2.0
    pts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            pts.append((0, s1 * 1, s2 * phi))
            pts.append((s1 * 1, s2 * phi, 0))
            pts.append((s1 * phi, 0, s2 * 1))
    pts = np.array(pts, dtype=float)
    half_extent = np.abs(pts).max()
    pts = pts / half_extent
    return oriented_convex_hull(pts)


# ---------------------------------------------------------------------------
# cylinder_3: regular decagon (10-sided) prism, circumradius 1.0, height 2.0
# (matches existing cylinder objects' 2.0 x 2.0 x 2.0 bounding box)
# ---------------------------------------------------------------------------
def make_cylinder_3():
    n = 10
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # radius 1.0
    top = np.concatenate([ring, np.ones((n, 1))], axis=1)
    bottom = np.concatenate([ring, -np.ones((n, 1))], axis=1)
    pts = np.concatenate([top, bottom], axis=0)
    return oriented_convex_hull(pts)


# ---------------------------------------------------------------------------
# block_3: "stair" block = cube (2x2x2, matching set_obj1_regular_block) with
# one full-height quarter (x in [0,1], y in [0,1]) removed -> L-shaped
# (hexagonal cross-section) prism. Non-convex, so:
#   - visual mesh: manually extruded L-hexagon (NOT a convex hull, that would
#     fill the notch back in)
#   - collision mesh: split into 2 convex boxes that exactly tile the L shape
# ---------------------------------------------------------------------------
def make_block_3_visual():
    # L-shaped hexagon cross-section (cube minus x in [0,1], y in [0,1] quadrant)
    hexagon = [(-1, -1), (1, -1), (1, 0), (0, 0), (0, 1), (-1, 1)]
    bottom = [(x, y, -1) for (x, y) in hexagon]
    top = [(x, y, 1) for (x, y) in hexagon]
    verts = bottom + top  # 0..5 bottom ring, 6..11 top ring
    n = len(hexagon)
    faces = []
    # side walls (2 triangles per edge), outward-facing given CCW hexagon + this winding
    for i in range(n):
        j = (i + 1) % n
        b_i, b_j = i, j
        t_i, t_j = i + n, j + n
        faces.append([b_i, b_j, t_j])
        faces.append([b_i, t_j, t_i])
    # bottom cap (fan), normal should point down (-z) -> reverse order vs a CCW-from-above fan
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
    # top cap (fan), normal should point up (+z)
    for i in range(1, n - 1):
        faces.append([n, n + i, n + i + 1])
    return np.array(verts, dtype=float), faces


def make_block_3_collision_parts():
    # Box A: x in [-1,1], y in [-1,0], z in [-1,1]
    box_a = []
    for sx in (-1, 1):
        for sy in (-1, 0):
            for sz in (-1, 1):
                box_a.append((sx, sy, sz))
    # Box B: x in [-1,0], y in [0,1], z in [-1,1]
    box_b = []
    for sx in (-1, 0):
        for sy in (0, 1):
            for sz in (-1, 1):
                box_b.append((sx, sy, sz))
    return oriented_convex_hull(box_a), oriented_convex_hull(box_b)


def save(name, verts, faces):
    write_obj("%s/%s.obj" % (OUT_DIR, name), verts, faces)
    print("wrote", name, "verts=%d faces=%d" % (len(verts), len(faces)))


if __name__ == "__main__":
    v, f = make_ball_1()
    save("ball_1", v, f)
    save("ball_1_0decompose", v, f)

    v, f = make_ball_4()
    save("ball_4", v, f)
    save("ball_4_0decompose", v, f)

    v, f = make_ball_2()
    save("ball_2", v, f)
    save("ball_2_0decompose", v, f)

    v, f = make_ball_3()
    save("ball_3", v, f)
    save("ball_3_0decompose", v, f)

    v, f = make_cylinder_3()
    save("cylinder_3", v, f)
    save("cylinder_3_0decompose", v, f)

    v, f = make_block_3_visual()
    save("block_3", v, f)
    (va, fa), (vb, fb) = make_block_3_collision_parts()
    save("block_3_0decompose", va, fa)
    save("block_3_1decompose", vb, fb)
