"""
One-off script: regenerate the embedded MESH_DATA / CATALOG blobs inside
object_viewer.html for the current 9-object training set. ball_1..4 and the
original procedurally-generated stair block_3 were removed from training
(pass 2); else_1/else_5 were promoted to block_3/block_5 and the rest of the
else_* pool renumbered to else_1..8 (pass 2); else_1..8 were then excluded
from training entirely (pass 3) -- they still exist as valid assets under
assets/urdf/objects/else_*.urdf but are intentionally left out of this
viewer's catalog since it mirrors the actual training roster (object_sets
["C"] in allegro_arm_morb_axis.py), not the full assets folder.
"""
import json
import re

MESH_DIR = "../assets/urdf/objects/meshes/set2"
SCALE = 0.03

# key -> (display name, category, numeric id badge)
CATALOG_INFO = [
    ("block_1", "Regular Block", "Blocks", 1),
    ("block_2", "Irregular Block · Time", "Blocks", 2),
    ("block_3", "Block", "Blocks", 3),
    ("block_4", "Block Corner", "Blocks", 4),
    ("block_5", "Block", "Blocks", 5),
    ("cylinder_1", "Cylinder", "Cylinders", 1),
    ("cylinder_2", "Cylinder Axis", "Cylinders", 2),
    ("cylinder_3", "Decagon Prism", "Cylinders", 3),
    ("cylinder_4", "Cylinder Corner", "Cylinders", 4),
]


def resolve_visual_mesh_path(key):
    """The urdf's own filename doesn't always match its visual mesh's base
    name (e.g. block_4.urdf references meshes/set2/block_6.obj after a
    relabeling that only renamed the .urdf file) -- read the actual
    reference out of the urdf instead of assuming key == mesh basename."""
    with open("../assets/urdf/objects/%s.urdf" % key, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    m = re.search(r'<visual>.*?filename="meshes/set2/([^"]+\.obj)"', content, re.S)
    return "%s/%s" % (MESH_DIR, m.group(1))


def load_obj_visual(key):
    verts = []
    faces = []
    with open(resolve_visual_mesh_path(key), "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = map(float, line.split()[1:4])
                verts.append([x * SCALE, y * SCALE, z * SCALE])
            elif line.startswith("f "):
                idxs = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                # fan-triangulate n-gon faces
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return verts, faces


def build_mesh_entry(key):
    verts, faces = load_obj_visual(key)

    # Flat shading: give every triangle its own 3 vertex copies and its own
    # single face normal, instead of averaging normals across all faces that
    # share a vertex index. All 22 objects here are hard-edged polyhedra
    # (cube variants, prisms, dodecahedron/icosahedron, the L-shaped stair
    # block), so averaging normals across faces that meet at sharp angles
    # (e.g. 90 degrees at a cube corner) produced smeared/warped-looking
    # shading. Per-face flat normals render every face as a crisp, uniformly
    # lit flat surface, which is what these shapes actually look like.
    flat_verts = []
    flat_normals = []
    flat_faces = []
    for f in faces:
        a = verts[f[0]]; b = verts[f[1]]; c = verts[f[2]]
        ux, uy, uz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
        vx, vy, vz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
        nx, ny, nz = uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
        length = (nx**2 + ny**2 + nz**2) ** 0.5
        if length > 1e-12:
            nx, ny, nz = nx/length, ny/length, nz/length
        base = len(flat_verts)
        for idx in f:
            flat_verts.append(verts[idx])
            flat_normals.append([nx, ny, nz])
        flat_faces.append([base, base + 1, base + 2])

    verts, normals, faces = flat_verts, flat_normals, flat_faces
    n = len(verts)

    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    bbox = {"min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]}

    flat_pos = [c for v in verts for c in v]
    flat_norm = [c for v in normals for c in v]
    flat_idx = [i for f in faces for i in f]

    return {
        "positions": flat_pos,
        "normals": flat_norm,
        "indices": flat_idx,
        "bbox": bbox,
        "vertexCount": n,
        "faceCount": len(faces),
    }


mesh_data = {}
for key, name, cat, idnum in CATALOG_INFO:
    mesh_data[key] = build_mesh_entry(key)

mesh_data_json = json.dumps(mesh_data, separators=(",", ":"))

catalog_lines = []
for key, name, cat, idnum in CATALOG_INFO:
    catalog_lines.append(
        '  { key: "%s", id: %d, name: "%s", cat: "%s" },' % (key, idnum, name, cat)
    )
catalog_js = "const CATALOG = [\n" + "\n".join(catalog_lines) + "\n];"

with open("object_viewer.html", "r", encoding="utf-8") as f:
    html = f.read()

# Replace the MESH_DATA line (starts with 'const MESH_DATA = ' up to the line's end)
html, n1 = re.subn(
    r"const MESH_DATA = \{.*?\};",
    "const MESH_DATA = " + mesh_data_json + ";",
    html,
    count=1,
    flags=re.S,
)

# Replace the CATALOG block (from 'const CATALOG = [' to the matching '];')
html, n2 = re.subn(
    r"const CATALOG = \[.*?\];",
    catalog_js,
    html,
    count=1,
    flags=re.S,
)

# Fix the subtitle builder that assumed a numeric "set_obj" id scheme
html, n3 = re.subn(
    r'"set_obj" \+ info\.id \+ " · " \+ info\.cat\.toLowerCase\(\);',
    'info.key + " · " + info.cat.toLowerCase();',
    html,
)

# Fix the category list used to group the sidebar (handles the pre-relabel
# state, the "Balls"-included state, and the "Else"-included state, since
# ball_* and else_* have since been removed from the training roster/viewer
# entirely -- only Blocks/Cylinders remain).
html, n4a = re.subn(
    r'const cats = \["Blocks", "Cylinders", "Irregular"\];',
    'const cats = ["Blocks", "Cylinders"];',
    html,
)
html, n4b = re.subn(
    r'const cats = \["Blocks", "Cylinders", "Balls", "Else"\];',
    'const cats = ["Blocks", "Cylinders"];',
    html,
)
html, n4c = re.subn(
    r'const cats = \["Blocks", "Cylinders", "Else"\];',
    'const cats = ["Blocks", "Cylinders"];',
    html,
)
n4 = n4a + n4b + n4c

assert n1 == 1, "MESH_DATA replace count: %d" % n1
assert n2 == 1, "CATALOG replace count: %d" % n2
# n3/n4 may already be 0 if this script has been run before (those fixes are
# idempotent - already applied and won't match a second time).
already_correct = 'info.key + " · " + info.cat.toLowerCase();' in html
assert n3 == 1 or already_correct, "subtitle fix count: %d" % n3
already_correct_cats = 'const cats = ["Blocks", "Cylinders"];' in html
assert n4 == 1 or already_correct_cats, "cats fix count: %d" % n4

with open("object_viewer.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Regenerated MESH_DATA for %d objects, CATALOG with %d entries." % (len(mesh_data), len(CATALOG_INFO)))
