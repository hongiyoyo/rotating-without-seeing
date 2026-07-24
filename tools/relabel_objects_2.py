"""
One-off script: second relabeling pass.
- Delete block_3 (stair L-shape) and ball_1..4 entirely.
- Promote else_1 -> block_3, else_5 -> block_5.
- Renumber the remaining else_2,3,4,6,7,8,9,10 -> else_1..else_8 sequentially.

Uses a 3-phase copy-via-temp-name -> delete-originals -> rename-temp-to-final
approach. This avoids the bug in the first version of this script: since
RENAME_MAP is a *chain* (e.g. else_2 -> else_1, else_3 -> else_2, ...), a
new_name can equal another pair's old_name, so copying directly to final
names and deleting "recorded old paths" afterwards can delete a file that
was already overwritten with different (wanted) content in the meantime.
Going through a temp name that never collides with any real object name
sidesteps that entirely.
"""
import glob
import os
import re
import shutil

MESH_DIR = "../assets/urdf/objects/meshes/set2"
URDF_DIR = "../assets/urdf/objects"
TMP_PREFIX = "__tmp2__"

RENAME_MAP = {
    "else_1": "block_3",
    "else_5": "block_5",
    "else_2": "else_1",
    "else_3": "else_2",
    "else_4": "else_3",
    "else_6": "else_4",
    "else_7": "else_5",
    "else_8": "else_6",
    "else_9": "else_7",
    "else_10": "else_8",
}
DELETE_OUTRIGHT = ["block_3", "ball_1", "ball_2", "ball_3", "ball_4"]

decompose_re = re.compile(r"^(.+)_(\d+)decompose\.obj$")


def mesh_files_for(name):
    visual = os.path.join(MESH_DIR, name + ".obj")
    parts = sorted(glob.glob(os.path.join(MESH_DIR, name + "_*decompose.obj")))
    # keep only exact-name matches (avoid e.g. "else_1" matching "else_10_...")
    parts = [p for p in parts if decompose_re.match(os.path.basename(p)).group(1) == name]
    return visual, parts


# ---------------------------------------------------------------------------
# Phase 1: copy old_name's files to a TEMP name (block_3 -> __tmp2__block_3)
# ---------------------------------------------------------------------------
tmp_info = {}  # new_name -> (n_parts,)
for old_name, new_name in RENAME_MAP.items():
    tmp_name = TMP_PREFIX + new_name
    old_visual, old_parts = mesh_files_for(old_name)

    shutil.copyfile(old_visual, os.path.join(MESH_DIR, tmp_name + ".obj"))
    for old_path in old_parts:
        idx = decompose_re.match(os.path.basename(old_path)).group(2)
        shutil.copyfile(old_path, os.path.join(MESH_DIR, "%s_%sdecompose.obj" % (tmp_name, idx)))

    old_urdf = os.path.join(URDF_DIR, old_name + ".urdf")
    with open(old_urdf, "r") as f:
        content = f.read()
    with open(os.path.join(URDF_DIR, tmp_name + ".urdf"), "w") as f:
        f.write(content.replace(old_name, new_name))

    tmp_info[new_name] = len(old_parts)
    print("%-8s -> %-8s (via temp, %d collision part%s)" % (
        old_name, new_name, len(old_parts), "" if len(old_parts) == 1 else "s"))

# ---------------------------------------------------------------------------
# Phase 2: delete ALL originals that are being replaced/removed -- every
# old_name in RENAME_MAP, plus the outright-deleted objects.
# ---------------------------------------------------------------------------
removed = []
for old_name in list(RENAME_MAP.keys()) + DELETE_OUTRIGHT:
    visual, parts = mesh_files_for(old_name)
    for p in [visual] + parts:
        if os.path.exists(p):
            os.remove(p)
            removed.append(p)
    urdf = os.path.join(URDF_DIR, old_name + ".urdf")
    if os.path.exists(urdf):
        os.remove(urdf)
        removed.append(urdf)
print("\nRemoved %d original files" % len(removed))

# ---------------------------------------------------------------------------
# Phase 3: rename temp files to their final names
# ---------------------------------------------------------------------------
finalized = []
for new_name in tmp_info:
    tmp_name = TMP_PREFIX + new_name
    tmp_visual, tmp_parts = mesh_files_for(tmp_name)
    os.rename(tmp_visual, os.path.join(MESH_DIR, new_name + ".obj"))
    finalized.append(new_name + ".obj")
    for p in tmp_parts:
        idx = decompose_re.match(os.path.basename(p)).group(2)
        os.rename(p, os.path.join(MESH_DIR, "%s_%sdecompose.obj" % (new_name, idx)))
        finalized.append("%s_%sdecompose.obj" % (new_name, idx))
    os.rename(os.path.join(URDF_DIR, tmp_name + ".urdf"), os.path.join(URDF_DIR, new_name + ".urdf"))
    finalized.append(new_name + ".urdf")

print("Finalized %d files:" % len(finalized), finalized)
