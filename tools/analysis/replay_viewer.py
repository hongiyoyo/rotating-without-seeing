# Phase 2: replay recorded failed episodes in the Isaac Gym viewer at real speed.
#   SPACE = next episode, R = replay this episode, B = previous episode, ESC / close window = quit.
# The episode freezes on its last frame; the end reason is drawn in the viewer (Isaac Gym has no
# text API, so text is drawn with line segments) and printed in the terminal.
#
# Playback is kinematic: each frame writes the recorded object pose and hand joint positions,
# then advances the sim by one step with gravity off and zero joint torques so nothing drifts.
# Playback follows the wall clock (1 recorded frame = 1/60 s); frames are skipped if rendering lags.
#
# Run in WSL (robosyn env):  python replay_viewer.py [--sort len] [--start N]
import isaacgym  # noqa: F401
from isaacgym import gymapi, gymtorch
import os, sys, math, time, pickle, argparse
REPO = "/mnt/c/YangJaehong/rotating-without-seeing"
sys.path.insert(0, REPO)
import numpy as np
import torch
from hydra import initialize_config_dir, compose
import isaacgymenvs

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="/mnt/c/YangJaehong/train_logs/viewer/block4_failed_episodes.pkl")
ap.add_argument("--sort", choices=["time", "len"], default="time", help="time = order they happened, len = shortest first")
ap.add_argument("--start", type=int, default=1, help="1-based episode number to start from")
args = ap.parse_args()

data = pickle.load(open(args.file, "rb"))
meta, episodes = data["meta"], data["episodes"]
if args.sort == "len":
    episodes = sorted(episodes, key=lambda e: e["control_steps"])
TARGET = meta["object"]
print(f"{len(episodes)} failed {TARGET} episodes ({meta['n_completed']} of {meta['n_total']} completed the full 50 s)")

# ---------------------------------------------------------------- env (viewer, no DR, no gravity)
with initialize_config_dir(config_dir=REPO + "/isaacgymenvs/cfg"):
    cfg = compose(config_name="config", overrides=[
        "task.env.numEnvs=9", "headless=False", "task.env.deterministicObjectAssign=True",
        "task.task.randomize=False", "task.sim.gravity=[0.0,0.0,0.0]"])
env = isaacgymenvs.make(cfg.seed, cfg.task_name, 9, cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
                        False, False, False, False, cfg)
gym, sim, viewer = env.gym, env.sim, env.viewer
if viewer is None:
    sys.exit("viewer could not be created")
names = list(env.used_training_objects)
E = names.index(TARGET)  # round-robin assignment: env E holds TARGET
env_ptr = env.envs[E]
ndof = env.num_arm_hand_dofs
obj_i = int(env.object_indices[E])
obj_idx_t = torch.tensor([obj_i], dtype=torch.int32, device=env.device)
hand_idx_t = torch.tensor([int(env.hand_indices[E])], dtype=torch.int32, device=env.device)
zero_torque = torch.zeros(env.num_envs * ndof, device=env.device)
for key, name in [(gymapi.KEY_SPACE, "next"), (gymapi.KEY_R, "replay"), (gymapi.KEY_B, "back")]:
    gym.subscribe_viewer_keyboard_event(viewer, key, name)

# camera on the target env's hand, from the front and above
target = env.object_init_state[E, 0:3].cpu().numpy().astype(float)
cam = target + np.array([0.32, 0.0, 0.22])
gym.viewer_camera_look_at(viewer, env_ptr, gymapi.Vec3(*cam), gymapi.Vec3(*target))
fwd = (target - cam) / np.linalg.norm(target - cam)
right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
up = np.cross(right, fwd)

# ---------------------------------------------------------------- stroke font (4 x 6 grid)
O_ = [[(0, 0), (0, 6), (4, 6), (4, 0), (0, 0)]]
S_ = [[(4, 6), (0, 6), (0, 3), (4, 3), (4, 0), (0, 0)]]
FONT = {
    "A": [[(0, 0), (0, 4), (2, 6), (4, 4), (4, 0)], [(0, 3), (4, 3)]],
    "B": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)], [(3, 3), (4, 2), (4, 1), (3, 0), (0, 0)]],
    "C": [[(4, 6), (0, 6), (0, 0), (4, 0)]],
    "D": [[(0, 0), (0, 6), (2, 6), (4, 4), (4, 2), (2, 0), (0, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]],
    "F": [[(4, 6), (0, 6), (0, 0)], [(0, 3), (3, 3)]],
    "G": [[(4, 6), (0, 6), (0, 0), (4, 0), (4, 3), (2, 3)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3), (4, 3)]],
    "I": [[(1, 6), (3, 6)], [(2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "J": [[(4, 6), (4, 0), (0, 0), (0, 2)]],
    "K": [[(0, 0), (0, 6)], [(4, 6), (0, 3), (4, 0)]],
    "L": [[(0, 6), (0, 0), (4, 0)]],
    "M": [[(0, 0), (0, 6), (2, 3), (4, 6), (4, 0)]],
    "N": [[(0, 0), (0, 6), (4, 0), (4, 6)]],
    "O": O_,
    "P": [[(0, 0), (0, 6), (4, 6), (4, 3), (0, 3)]],
    "Q": O_ + [[(2, 2), (4, -1)]],
    "R": [[(0, 0), (0, 6), (4, 6), (4, 3), (0, 3), (4, 0)]],
    "S": S_,
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]],
    "U": [[(0, 6), (0, 0), (4, 0), (4, 6)]],
    "V": [[(0, 6), (2, 0), (4, 6)]],
    "W": [[(0, 6), (1, 0), (2, 3), (3, 0), (4, 6)]],
    "X": [[(0, 0), (4, 6)], [(0, 6), (4, 0)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]],
    "Z": [[(0, 6), (4, 6), (0, 0), (4, 0)]],
    "0": O_ + [[(0, 0), (4, 6)]],
    "1": [[(1, 5), (2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "2": [[(0, 6), (4, 6), (4, 3), (0, 3), (0, 0), (4, 0)]],
    "3": [[(0, 6), (4, 6), (4, 0), (0, 0)], [(1, 3), (4, 3)]],
    "4": [[(0, 6), (0, 3), (4, 3)], [(4, 6), (4, 0)]],
    "5": S_,
    "6": [[(4, 6), (0, 6), (0, 0), (4, 0), (4, 3), (0, 3)]],
    "7": [[(0, 6), (4, 6), (1, 0)]],
    "8": O_ + [[(0, 3), (4, 3)]],
    "9": [[(4, 3), (0, 3), (0, 6), (4, 6), (4, 0), (0, 0)]],
    ".": [[(2, 0), (2, 0.7)]], ",": [[(2, 0.7), (1.4, -1)]], ":": [[(2, 1), (2, 1.7)], [(2, 4), (2, 4.7)]],
    "/": [[(0, 0), (4, 6)]], "-": [[(1, 3), (3, 3)]], "+": [[(1, 3), (3, 3)], [(2, 2), (2, 4)]],
    "=": [[(1, 2), (3, 2)], [(1, 4), (3, 4)]], "<": [[(4, 6), (0, 3), (4, 0)]], ">": [[(0, 6), (4, 3), (0, 0)]],
    "(": [[(3, 6), (2, 5), (2, 1), (3, 0)]], ")": [[(1, 6), (2, 5), (2, 1), (1, 0)]], "_": [[(0, 0), (4, 0)]],
    "%": [[(0, 0), (4, 6)], [(0, 6), (0, 5)], [(4, 1), (4, 0)]], "|": [[(2, -1), (2, 7)]],
}


def text_lines(rows, origin, size, color):
    """rows: list of strings, top row first. Returns (verts (2n,3), colors (n,3))."""
    u = size / 6.0
    verts, cols = [], []
    for r, row in enumerate(rows):
        base = origin - up * (r * size * 1.7)
        for c, ch in enumerate(row.upper()):
            o = base + right * (c * 6 * u)
            for stroke in FONT.get(ch, []):
                for (x0, y0), (x1, y1) in zip(stroke[:-1], stroke[1:]):
                    for d in (0.0, 0.25 * u):  # second copy slightly offset = thicker strokes
                        p0 = o + right * (x0 * u + d) + up * (y0 * u)
                        p1 = o + right * (x1 * u + d) + up * (y1 * u)
                        verts += [p0, p1]
                        cols.append(color)
    return np.array(verts, dtype=np.float32).reshape(-1, 3), np.array(cols, dtype=np.float32).reshape(-1, 3)


TEXT_ORIGIN = target + up * 0.13 - right * 0.17
WHITE, YELLOW, RED, GREY = (1, 1, 1), (1, 0.85, 0.2), (1, 0.3, 0.25), (0.7, 0.7, 0.7)
GREEN, CYAN, ORANGE = (0.3, 1, 0.4), (0.45, 0.85, 1), (1, 0.6, 0.15)


def policy_row(e, idx):
    """For switch recordings (record_switch.py): which policy acted at this frame."""
    if "policy" not in e:
        return []
    p = int(e["policy"][min(idx, len(e["policy"]) - 1)])
    name = meta["policies"][p]
    if p == 0:
        return [(f"POLICY: {name}  (SWITCH AT {meta['switch_sec']:.1f} S)", CYAN)]
    return [(f"POLICY: {name}  (SWITCHED AT {meta['switch_sec']:.1f} S)", ORANGE)]


def draw_overlay(header, detail=None):
    gym.clear_lines(viewer)
    v, c = text_lines(header, TEXT_ORIGIN, 0.0085, WHITE)
    allv, allc = [v], [c]
    if detail:
        for k, (row, color) in enumerate(detail):
            v, c = text_lines([row], TEXT_ORIGIN - up * (len(header) + k + 0.5) * 0.0085 * 1.7, 0.0085, color)
            allv.append(v); allc.append(c)
    v = np.concatenate(allv); c = np.concatenate(allc)
    if len(c):
        gym.add_lines(viewer, env_ptr, len(c), v, c)


def reason_rows(e):
    rows = []
    if e.get("completed"):
        rows.append((f"END: COMPLETED {meta['max_episode_length'] * meta['control_dt']:.0f} S (TIME LIMIT)", GREEN))
    elif e["hit_tilt"]:
        rows.append((f"END: TILT {e['tilt_deg']:.0f} DEG > {meta['tilt_limit_deg']:.0f}", RED))
    if not e.get("completed") and e["hit_x"]:
        rows.append((f"END: X {e['x']:.3f} < {meta['x_limit']:.2f}", RED))
    if not e.get("completed") and e["hit_fall"]:
        rows.append((f"END: DIST {e['goal_dist']:.3f} >= {meta['fall_dist']:.2f}", RED))
    if not rows:
        rows.append(("END: UNKNOWN", RED))
    rows.append((f"TILT {e['tilt_deg']:.0f} DEG  X {e['x']:.3f}  DIST {e['goal_dist']:.3f}", GREY))
    ox, oy, oz = e["start_offset_cm"]
    rows.append((f"START YAW {e['start_yaw_deg']:+.0f} DEG  DX {ox:+.1f} DY {oy:+.1f} DZ {oz:+.1f} CM", GREY))
    rows.append(("SPACE: NEXT   R: REPLAY   B: BACK", YELLOW))
    return rows


def set_frame(fr):
    rs = env.root_state_tensor
    rs[obj_i, 0:7] = torch.as_tensor(fr[0:7], device=env.device)
    rs[obj_i, 7:13] = 0
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rs), gymtorch.unwrap_tensor(obj_idx_t), 1)
    ds = env.dof_state
    ds[E * ndof:(E + 1) * ndof, 0] = torch.as_tensor(fr[7:7 + ndof], device=env.device)
    ds[E * ndof:(E + 1) * ndof, 1] = 0
    gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(ds), gymtorch.unwrap_tensor(hand_idx_t), 1)
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(zero_torque))
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)


def poll():
    if gym.query_viewer_has_closed(viewer):
        sys.exit(0)
    acts = []
    for evt in gym.query_viewer_action_events(viewer):
        if evt.value > 0:
            if evt.action == "QUIT":
                sys.exit(0)
            acts.append(evt.action)
    return acts


dt = meta["physics_dt"]
k = max(0, min(args.start - 1, len(episodes) - 1))
while True:
    e = episodes[k]
    fr = e["frames"]
    T = len(fr)
    head_base = f"{TARGET}  EPISODE {k + 1}/{len(episodes)}  ({e['control_steps']} STEPS = {e['control_steps'] * meta['control_dt']:.1f} S)"
    print(f"\n[{k + 1}/{len(episodes)}] env {e['env']} ep {e['ep_in_env']}: {e['control_steps']} control steps, "
          f"tilt {e['tilt_deg']:.1f} deg, x {e['x']:.3f}, dist {e['goal_dist']:.3f} | "
          + ", ".join(r for r, _ in reason_rows(e)[:-3]), flush=True)
    t0 = time.perf_counter()
    action = None
    while True:  # real-time playback
        idx = int((time.perf_counter() - t0) / dt)
        if idx >= T:
            break
        set_frame(fr[idx])
        draw_overlay([head_base, f"T {idx * dt:5.2f} S"], policy_row(e, idx))
        gym.draw_viewer(viewer, sim, True)
        acts = poll()
        if acts:
            action = acts[-1]
            break
    if action is None:  # freeze on the last frame until a key
        set_frame(fr[-1])
        draw_overlay([head_base, f"T {(T - 1) * dt:5.2f} S  (STOPPED)"], policy_row(e, T - 1) + reason_rows(e))
        while action is None:
            gym.draw_viewer(viewer, sim, True)
            acts = poll()
            action = acts[-1] if acts else None
            time.sleep(1 / 60)
    if action == "next":
        if k == len(episodes) - 1:
            print("last episode reached; SPACE replays it, B goes back")
        k = min(k + 1, len(episodes) - 1)
    elif action == "back":
        k = max(k - 1, 0)
    # "replay" keeps k
