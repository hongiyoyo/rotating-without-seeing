# tools/collect_tactile_data.py
#
# Roll out a trained (frozen) touch-only in-hand-rotation policy and record, per
# episode, the time series of tactile + joint-position readings the policy itself
# perceives, labeled with the (sim-only) ground-truth object identity. The policy
# never observes object identity (see README / task file), so these tactile
# responses are purely shape-driven -- this dataset is meant to train a downstream
# CNN that classifies which of the training objects (object_sets["C"], 9 of
# them) is being manipulated, from touch alone.
#
# Usage (from the repo root, in the project's conda env):
#   python tools/collect_tactile_data.py checkpoint=<path/to/checkpoint.pth>
#
# Useful overrides:
#   task.env.numEnvs=1600                 envs run in parallel (multiple of 16 keeps
#                                          round-robin class balance exact)
#   +target_episodes_per_class=300        stop once every class has this many episodes
#   +max_control_steps=2000               hard cap on total env.step() calls
#   +out_path=tools/data/tactile_dataset.npz
#   +channels=[tactile,joint_pos]         which CHANNEL_SPECS to record, concatenated in order
#   +target_full_per_class=200            also keep going until every class has this many
#                                          episodes that ran the full 50 s (needed for the
#                                          longest classifier window; 0 = off)
#   task.env.objInitPosZHigh=0.001        evaluation placement height (paper setting: 0-1 mm)
#
# Per episode the file also stores whether it ran the full length and which termination
# condition ended it (same three conditions as compute_hand_reward_finger), so the
# classifier can be trained on fixed-length windows from the start of each episode.
#
# (target_episodes_per_class/max_control_steps/out_path/channels are not part of the
# base Hydra config schema, so they need the leading "+" to be added as new keys.)

import isaacgym  # must be imported before torch

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

CHANNEL_SPECS = {
    "tactile": lambda env: env.sensed_contacts,
    "joint_pos": lambda env: env.last_obs_buf[:, 6:22],
}
CHANNEL_WIDTHS = {"tactile": 16, "joint_pos": 16}


def read_frame(env, channels):
    parts = [CHANNEL_SPECS[c](env).detach().cpu().numpy() for c in channels]
    return np.concatenate(parts, axis=1)  # (num_envs, sum(channel widths))


def end_conditions(env):
    """Per-env values of the three termination conditions, read right after env.step()
    (before the next step's reset): tilt of the spin axis (deg), object x, distance to goal."""
    import torch
    from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse
    main = env.spin_axis
    cur = quat_rotate(env.object_rot, quat_rotate_inverse(env.object_init_quat, main))
    tilt = torch.rad2deg(torch.arccos(torch.clamp((main * cur).sum(-1), -1.0, 1.0)))
    gd = torch.norm(env.object_pos - env.goal_pos, dim=-1)
    z_off = env.object_init_pos[:, 2] - env.object_init_state[:, 2]
    return (tilt.cpu().numpy(), env.object_pos[:, 0].cpu().numpy(), gd.cpu().numpy(), z_off.cpu().numpy())


@hydra.main(config_name="config", config_path="../isaacgymenvs/cfg")
def collect(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    import isaacgymenvs

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    if not cfg.checkpoint:
        raise ValueError("Pass a checkpoint to roll out: checkpoint=<path/to/*.pth>")
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # inference only: force test mode, headless, and balanced (round-robin) object assignment
    cfg.test = True
    cfg.headless = True
    cfg.task.env.deterministicObjectAssign = True

    target_episodes_per_class = cfg.get("target_episodes_per_class", 300)
    target_full_per_class = cfg.get("target_full_per_class", 0)
    max_control_steps = cfg.get("max_control_steps", 2000)
    out_path = cfg.get("out_path", "tools/data/tactile_dataset.npz")
    channels = list(cfg.get("channels", ["tactile", "joint_pos"]))
    for c in channels:
        if c not in CHANNEL_SPECS:
            raise ValueError(f"Unknown channel '{c}', choose from {list(CHANNEL_SPECS)}")

    def create_env_thunk(**kwargs):
        return isaacgymenvs.make(
            cfg.seed, cfg.task_name, cfg.task.env.numEnvs, cfg.sim_device, cfg.rl_device,
            cfg.graphics_device_id, cfg.headless, cfg.multi_gpu, cfg.capture_video,
            cfg.force_render, cfg, **kwargs,
        )

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {'vecenv_type': 'RLGPU', 'env_creator': create_env_thunk})

    runner = Runner()
    runner.load(omegaconf_to_dict(cfg.train))
    player = runner.create_player()
    player.restore(cfg.checkpoint)

    env = player.env
    num_envs = env.num_envs
    label_names = list(env.used_training_objects)
    num_classes = len(label_names)
    obj_class = env.object_class_indices.detach().cpu().numpy()  # fixed for the process lifetime
    max_episode_length = env.max_episode_length

    print(f"[collect] {num_envs} envs, {num_classes} classes, channels={channels}, "
          f"target={target_episodes_per_class} episodes/class, max_control_steps={max_control_steps}")

    obses = player.env_reset(env)
    player.get_batch_size(obses, 1)  # sets has_batch_dimension/batch_size (normally done inside player.run())
    per_env_buf = [[] for _ in range(num_envs)]
    episodes = []  # list of (seq: (T, C) float32 array, label: int, meta: dict)
    episode_counts = np.zeros(num_classes, dtype=np.int64)
    full_counts = np.zeros(num_classes, dtype=np.int64)
    tilt_limit_deg = 0.4 * 180.0  # compute_hand_reward_finger: angle_difference > 0.4 * pi

    t0 = time.time()
    step = 0
    for step in range(max_control_steps):
        action = player.get_action(obses, is_deterministic=True)
        obses, rewards, dones, infos = player.env_step(env, action)

        frame = read_frame(env, channels)
        dones_np = dones.detach().cpu().numpy().astype(bool).reshape(-1)
        if dones_np.any():
            tilt, obj_x, goal_dist, z_off = end_conditions(env)
            progress = env.progress_buf.detach().cpu().numpy()

        # Append THIS step's reading first, then finalize on done: the reset for a
        # done env only happens inside the NEXT env.step() call's pre_physics_step,
        # so this step's reading still legitimately belongs to the ending episode.
        for e in range(num_envs):
            per_env_buf[e].append(frame[e])
            if dones_np[e]:
                seq = np.stack(per_env_buf[e], axis=0).astype(np.float32)
                per_env_buf[e] = []
                if progress[e] <= 0:
                    continue  # not a real rollout (reset again before taking a step)
                label = int(obj_class[e])
                full = bool(progress[e] >= max_episode_length - 1)
                meta = dict(full=full,
                            end_tilt=bool(not full and tilt[e] > tilt_limit_deg),
                            end_x=bool(not full and obj_x[e] < 0.59),
                            end_fall=bool(not full and goal_dist[e] >= env.fall_dist),
                            z_off=float(z_off[e]))
                episodes.append((seq, label, meta))
                episode_counts[label] += 1
                full_counts[label] += int(full)

        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[collect] step {step + 1}/{max_control_steps} "
                  f"({(step + 1) / max(elapsed, 1e-6):.2f} steps/s) | "
                  f"episodes/class: {dict(zip(label_names, episode_counts.tolist()))} | "
                  f"full: {full_counts.tolist()}")

        if episode_counts.min() >= target_episodes_per_class and full_counts.min() >= target_full_per_class:
            print(f"[collect] reached target of {target_episodes_per_class} episodes/class "
                  f"(and {target_full_per_class} full-length) at step {step + 1}, stopping.")
            break
    else:
        print(f"[collect] hit max_control_steps={max_control_steps} before every class reached "
              f"target; min count = {episode_counts.min()}.")

    # any envs with an in-flight (non-terminated) episode at the end are dropped:
    # a truncated mid-episode sequence would mislabel a partial rollout as "complete".

    if not episodes:
        raise RuntimeError("No complete episodes were collected -- increase max_control_steps.")

    lengths = np.array([seq.shape[0] for seq, _, _ in episodes], dtype=np.int64)
    if lengths.max() > max_episode_length:
        raise RuntimeError(f"Episode length {lengths.max()} exceeds max_episode_length="
                            f"{max_episode_length} -- padding array would be undersized.")

    num_channels = sum(CHANNEL_WIDTHS[c] for c in channels)
    n = len(episodes)
    padded = np.zeros((n, max_episode_length, num_channels), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    for i, (seq, label, _) in enumerate(episodes):
        padded[i, :seq.shape[0], :] = seq
        labels[i] = label
    metas = [m for _, _, m in episodes]

    out_path = out_path if os.path.isabs(out_path) else to_absolute_path(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        tactile=padded,
        lengths=lengths,
        labels=labels,
        label_names=np.array(label_names),
        channels=np.array(channels),
        full_duration=np.array([m["full"] for m in metas]),
        end_tilt=np.array([m["end_tilt"] for m in metas]),
        end_x=np.array([m["end_x"] for m in metas]),
        end_fall=np.array([m["end_fall"] for m in metas]),
        z_offset=np.array([m["z_off"] for m in metas], dtype=np.float32),
        control_dt=np.float32(env.dt * env.control_freq_inv),
        obj_init_pos_z_high=np.float32(cfg.task.env.get("objInitPosZHigh", 0.01)),
    )

    print(f"\n[collect] saved {n} episodes to {out_path}")
    print(f"[collect] episode length stats (control steps), overall: "
          f"min={lengths.min()} mean={lengths.mean():.1f} median={np.median(lengths):.1f} max={lengths.max()}")
    for c in range(num_classes):
        mask = labels == c
        if not mask.any():
            print(f"  {label_names[c]:12s}: n=0 (!)")
            continue
        l = lengths[mask]
        print(f"  {label_names[c]:12s}: n={mask.sum():4d} min={l.min():3d} mean={l.mean():6.1f} "
              f"median={np.median(l):6.1f} max={l.max():3d} full={full_counts[c]}")


if __name__ == "__main__":
    collect()
