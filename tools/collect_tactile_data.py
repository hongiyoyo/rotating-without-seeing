# tools/collect_tactile_data.py
#
# Roll out a trained (frozen) touch-only in-hand-rotation policy and record, per
# episode, the time series of tactile + joint-position readings the policy itself
# perceives, labeled with the (sim-only) ground-truth object identity. The policy
# never observes object identity (see README / task file), so these tactile
# responses are purely shape-driven -- this dataset is meant to train a downstream
# CNN that classifies which of the training objects (object_sets["set16"], 16 of
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
    episodes = []  # list of (seq: (T, C) float32 array, label: int)
    episode_counts = np.zeros(num_classes, dtype=np.int64)

    t0 = time.time()
    step = 0
    for step in range(max_control_steps):
        action = player.get_action(obses, is_deterministic=True)
        obses, rewards, dones, infos = player.env_step(env, action)

        frame = read_frame(env, channels)
        dones_np = dones.detach().cpu().numpy().astype(bool).reshape(-1)

        # Append THIS step's reading first, then finalize on done: the reset for a
        # done env only happens inside the NEXT env.step() call's pre_physics_step,
        # so this step's reading still legitimately belongs to the ending episode.
        for e in range(num_envs):
            per_env_buf[e].append(frame[e])
            if dones_np[e]:
                seq = np.stack(per_env_buf[e], axis=0).astype(np.float32)
                label = int(obj_class[e])
                episodes.append((seq, label))
                episode_counts[label] += 1
                per_env_buf[e] = []

        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[collect] step {step + 1}/{max_control_steps} "
                  f"({(step + 1) / max(elapsed, 1e-6):.2f} steps/s) | "
                  f"episodes/class: {dict(zip(label_names, episode_counts.tolist()))}")

        if episode_counts.min() >= target_episodes_per_class:
            print(f"[collect] reached target of {target_episodes_per_class} episodes/class "
                  f"at step {step + 1}, stopping.")
            break
    else:
        print(f"[collect] hit max_control_steps={max_control_steps} before every class reached "
              f"target; min count = {episode_counts.min()}.")

    # any envs with an in-flight (non-terminated) episode at the end are dropped:
    # a truncated mid-episode sequence would mislabel a partial rollout as "complete".

    if not episodes:
        raise RuntimeError("No complete episodes were collected -- increase max_control_steps.")

    lengths = np.array([seq.shape[0] for seq, _ in episodes], dtype=np.int64)
    if lengths.max() > max_episode_length:
        raise RuntimeError(f"Episode length {lengths.max()} exceeds max_episode_length="
                            f"{max_episode_length} -- padding array would be undersized.")

    num_channels = sum(CHANNEL_WIDTHS[c] for c in channels)
    n = len(episodes)
    padded = np.zeros((n, max_episode_length, num_channels), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    for i, (seq, label) in enumerate(episodes):
        padded[i, :seq.shape[0], :] = seq
        labels[i] = label

    out_path = out_path if os.path.isabs(out_path) else to_absolute_path(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        tactile=padded,
        lengths=lengths,
        labels=labels,
        label_names=np.array(label_names),
        channels=np.array(channels),
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
              f"median={np.median(l):6.1f} max={l.max():3d}")


if __name__ == "__main__":
    collect()
