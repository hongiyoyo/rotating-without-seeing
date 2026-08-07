# tools/evaluate_policy_per_object.py
#
# Clean, dedicated evaluation of a trained checkpoint's per-object rotation performance:
# deterministic rollout, round-robin object assignment for class balance, and harvesting the
# same per-episode {obj, rotations, ep_len, ep_reward} records the training-time wandb logger
# already uses (self._pending_obj_stats, populated in allegro_arm_morb_axis.py's reset_idx).
# This gives a clean snapshot of the FINAL policy's behavior, unlike the training log (which
# reflects the policy as it was evolving, with per-epoch batch noise).
#
# Usage (from the repo root, in the project's conda env):
#   python tools/evaluate_policy_per_object.py checkpoint=<path/to/checkpoint.pth>
#
# Useful overrides:
#   task.env.numEnvs=900              envs run in parallel (multiple of 9 keeps class balance exact)
#   +target_episodes_per_class=300    stop once every class has this many episodes
#   +max_control_steps=3000           hard cap on total env.step() calls
#   +out_path=tools/data/policy_eval.npz

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


@hydra.main(config_name="config", config_path="../isaacgymenvs/cfg")
def evaluate(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    import isaacgymenvs

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    if not cfg.checkpoint:
        raise ValueError("Pass a checkpoint to evaluate: checkpoint=<path/to/*.pth>")
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg.test = True
    cfg.headless = True
    cfg.task.env.deterministicObjectAssign = True

    target_episodes_per_class = cfg.get("target_episodes_per_class", 300)
    max_control_steps = cfg.get("max_control_steps", 3000)
    out_path = cfg.get("out_path", "tools/data/policy_eval.npz")

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
    name_to_idx = {n: i for i, n in enumerate(label_names)}
    max_episode_length = env.max_episode_length

    print(f"[eval] {num_envs} envs, {num_classes} classes, "
          f"target={target_episodes_per_class} episodes/class, max_control_steps={max_control_steps}")

    obses = player.env_reset(env)
    player.get_batch_size(obses, 1)  # sets has_batch_dimension/batch_size (normally done inside player.run())
    # env_reset() does a full reset internally, which pushes one bogus all-zero
    # {rotations:0, ep_len:0, ep_reward:0} record per env into _pending_obj_stats -- discard it.
    if hasattr(env, "_pending_obj_stats"):
        env._pending_obj_stats.clear()

    episodes = []  # list of {"obj": str, "rotations": int, "ep_len": int, "ep_reward": float}
    episode_counts = np.zeros(num_classes, dtype=np.int64)

    t0 = time.time()
    step = 0
    for step in range(max_control_steps):
        action = player.get_action(obses, is_deterministic=True)
        obses, rewards, dones, infos = player.env_step(env, action)

        if hasattr(env, "_pending_obj_stats") and env._pending_obj_stats:
            for stat in env._pending_obj_stats:
                episodes.append(stat)
                episode_counts[name_to_idx[stat["obj"]]] += 1
            env._pending_obj_stats.clear()

        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[eval] step {step + 1}/{max_control_steps} "
                  f"({(step + 1) / max(elapsed, 1e-6):.2f} steps/s) | "
                  f"episodes/class: {dict(zip(label_names, episode_counts.tolist()))}")

        if episode_counts.min() >= target_episodes_per_class:
            print(f"[eval] reached target of {target_episodes_per_class} episodes/class at step {step + 1}, stopping.")
            break
    else:
        print(f"[eval] hit max_control_steps={max_control_steps} before every class reached target; "
              f"min count = {episode_counts.min()}.")

    if not episodes:
        raise RuntimeError("No complete episodes were collected -- increase max_control_steps.")

    labels = np.array([name_to_idx[e["obj"]] for e in episodes], dtype=np.int64)
    rotations = np.array([e["rotations"] for e in episodes], dtype=np.int64)
    ep_len = np.array([e["ep_len"] for e in episodes], dtype=np.int64)
    ep_reward = np.array([e["ep_reward"] for e in episodes], dtype=np.float32)
    # episodes time out at progress_buf >= max_episode_length - 1 (see compute_hand_reward_finger),
    # so max_episode_length itself is never actually reached -- compare against max_episode_length - 1.
    full_duration = (ep_len >= max_episode_length - 1).astype(np.int64)

    out_path = out_path if os.path.isabs(out_path) else to_absolute_path(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        labels=labels, rotations=rotations, ep_len=ep_len, ep_reward=ep_reward,
        full_duration=full_duration, label_names=np.array(label_names),
        max_episode_length=np.array(max_episode_length),
    )

    print(f"\n[eval] saved {len(episodes)} episodes to {out_path}")
    print(f"[eval] {'object':12s} {'n':>5s} {'reward(mean±std)':>20s} {'rotations(mean)':>16s} "
          f"{'ep_len(mean)':>13s} {'full-duration%':>15s}")
    for c, name in enumerate(label_names):
        mask = labels == c
        if not mask.any():
            print(f"  {name:12s}   n=0 (!)")
            continue
        r, rot, el, fd = ep_reward[mask], rotations[mask], ep_len[mask], full_duration[mask]
        print(f"  {name:12s} {mask.sum():5d} {r.mean():9.2f}±{r.std():8.2f} {rot.mean():16.2f} "
              f"{el.mean():13.1f} {fd.mean()*100:14.1f}%")


if __name__ == "__main__":
    evaluate()
