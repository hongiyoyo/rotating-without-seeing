# tools/train_tactile_classifier.py
#
# Train a small 1D-CNN to classify which of the training objects (object_sets
# ["C"], 9 of them) is being manipulated, from the tactile (+ optional
# proprioceptive) time series recorded
# by tools/collect_tactile_data.py. Classification only -- no shape reconstruction.
#
# Usage (from the repo root, in the project's conda env):
#   python tools/train_tactile_classifier.py
#   python tools/train_tactile_classifier.py --data tools/data/tactile_dataset.npz --epochs 50
#   python tools/train_tactile_classifier.py --window-sec 5 --out-json results.json --save-model k5.pt
#
# --window-sec K: classify from the first K seconds of each episode only. Episodes shorter than
# K seconds are left out, and every remaining sequence is cut to exactly K seconds, so all inputs
# have the same length and how/when an episode ended cannot leak into the prediction.
# The train/val/test split is made once over ALL episodes (stratified by label, fixed seed) and
# only then filtered by length, so an episode stays in the same split for every K.

import argparse
import copy
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Must match CHANNEL_WIDTHS in collect_tactile_data.py -- duplicated (not imported) so this
# training-only script never has to import isaacgym (which must precede torch and needs a GPU env).
CHANNEL_WIDTHS = {"tactile": 16, "joint_pos": 16}


class TactileDataset(Dataset):
    def __init__(self, seqs, lengths, labels):
        self.seqs = seqs      # (N, T, C) float32
        self.lengths = lengths  # (N,) int64
        self.labels = labels  # (N,) int64

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.lengths[idx], self.labels[idx]


class TactileCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes),
        )

    def forward(self, x, lengths):
        # x: (B, T, C) -> (B, C, T). All conv layers use padding="same" so T is
        # preserved throughout, which is what makes the masked pooling below exact.
        x = x.permute(0, 2, 1)
        feats = self.conv(x)  # (B, 128, T)

        t = feats.shape[-1]
        mask = (torch.arange(t, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).float()  # (B, T)
        mask = mask.unsqueeze(1)  # (B, 1, T)
        # True mean over valid (non-padded) timesteps only -- naive AdaptiveAvgPool1d over the
        # zero-padded tensor would bias against short/early-terminated episodes, since conv bias
        # + BatchNorm can make the padded region's activations nonzero even though the input was 0.
        pooled = (feats * mask).sum(dim=-1) / lengths.unsqueeze(1).float().clamp(min=1)

        return self.head(pooled)


def load_dataset(path):
    d = np.load(path, allow_pickle=False)
    control_dt = float(d["control_dt"]) if "control_dt" in d.files else 0.1
    return d["tactile"], d["lengths"], d["labels"], list(d["label_names"]), list(d["channels"]), control_dt


def select_channels(data, stored_channels, wanted_channels):
    """Slice the (N, T, C) array down to the requested subset of the stored channel groups,
    in stored order (not wanted-argument order, so results stay consistent regardless of how
    --channels was typed)."""
    if wanted_channels is None or list(wanted_channels) == list(stored_channels):
        return data, stored_channels
    for c in wanted_channels:
        if c not in stored_channels:
            raise ValueError(f"Channel '{c}' not present in dataset (has {stored_channels})")
    offsets = {}
    off = 0
    for c in stored_channels:
        offsets[c] = (off, off + CHANNEL_WIDTHS[c])
        off += CHANNEL_WIDTHS[c]
    keep = [c for c in stored_channels if c in wanted_channels]
    slices = [data[:, :, offsets[c][0]:offsets[c][1]] for c in keep]
    return np.concatenate(slices, axis=-1), keep


def stratified_splits(labels, seed):
    idx = np.arange(len(labels))
    train_idx, rest_idx = train_test_split(idx, test_size=0.3, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, stratify=labels[rest_idx], random_state=seed)
    return train_idx, val_idx, test_idx


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for seqs, lengths, labels in loader:
            seqs, lengths, labels = seqs.to(device), lengths.to(device), labels.to(device)
            logits = model(seqs, lengths)
            loss = loss_fn(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_n += labels.size(0)
    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="tools/data/tactile_dataset.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", default=None,
                         help="comma-separated subset of the dataset's stored channels to train on "
                              "(default: use everything in the file), e.g. --channels joint_pos")
    parser.add_argument("--window-sec", type=float, default=None,
                         help="use only the first K seconds of episodes that lasted at least K seconds")
    parser.add_argument("--split-seed", type=int, default=42,
                         help="seed of the train/val/test split (kept separate from --seed so model "
                              "init can vary while the split stays fixed)")
    parser.add_argument("--out-json", default=None, help="write test metrics here")
    parser.add_argument("--save-model", default=None, help="save the best (val) model state_dict here")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tactile, lengths, labels, label_names, channels, control_dt = load_dataset(args.data)
    wanted = args.channels.split(",") if args.channels else None
    tactile, channels = select_channels(tactile, channels, wanted)
    num_classes = len(label_names)
    print(f"[train] loaded {len(labels)} episodes, channels={channels}, classes={label_names}")

    train_idx, val_idx, test_idx = stratified_splits(labels, args.split_seed)
    window_steps = None
    if args.window_sec is not None:
        window_steps = int(round(args.window_sec / control_dt))
        full_len = int(lengths.max())  # a full episode times out at max_episode_length - 1 = 499 steps
        if window_steps > full_len:
            print(f"[train] --window-sec {args.window_sec:g} = {window_steps} steps is longer than a full "
                  f"episode ({full_len} steps); using full-length episodes only")
            window_steps = full_len
        keep = lengths >= window_steps
        train_idx, val_idx, test_idx = (i[keep[i]] for i in (train_idx, val_idx, test_idx))
        tactile = tactile[:, :window_steps]
        lengths = np.minimum(lengths, window_steps)
        print(f"[train] window: first {args.window_sec:g} s = {window_steps} control steps; "
              f"kept {keep.sum()}/{len(keep)} episodes that lasted at least that long")
    print(f"[train] split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    def make_loader(idx, shuffle):
        ds = TactileDataset(
            torch.from_numpy(tactile[idx]), torch.from_numpy(lengths[idx]), torch.from_numpy(labels[idx]))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    model = TactileCNN(in_channels=tactile.shape[-1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"[train] epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, device, optimizer=None)
    print(f"\n[train] best val acc = {best_val_acc:.3f}")
    print(f"[train] test loss {test_loss:.4f} acc {test_acc:.3f} "
          f"(random {num_classes}-way baseline = {1 / num_classes:.3f})")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, lens, labs in test_loader:
            logits = model(seqs.to(device), lens.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labs.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print("\n[train] per-class test accuracy:")
    per_class = {}
    for c, name in enumerate(label_names):
        mask = all_labels == c
        if mask.any():
            acc = (all_preds[mask] == all_labels[mask]).mean()
            per_class[name] = {"acc": float(acc), "n": int(mask.sum())}
            print(f"  {name:12s}: {acc:.3f} (n={mask.sum()})")

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(dict(data=args.data, channels=channels, window_sec=args.window_sec, window_steps=window_steps,
                           seed=args.seed, split_seed=args.split_seed, epochs=args.epochs,
                           n_train=int(len(train_idx)), n_val=int(len(val_idx)), n_test=int(len(test_idx)),
                           best_val_acc=float(best_val_acc), test_acc=float(test_acc),
                           label_names=label_names, per_class=per_class, confusion=cm.tolist()), f, indent=1)
    if args.save_model:
        torch.save(dict(state_dict=model.state_dict(), in_channels=tactile.shape[-1], num_classes=num_classes,
                        channels=channels, label_names=label_names, window_steps=window_steps), args.save_model)
    print("\n[train] confusion matrix (rows=true, cols=predicted):")
    header = " " * 13 + " ".join(f"{n[:8]:>8s}" for n in label_names)
    print(header)
    for i, name in enumerate(label_names):
        row = " ".join(f"{v:8d}" for v in cm[i])
        print(f"  {name:10s} {row}")


if __name__ == "__main__":
    main()
