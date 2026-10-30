# tools/make_finetune_init.py
#
# Turn a trained rl_games checkpoint into a starting point for fine-tuning (e.g. an object-
# specialized policy started from the generalist). rl_games' restore() also brings back the epoch
# counter, frame count and best-reward record, which would make the new run stop almost at once
# (max_epochs already reached) and never save a "best" checkpoint until it beats the old record.
# This keeps the network weights, observation/value normalizers and optimizer state, and resets
# only those three counters.
#
# Usage (any Python with torch):
#   python tools/make_finetune_init.py <in.pth> <out.pth>
# Then train with:  checkpoint=<out.pth>

import sys

import torch


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python tools/make_finetune_init.py <in.pth> <out.pth>")
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu")
    print("source:", src)
    print(f"  epoch {ck.get('epoch')}, frame {ck.get('frame')}, best reward {ck.get('last_mean_rewards')}")
    ck["epoch"] = 0
    ck["frame"] = 0
    ck["last_mean_rewards"] = -100500  # rl_games' "no best yet" value
    torch.save(ck, dst)
    print("saved:", dst, "(epoch, frame and best reward reset; weights, normalizers, optimizer kept)")


if __name__ == "__main__":
    main()
