"""Imitation learning: a CNN that looks at a frame and predicts the action a person pressed
within the next few hundred ms (stay / left / right / jump / roll). Offline training only (see
CLAUDE.md): the exported JSON runs in Rust (`il.rs`).

  ssbot run --human                                   # play, key presses are logged
  ssbot il-data runs/<human run> --out data/il
  ../.venv/bin/python sidecar/il_cnn.py --export models/il.json
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIONS = ["stay", "left", "right", "jump", "roll"]
W, H = 128, 72
SWAP = {1: 2, 2: 1}  # mirroring the picture swaps left and right


def load(path):
    meta = [json.loads(line) for line in open(os.path.join(path, "meta.jsonl"))]
    x = np.fromfile(os.path.join(path, "frames.bin"), dtype=np.uint8).reshape(len(meta), H, W, 3)
    return x, np.array([m["label"] for m in meta]), np.array([m["run"] for m in meta]), np.array([m["t"] for m in meta])


class Net(nn.Module):
    # (in, out, kernel, stride)
    SPEC = [(3, 16, 5, 2), (16, 32, 3, 2), (32, 48, 3, 2), (48, 64, 3, 2)]

    def __init__(self, hidden=96):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(i, o, k, s, k // 2) for i, o, k, s in self.SPEC])
        self.bns = nn.ModuleList([nn.BatchNorm2d(o) for _, o, _, _ in self.SPEC])
        self.fc1 = nn.Linear(64 * 5 * 8, hidden)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, len(ACTIONS))

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        return self.fc2(self.drop(F.relu(self.fc1(x.flatten(1)))))


def tensor(x):
    return torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0


def augment(x, y):
    flip = torch.rand(x.shape[0]) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    swapped = y.clone().apply_(lambda v: SWAP.get(v, v))
    y = torch.where(flip, swapped, y)
    x = x * (0.8 + 0.4 * torch.rand(x.shape[0], 1, 1, 1)) + 0.08 * (torch.rand(x.shape[0], 3, 1, 1) - 0.5)
    return x.clamp(0, 1), y


def train(x, y, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xt, yt = tensor(x), torch.from_numpy(y).long()
    counts = torch.bincount(yt, minlength=len(ACTIONS)).float()
    weights = (len(yt) / (len(ACTIONS) * counts.clamp(min=1))).sqrt()
    net = Net().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    steps = epochs * ((len(yt) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 64):
            idx = perm[i : i + 64]
            xb, yb = augment(xt[idx], yt[idx].clone())
            loss = F.cross_entropy(net(xb.to(DEVICE)), yb.to(DEVICE), weight=weights.to(DEVICE), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval().cpu()


@torch.no_grad()
def predict(net, x):
    out = []
    for i in range(0, len(x), 256):
        out.append(F.softmax(net(tensor(x[i : i + 256])), 1))
    return torch.cat(out).numpy()


def report(name, y, prob):
    pred = prob.argmax(1)
    print(f"{name}: {len(y)} frames, accuracy {np.mean(pred == y):.3f}, always-stay {np.mean(y == 0):.3f}")
    conf = Counter(zip(y.tolist(), pred.tolist()))
    for t, a in enumerate(ACTIONS):
        n = sum(conf[(t, p)] for p in range(5))
        if n:
            row = {ACTIONS[p]: conf[(t, p)] for p in range(5) if conf[(t, p)]}
            print(f"   {a:6s} n={n:5d} recall {conf[(t, t)] / n:.2f}  predicted as {row}")


def export(net, path, note):
    """JSON for il.rs, with each batch norm folded into the conv before it."""
    layers = []
    for conv, bn in zip(net.convs, net.bns):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * scale[:, None, None, None]
        b = (conv.bias - bn.running_mean) * scale + bn.bias
        layers.append({"w": w.flatten().tolist(), "b": b.tolist(), "cin": conv.in_channels, "cout": conv.out_channels,
                       "k": conv.kernel_size[0], "stride": conv.stride[0], "pad": conv.padding[0]})
    dense = lambda fc: {"w": fc.weight.flatten().tolist(), "b": fc.bias.tolist(), "din": fc.in_features, "dout": fc.out_features}
    json.dump({"kind": "il", "actions": ACTIONS, "width": W, "height": H, "convs": layers, "fc1": dense(net.fc1),
               "fc2": dense(net.fc2), "note": note}, open(path, "w"), separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "il"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block_s", type=float, default=6.0, help="seconds per validation block")
    ap.add_argument("--export", help="train on everything and write the model JSON for Rust")
    args = ap.parse_args()
    x, y, run, t = load(args.data)
    print("frames:", len(y), dict(zip(ACTIONS, np.bincount(y, minlength=5).tolist())))
    block = (t // (args.block_s * 1000)).astype(int) + run * 100000
    val = (block % 5) == 4  # every fifth block held out, so neighbouring frames don't leak
    net = train(x[~val], y[~val], args.epochs, args.seed)
    report("train", y[~val], predict(net, x[~val]))
    if val.any():
        prob = predict(net, x[val])
        report("held-out", y[val], prob)
        # An action is only worth taking if it is right when the model commits to it.
        yv = y[val]
        for thr in (0.5, 0.7, 0.85, 0.95):
            row = []
            for a in range(1, 5):
                fire = prob[:, a] >= thr
                row.append(f"{ACTIONS[a]} fires {fire.sum():4d} precision {np.mean(yv[fire] == a) if fire.any() else 0:.2f}")
            print(f"   at p>={thr}: " + " | ".join(row))
    if args.export:
        net = train(x, y, args.epochs, args.seed)
        export(net, args.export, f"IL CNN on {len(y)} frames")
        print("wrote", args.export)


if __name__ == "__main__":
    sys.exit(main())
