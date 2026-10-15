"""Level classifier: is the runner on the ground, on a train roof or in the air? A small CNN on the
whole frame (SPEC-roofs-and-powerups §4.3). Offline training only (see CLAUDE.md): inference runs
in Rust (`perception/level.rs`) from the JSON this script exports. Same layer layout and JSON as
the zone CNN (`zone_cnn.py`), with `kind: "level-cnn"`.

  ssbot level-frames data/roofpow_r1 data/... --out data/level_frames
  ../.venv/bin/python sidecar/level_cnn.py --holdout roofpow_r1:0.25              # score
  ../.venv/bin/python sidecar/level_cnn.py --holdout "" --export level_model.json
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["Ground", "Roof", "Air"]


class Net(nn.Module):
    def __init__(self, size, c=(3, 16, 32, 48), hidden=64):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(c[i], c[i + 1], 3, padding=1) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(c[i + 1]) for i in range(3)])
        self.fc1 = nn.Linear(c[3] * (size // 8) ** 2, hidden)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, len(CLASSES))

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.max_pool2d(F.relu(bn(conv(x))), 2)
        return self.fc2(self.drop(F.relu(self.fc1(x.flatten(1)))))


def load(path):
    info = json.load(open(os.path.join(path, "frames.json")))
    meta = [json.loads(line) for line in open(os.path.join(path, "meta.jsonl"))]
    size = info["size"]
    x = np.fromfile(os.path.join(path, "frames.bin"), dtype=np.uint8).reshape(len(meta), size, size, 3)
    return info, [{"x": x[i], "y": CLASSES.index(m["level"]), "set": m["set"], "id": m["id"]} for i, m in enumerate(meta)]


def to_tensor(samples):
    return torch.from_numpy(np.stack([s["x"] for s in samples])).permute(0, 3, 1, 2).float() / 255.0


def train(samples, size, epochs, seed):
    torch.manual_seed(seed)
    x, y = to_tensor(samples), torch.tensor([s["y"] for s in samples])
    counts = torch.bincount(y, minlength=len(CLASSES)).float()
    weights = (len(y) / (len(CLASSES) * counts.clamp(min=1))) ** 0.5
    net = Net(size)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 64):
            idx = perm[i : i + 64]
            xb = x[idx]
            # Mirroring keeps the level; brightness varies across worlds.
            flip = torch.rand(len(idx)) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb) * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, 1))
            loss = F.cross_entropy(net(xb.clamp(0, 1)), y[idx], weight=weights, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


@torch.no_grad()
def predict(net, samples):
    return net(to_tensor(samples)).argmax(1).tolist()


def report(name, samples, pred):
    truth = [s["y"] for s in samples]
    acc = np.mean([t == p for t, p in zip(truth, pred)])
    roof = [(t, p) for t, p in zip(truth, pred) if CLASSES[t] == "Roof"]
    roof_ok = sum(t == p for t, p in roof)
    print(f"{name}: {len(samples)} frames, accuracy {acc:.3f}, roof recall {roof_ok}/{len(roof)} ({100 * roof_ok / max(1, len(roof)):.1f}%)")
    conf = Counter((CLASSES[t], CLASSES[p]) for t, p in zip(truth, pred))
    for t in CLASSES:
        print(f"   {t:7s}", {CLASSES[p]: conf[(t, CLASSES[p])] for p in range(len(CLASSES)) if conf[(t, CLASSES[p])]})
    return acc, roof_ok / max(1, len(roof))


def export(net, path, info, note):
    convs = []
    for conv, bn in zip(net.convs, net.bns):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * scale[:, None, None, None]
        b = (conv.bias - bn.running_mean) * scale + bn.bias
        convs.append({"w": w.flatten().tolist(), "b": b.tolist(), "cin": conv.in_channels, "cout": conv.out_channels})
    dense = lambda fc: {"w": fc.weight.flatten().tolist(), "b": fc.bias.tolist(), "din": fc.in_features, "dout": fc.out_features}
    out = {"kind": "level-cnn", "classes": CLASSES, "convs": convs, "fc1": dense(net.fc1), "fc2": dense(net.fc2), "note": note,
           "crop": info["size"], "hires": bool(info.get("native", False))}
    json.dump(out, open(path, "w"), separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(ROOT, "data", "level_frames"))
    ap.add_argument("--holdout", default="", help="SET:FRACTION — hold out that share of a set's frames, by frame id, for scoring")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", help="write the model JSON for the Rust perceiver")
    args = ap.parse_args()
    info, samples = load(args.frames)
    print("levels:", dict(Counter(CLASSES[s["y"]] for s in samples)), "sets:", dict(Counter(s["set"] for s in samples)))
    hold = []
    if args.holdout:
        name, frac = args.holdout.split(":")
        ids = sorted({s["id"] for s in samples if s["set"] == name})
        # Hold out whole source runs (ids are run index × 10,000,000 + frame), so neighbouring
        # frames of one run never sit on both sides.
        runs = sorted({i // 10_000_000 for i in ids})
        test_runs = set(runs[:: max(1, round(1 / float(frac)))])
        hold = [s for s in samples if s["set"] == name and s["id"] // 10_000_000 in test_runs]
        samples = [s for s in samples if not (s["set"] == name and s["id"] // 10_000_000 in test_runs)]
    net = train(samples, info["size"], args.epochs, args.seed)
    report("train", samples, predict(net, samples))
    result = report(f"held-out {args.holdout}", hold, predict(net, hold)) if hold else None
    if args.export:
        export(net, args.export, info, f"level CNN on {len(samples)} frames; held-out {args.holdout or 'none'}")
        print("wrote", args.export)
    return result


if __name__ == "__main__":
    main()
    sys.exit(0)
