"""Zone classifier: a small CNN on zone crops, replacing the histogram + logistic regression in
perception/model.rs (SPEC §4.3 v2). Offline training only (see CLAUDE.md): inference runs in
Rust (`perception/cnn.rs`) from the JSON this script exports.

The crops come from Rust, so training and live crops match exactly:

  ssbot zone-crops data/live1 data/human_... data/crash_r1 --out data/zone_crops
  ../.venv/bin/python sidecar/zone_cnn.py --holdout crash_r1            # score on a held-out set
  ../.venv/bin/python sidecar/zone_cnn.py --holdout crash_r1 --cv 5     # more domain data, by run
  ../.venv/bin/python sidecar/zone_cnn.py --holdout "" --export zone_model.cnn.json
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
CLASSES = ["Free", "TrainBody", "TrainRamp", "LowBarrier", "HighBarrier", "OverheadBar"]
HAZARD = {"TrainBody", "LowBarrier", "HighBarrier", "OverheadBar"}
SIZE = 32
# Recordings whose frames don't line up with the zones (cyan border: the game didn't fill the
# window) would teach the classifier the wrong crops.
EXCLUDE = {"human_round2"}


def load_crops(path):
    global SIZE
    meta = [json.loads(line) for line in open(os.path.join(path, "meta.jsonl"))]
    info = json.load(open(os.path.join(path, "crops.json"))) if os.path.exists(os.path.join(path, "crops.json")) else {}
    SIZE = info.get("crop", 32)
    load_crops.native = bool(info.get("native", False))
    load_crops.ctx = float(info.get("ctx", 0.0))
    raw = np.fromfile(os.path.join(path, "crops.bin"), dtype=np.uint8)
    crops = raw.reshape(len(meta), SIZE, SIZE, 3)
    return [
        {"crop": crops[i], "y": CLASSES.index(m["class"]), "set": m["set"], "id": m["id"]}
        for i, m in enumerate(meta)
        if m["class"] in CLASSES and m["set"] not in EXCLUDE
    ]


class Net(nn.Module):
    def __init__(self, k=len(CLASSES), c=(3, 24, 48, 64), hidden=96):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(c[i], c[i + 1], 3, padding=1) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(c[i + 1]) for i in range(3)])
        self.fc1 = nn.Linear(c[3] * (SIZE // 8) ** 2, hidden)  # SIZE is set when the crops are loaded
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, k)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.max_pool2d(F.relu(bn(conv(x))), 2)
        return self.fc2(self.drop(F.relu(self.fc1(x.flatten(1)))))


def to_tensor(samples):
    return torch.from_numpy(np.stack([s["crop"] for s in samples])).permute(0, 3, 1, 2).float() / 255.0


def augment(x):
    flip = torch.rand(x.shape[0]) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    x = x * (0.8 + 0.4 * torch.rand(x.shape[0], 1, 1, 1)) + 0.1 * (torch.rand(x.shape[0], 3, 1, 1) - 0.5)
    shift = np.random.randint(-2, 3, size=2)
    return torch.roll(x, shifts=(int(shift[0]), int(shift[1])), dims=(2, 3)).clamp(0, 1)


def train(samples, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = to_tensor(samples)
    y = torch.tensor([s["y"] for s in samples])
    counts = torch.bincount(y, minlength=len(CLASSES)).float()
    weights = (len(y) / (len(CLASSES) * counts.clamp(min=1))).sqrt()  # damped class balancing
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    steps = epochs * ((len(y) + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 128):
            idx = perm[i : i + 128]
            loss = F.cross_entropy(net(augment(x[idx])), y[idx], weight=weights, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def predict(net, samples):
    return net(to_tensor(samples)).argmax(1).tolist()


def report(name, samples, pred):
    truth = [s["y"] for s in samples]
    acc = np.mean([t == p for t, p in zip(truth, pred)])
    haz = [(t, p) for t, p in zip(truth, pred) if CLASSES[t] in HAZARD]
    free = [(t, p) for t, p in zip(truth, pred) if CLASSES[t] == "Free"]
    missed = sum(CLASSES[p] == "Free" for _, p in haz)
    alarms = sum(CLASSES[p] in HAZARD for _, p in free)
    print(f"{name}: {len(samples)} zones, accuracy {acc:.3f}, missed hazards {missed}/{len(haz)}, false alarms {alarms}/{len(free)}")
    conf = Counter((CLASSES[t], CLASSES[p]) for t, p in zip(truth, pred))
    for t in CLASSES:
        row = {CLASSES[p]: conf[(t, CLASSES[p])] for p in range(len(CLASSES)) if conf[(t, CLASSES[p])]}
        if row:
            print(f"   {t:12s} {row}")


def export(net, path, note):
    """JSON for perception/cnn.rs, with each batch norm folded into the conv before it."""
    convs = []
    for conv, bn in zip(net.convs, net.bns):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * scale[:, None, None, None]
        b = (conv.bias - bn.running_mean) * scale + bn.bias
        convs.append({"w": w.flatten().tolist(), "b": b.tolist(), "cin": conv.in_channels, "cout": conv.out_channels})
    dense = lambda fc: {"w": fc.weight.flatten().tolist(), "b": fc.bias.tolist(), "din": fc.in_features, "dout": fc.out_features}
    out = {"kind": "cnn", "classes": CLASSES, "convs": convs, "fc1": dense(net.fc1), "fc2": dense(net.fc2), "note": note,
           "crop": SIZE, "hires": bool(getattr(load_crops, "native", False)), "ctx": float(getattr(load_crops, "ctx", 0.0))}
    json.dump(out, open(path, "w"), separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=os.path.join(ROOT, "data", "zone_crops"))
    ap.add_argument("--holdout", default="crash_r1", help="set used only for scoring ('' trains on everything)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv", type=int, default=0, help="cross-validate the holdout set by run in this many folds, training on the rest of it too")
    ap.add_argument("--export", help="train and write the model JSON for the Rust perceiver")
    args = ap.parse_args()

    samples = load_crops(args.crops)
    hold = [s for s in samples if args.holdout and s["set"] == args.holdout]
    base = [s for s in samples if not (args.holdout and s["set"] == args.holdout)]
    print("sets:", dict(Counter(s["set"] for s in samples)))
    print("classes:", dict(Counter(CLASSES[s["y"]] for s in samples)))

    if args.cv and hold:
        runs = sorted({s["id"] // 10_000_000 for s in hold})
        seen, preds = [], []
        for k in range(args.cv):
            test_runs = {r for i, r in enumerate(runs) if i % args.cv == k}
            tr = base + [s for s in hold if s["id"] // 10_000_000 not in test_runs]
            te = [s for s in hold if s["id"] // 10_000_000 in test_runs]
            net = train(tr, args.epochs, args.seed)
            seen += te
            preds += predict(net, te)
            print(f"fold {k}: trained on {len(tr)} zones, tested on {len(te)}", flush=True)
        report(f"{args.cv}-fold by run on {args.holdout}", seen, preds)
        return

    net = train(base, args.epochs, args.seed)
    report("train", base, predict(net, base))
    if hold:
        report(f"held-out {args.holdout}", hold, predict(net, hold))
    if args.export:
        export(net, args.export, f"CNN on {len(base)} zones; held-out {args.holdout or 'none'}")
        print("wrote", args.export)


if __name__ == "__main__":
    sys.exit(main())
