"""Assemble a labelling set from mined frames (SPEC-roofs-and-powerups §4.1 → §4.2). Offline glue.

Takes the top frames per concept from `frame_search.py search` (data/mined/<concept>/), frames per
power-up from the HUD icon clusters (`hud_icons.py`), and plain ground running frames as a
contrast, and copies them into data/<name>/ with the frame ids `ssbot label` and
`ssbot zone-crops` expect: run index × 10,000,000 + frame id. data/<name>/sources.json maps each
id back to its run, frame and concept.

  ../.venv/bin/python sidecar/mined_set.py --name roofpow_r1
"""

import argparse
import glob
import json
import os
import random
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINED = os.path.join(ROOT, "data", "mined")
# HUD icon clusters (hud_icons.py, 2026-10-15) and the power-up each shows.
CLUSTERS = {"jetpack": [2, 6], "super_sneakers": [0, 3, 5], "magnet": [1, 7, 8, 9], "hoverboard": [4]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="roofpow_r1")
    ap.add_argument("--roof", type=int, default=250)
    ap.add_argument("--ramp", type=int, default=120)
    ap.add_argument("--per-powerup", type=int, default=40)
    ap.add_argument("--ground", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    picks = []  # (run, frame_id, concept)
    seen = set()

    def add(run, frame, concept):
        if (run, frame) not in seen:
            seen.add((run, frame))
            picks.append((run, int(frame), concept))

    manifest = os.path.join(MINED, "manifest.jsonl")
    for row in map(json.loads, open(manifest)) if args.roof + args.ramp and os.path.exists(manifest) else []:
        limit = {"roof": args.roof, "ramp": args.ramp}.get(row["concept"], 0)
        if sum(c == row["concept"] for _, _, c in picks) < limit:
            add(row["run"], row["frame_id"], row["concept"])
    z = np.load(os.path.join(MINED, "hud_icons", "clusters.npz"))
    keys, labels = list(z["keys"]), z["labels"]
    for kind, clusters in CLUSTERS.items() if args.per_powerup else []:
        members = [keys[i] for i in range(len(keys)) if labels[i] in clusters]
        for k in rng.sample(members, min(args.per_powerup, len(members))):
            run, frame = k.split("__")
            add(run, frame, f"powerup:{kind}")
    # Plain running frames that no filter picked: the ground contrast for the level classifier.
    cands = {(c["run"], c["frame_id"]) for c in map(json.loads, open(os.path.join(MINED, "candidates.jsonl")))}
    runs = sorted(glob.glob(os.path.join(ROOT, "runs", "2026*")))
    while sum(c == "ground" for _, _, c in picks) < args.ground:
        run = rng.choice(runs)
        frames = glob.glob(os.path.join(run, "frames", "*.jpg"))
        if not frames:
            continue
        f = rng.choice(frames)
        name = os.path.basename(run)
        frame = int(os.path.basename(f)[:-4])
        if (name, frame) not in cands:
            add(name, frame, "ground")

    out = os.path.join(ROOT, "data", args.name)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    run_index = {r: i + 1 for i, r in enumerate(sorted({r for r, _, _ in picks}))}
    sources = {}
    for run, frame, concept in picks:
        fid = run_index[run] * 10_000_000 + frame
        shutil.copy(os.path.join(ROOT, "runs", run, "frames", f"{frame}.jpg"), os.path.join(out, f"{fid}.jpg"))
        sources[fid] = {"run": run, "frame_id": frame, "concept": concept}
    json.dump(sources, open(os.path.join(out, "sources.json"), "w"), indent=0)
    counts = {}
    for _, _, c in picks:
        counts[c] = counts.get(c, 0) + 1
    print(f"{len(picks)} frames in {out}: {counts}")


if __name__ == "__main__":
    main()
