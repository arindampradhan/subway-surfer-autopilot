"""Scores a local vision-language model (Ollama) on the labelled frames: can it read the nine
zones? Reports per-class accuracy and seconds per frame, to decide whether it could label data
(a teacher for the fast classifier) or act as a slow second opinion. Offline only (CLAUDE.md).

  ../.venv/bin/python sidecar/vlm_eval.py --model gemma4:latest --set crash_r3 --n 40
"""

import argparse
import base64
import json
import os
import random
import statistics
import sys
import time
import urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONES = [f"{l}-{b}" for l in "LCR" for b in ("near", "mid", "far")]
CLASSES = ["Free", "TrainBody", "TrainRamp", "LowBarrier", "HighBarrier", "OverheadBar"]
HAZARD = {"TrainBody", "LowBarrier", "HighBarrier", "OverheadBar"}

PROMPT = """This is a frame from the game Subway Surfers. The runner is at the bottom, running away from the camera along three parallel tracks. Nine zones are outlined on the track ahead, named by lane (L=left, C=centre, R=right, cyan/magenta/yellow outlines) and distance (near, mid, far). Judge only what is inside each outline.

Classify each zone as exactly one of:
- Free: clear track (coins and scenery don't count)
- TrainBody: the side or front of a train carriage fills the zone
- TrainRamp: a sloped ramp leading up onto a train roof
- LowBarrier: a short striped board or barrier you jump over
- HighBarrier: a tall red/white chevron barrier with two boards
- OverheadBar: a beam raised overhead

Answer with JSON only, one key per zone: {"L-near":"Free","L-mid":"Free","L-far":"Free","C-near":"Free","C-mid":"Free","C-far":"Free","R-near":"Free","R-mid":"Free","R-far":"Free"}"""


def ask(model, image_path, host):
    img = base64.b64encode(open(image_path, "rb").read()).decode()
    body = json.dumps({"model": model, "prompt": PROMPT, "images": [img], "stream": False, "format": "json",
                       "think": False, "options": {"temperature": 0, "num_predict": 200}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", body, {"Content-Type": "application/json"})
    t = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=300))
    return out["response"], time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:latest")
    ap.add_argument("--set", default="crash_r3")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--host", default="http://localhost:11434")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "labels", f"{args.set}.jsonl")) if l.strip()]
    rows = [r for r in rows if r["label"]["game_state"] == "Running"]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    conf, times, bad = Counter(), [], 0
    for i, r in enumerate(rows):
        path = os.path.join(ROOT, "labels", f"{args.set}.preview", f"{r['frame_id']}.jpg")
        if not os.path.exists(path):
            continue
        text, dt = ask(args.model, path, args.host)
        times.append(dt)
        try:
            pred = json.loads(text)
        except json.JSONDecodeError:
            bad += 1
            continue
        for z in r["label"]["zones"]:
            if z["sure"] and z["obstacle"] in CLASSES:
                p = pred.get(z["zone"], "?")
                conf[(z["obstacle"], p if p in CLASSES else "?")] += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)} frames, {statistics.median(times):.1f} s each", flush=True)

    total = sum(conf.values())
    ok = sum(v for (t, p), v in conf.items() if t == p)
    haz = [(t, p) for (t, p), v in conf.items() for _ in range(v) if t in HAZARD]
    free = [(t, p) for (t, p), v in conf.items() for _ in range(v) if t == "Free"]
    print(f"\n{args.model} on {args.set}: {total} zones, accuracy {ok / max(1, total):.3f}, "
          f"missed hazards {sum(p == 'Free' for _, p in haz)}/{len(haz)}, "
          f"false alarms {sum(p in HAZARD for _, p in free)}/{len(free)}, unparseable replies {bad}")
    for t in CLASSES:
        n = sum(v for (tt, _), v in conf.items() if tt == t)
        if n:
            print(f"   {t:12s} n={n:4d} recall {conf[(t, t)] / n:.2f}   " +
                  ", ".join(f"{p} {v}" for (tt, p), v in sorted(conf.items(), key=lambda x: -x[1]) if tt == t and p != t)[:90])
    if times:
        print(f"seconds per frame: median {statistics.median(times):.1f}, p95 {sorted(times)[int(len(times) * .95) - 1]:.1f}")


if __name__ == "__main__":
    sys.exit(main())
