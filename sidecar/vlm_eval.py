"""Scores a local vision-language model (Ollama) on the labelled frames: can it read the nine
zones? Reports per-class accuracy and seconds per frame, to decide whether it could label data
(a teacher for the fast classifier) or act as a slow second opinion. Offline only (CLAUDE.md).

  ../.venv/bin/python sidecar/vlm_eval.py --model qwen3-vl:8b-instruct --set crash_r3 --n 40
  ../.venv/bin/python sidecar/vlm_eval.py --model qwen3-vl:8b-instruct --set crash_r3 --n 60 --crops
"""

import argparse
import base64
import io
import json
import os
import random
import statistics
import sys
import time
import tomllib
import urllib.request
from collections import Counter

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONES = [f"{l}-{b}" for l in "LCR" for b in ("near", "mid", "far")]
CLASSES = ["Free", "TrainBody", "TrainRamp", "LowBarrier", "HighBarrier", "OverheadBar"]
HAZARD = {"TrainBody", "LowBarrier", "HighBarrier", "OverheadBar"}

CLASS_LIST = """- Free: clear track (coins and scenery don't count)
- TrainBody: the side or front of a train carriage fills the zone
- TrainRamp: a sloped ramp leading up onto a train roof
- LowBarrier: a short striped board or barrier you jump over
- HighBarrier: a tall red/white chevron barrier with two boards
- OverheadBar: a beam raised overhead"""

PROMPT = f"""This is a frame from the game Subway Surfers. The runner is at the bottom, running away from the camera along three parallel tracks. Nine zones are outlined on the track ahead, named by lane (L=left, C=centre, R=right, cyan/magenta/yellow outlines) and distance (near, mid, far). Judge only what is inside each outline.

Classify each zone as exactly one of:
{CLASS_LIST}

Answer with JSON only: one key per zone (L-near, L-mid, L-far, C-near, C-mid, C-far, R-near, R-mid, R-far), each value one of the six class names above."""

CROP_PROMPT = """This is a close-up of one zone of the track in the game Subway Surfers: the {lane} lane, at {band} distance from the runner, outlined in red. The runner runs away from the camera, so obstacles come towards the viewer from the top of the picture. Judge only what is inside the red outline.

Classify the zone as exactly one of:
""" + CLASS_LIST + """

Answer with JSON only, with one key, "obstacle"."""

# Constrains the reply to the six classes. Don't put an example answer in the prompt instead:
# a filled-in example (it used to be all Free) gets copied by small models.
SCHEMA = {"type": "object", "properties": {z: {"type": "string", "enum": CLASSES} for z in ZONES},
          "required": ZONES, "additionalProperties": False}
CROP_SCHEMA = {"type": "object", "properties": {"obstacle": {"type": "string", "enum": CLASSES}},
               "required": ["obstacle"], "additionalProperties": False}
LANE_NAMES = {"L": "left", "C": "centre", "R": "right"}


def ask(model, jpeg, prompt, schema, host):
    body = json.dumps({"model": model, "prompt": prompt, "images": [base64.b64encode(jpeg).decode()],
                       "stream": False, "format": schema, "think": False,
                       "options": {"temperature": 0, "num_predict": 200}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", body, {"Content-Type": "application/json"})
    t = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=300))
    return out["response"], time.time() - t


def zone_crop(frame, quad, width=224):
    """The zone's bounding box plus margin (more above, where obstacles stand up), scaled to
    `width` px with the zone outlined in red. Returns JPEG bytes."""
    w, h = frame.size
    pts = [(x * w, y * h) for x, y in quad]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    mx, my = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    box = (max(0, x0 - mx), max(0, y0 - 3 * my), min(w, x1 + mx), min(h, y1 + my))
    s = width / (box[2] - box[0])
    crop = frame.crop(tuple(round(v) for v in box)).resize(
        (width, round((box[3] - box[1]) * s)), Image.LANCZOS)
    ImageDraw.Draw(crop).polygon([((x - box[0]) * s, (y - box[1]) * s) for x, y in pts], outline=(255, 0, 0), width=3)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--set", default="crash_r3")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--crops", action="store_true",
                    help="ask about one zone at a time, cropped from data/<set>/ with calibration.toml, "
                         "instead of all nine zones on the overlay preview")
    ap.add_argument("--save-crops", help="also write each crop to this directory, to check them by eye")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(ROOT, "labels", f"{args.set}.jsonl")) if l.strip()]
    rows = [r for r in rows if r["label"]["game_state"] == "Running"]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    lanes = tomllib.load(open(os.path.join(ROOT, "calibration.toml"), "rb"))["lanes"]
    conf, times, bad = Counter(), [], 0
    for i, r in enumerate(rows):
        sure = [z for z in r["label"]["zones"] if z["sure"] and z["obstacle"] in CLASSES]
        if args.crops:
            path = os.path.join(ROOT, "data", args.set, r["frame"])
            if not os.path.exists(path):
                continue
            frame, dt, pred = Image.open(path).convert("RGB"), 0.0, {}
            for z in sure:
                lane, band = z["zone"].split("-")
                jpeg = zone_crop(frame, lanes["LCR".index(lane)][band])
                if args.save_crops:
                    os.makedirs(args.save_crops, exist_ok=True)
                    open(os.path.join(args.save_crops, f"{r['frame_id']}_{z['zone']}_{z['obstacle']}.jpg"), "wb").write(jpeg)
                text, t = ask(args.model, jpeg, CROP_PROMPT.format(lane=LANE_NAMES[lane], band=band), CROP_SCHEMA, args.host)
                dt += t
                try:
                    pred[z["zone"]] = json.loads(text)["obstacle"]
                except (json.JSONDecodeError, KeyError):
                    bad += 1
        else:
            path = os.path.join(ROOT, "labels", f"{args.set}.preview", f"{r['frame_id']}.jpg")
            if not os.path.exists(path):
                continue
            text, dt = ask(args.model, open(path, "rb").read(), PROMPT, SCHEMA, args.host)
            try:
                pred = json.loads(text)
            except json.JSONDecodeError:
                bad += 1
                continue
        times.append(dt)
        for z in sure:
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
