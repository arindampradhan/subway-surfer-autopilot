"""Merge labelling parts into labels/<set>.jsonl and report the R1 checks
(SPEC-roofs-and-powerups §4.2, §6). Offline glue for labels written by Claude Code subagents.

  ../.venv/bin/python sidecar/label_merge.py powerup_r1

Reads labels/<set>.part_*.jsonl, drops lines that don't validate, writes labels/<set>.jsonl
(first pass first, repeats after), and prints:
  - label-twice agreement on runner_level, active_powerup and sure zone obstacles
  - for mined power-up frames, how often active_powerup matches the HUD cluster it was mined from
"""

import glob
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONES = [f"{lane}-{band}" for band in ("near", "mid", "far") for lane in "LCR"]
OBSTACLES = {"Free", "TrainBody", "TrainRamp", "LowBarrier", "HighBarrier", "OverheadBar", "Unknown", "TrainRoof", "RoofEnd"}
STATES = {"Loading", "AdBreak", "Ad", "Menu", "Running", "Crashed", "RevivePrompt", "NewHighScore", "ScoreScreen", "Paused", "Unknown"}
LEVELS = {"ground", "roof", "air", "unknown"}
ACTIVE = {"none", "jetpack", "super_sneakers", "pogo_stick", "magnet", "multiplier", "hoverboard", "unknown"}
KINDS = {"none", "jetpack", "super_sneakers", "pogo_stick", "magnet", "multiplier", "mystery_box", "unknown"}


def problems(rec):
    l = rec.get("label") or {}
    out = []
    if l.get("game_state") not in STATES:
        out.append("game_state")
    if l.get("player_lane") not in {"L", "C", "R", "unknown"}:
        out.append("player_lane")
    if l.get("runner_level") not in LEVELS:
        out.append("runner_level")
    if l.get("active_powerup") not in ACTIVE:
        out.append("active_powerup")
    zones = l.get("zones") or []
    if sorted(z.get("zone") for z in zones) != sorted(ZONES):
        out.append("zones")
    for z in zones:
        if z.get("obstacle") not in OBSTACLES or z.get("powerup_kind") not in KINDS or not all(isinstance(z.get(k), bool) for k in ("coins", "powerup", "sure")):
            out.append(f"zone {z.get('zone')}")
        elif z["obstacle"] in {"TrainRoof", "RoofEnd"} and l.get("runner_level") != "roof":
            out.append(f"{z['zone']} roof class off a roof")
    return out


def main():
    name = sys.argv[1]
    labels = os.path.join(ROOT, "labels")
    recs, bad = [], []
    for part in sorted(glob.glob(os.path.join(labels, f"{name}.part_*.jsonl"))):
        for i, line in enumerate(open(part)):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                bad.append(f"{os.path.basename(part)}:{i + 1}: {e}")
                continue
            p = problems(r)
            if p:
                bad.append(f"{os.path.basename(part)}:{i + 1} frame {r.get('frame_id')}: {', '.join(p[:4])}")
                continue
            recs.append(r)
    first = {r["frame_id"]: r for r in recs if not r["repeat"]}
    repeats = {r["frame_id"]: r for r in recs if r["repeat"]}
    with open(os.path.join(labels, f"{name}.jsonl"), "w") as f:
        for r in sorted(first.values(), key=lambda r: r["frame_id"]) + sorted(repeats.values(), key=lambda r: r["frame_id"]):
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(first)} frames, {len(repeats)} repeats written; {len(bad)} lines rejected")
    for b in bad[:10]:
        print("  rejected", b)

    both = [(first[i]["label"], repeats[i]["label"]) for i in repeats if i in first]
    if both:
        for field in ("runner_level", "active_powerup", "game_state", "player_lane"):
            agree = sum(a[field] == b[field] for a, b in both)
            print(f"  label-twice {field}: {agree}/{len(both)} = {agree / len(both):.0%}")
        zones = [(za["obstacle"], zb["obstacle"]) for a, b in both for za in a["zones"] for zb in b["zones"] if za["zone"] == zb["zone"] and za["sure"] and zb["sure"]]
        if zones:
            agree = sum(x == y for x, y in zones)
            print(f"  label-twice sure zone obstacles: {agree}/{len(zones)} = {agree / len(zones):.0%}")

    sources_path = os.path.join(ROOT, "data", name, "sources.json")
    if os.path.exists(sources_path):
        sources = json.load(open(sources_path))
        hud = [(s["concept"].split(":")[1], first[int(i)]["label"]["active_powerup"]) for i, s in sources.items() if s["concept"].startswith("powerup:") and int(i) in first]
        if hud:
            agree = sum(a == b for a, b in hud)
            print(f"  active_powerup vs the HUD cluster: {agree}/{len(hud)} = {agree / len(hud):.0%}; misses {Counter((a, b) for a, b in hud if a != b).most_common(5)}")
        concepts = Counter()
        for i, s in sources.items():
            if int(i) in first:
                concepts[(s["concept"], first[int(i)]["label"]["runner_level"])] += 1
        print("  runner_level by mined concept:", dict(concepts))


if __name__ == "__main__":
    main()
