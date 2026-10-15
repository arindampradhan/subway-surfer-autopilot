"""Group the power-up icons in the bottom-left HUD so each group can be named once (SPEC-roofs-and-powerups
§4.3). Offline only. The icon is too small for EmbeddingGemma at 70 image tokens, so this clusters
the icon pixels directly.

  ../.venv/bin/python sidecar/hud_icons.py cluster --k 10      # writes data/mined/hud_icons/cluster_<i>.png
  ../.venv/bin/python sidecar/hud_icons.py export --names '{"0":"Magnet","3":"Jetpack"}'

`export` copies one reference frame per named cluster to calibration/hud_<kind>.jpg and prints
the `[[hud.icons]]` entries for calibration.toml (max_diff from the cluster's own spread).
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "mined", "hud_icons")
ICON = (0.012, 0.92, 0.036, 0.062)  # HudConfig::default().icon
SIDE = 16


def icon(path, grey=False):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    box = (round(ICON[0] * w), round(ICON[1] * h), round((ICON[0] + ICON[2]) * w), round((ICON[1] + ICON[3]) * h))
    crop = img.crop(box)
    return crop.convert("L") if grey else crop


def load():
    cands = [json.loads(line) for line in open(os.path.join(ROOT, "data", "mined", "candidates.jsonl"))]
    hud = [c for c in cands if "hud" in c["concepts"]]
    X = np.stack([np.asarray(icon(os.path.join(ROOT, c["path"])).resize((SIDE, SIDE)), np.float32).ravel() / 255 for c in hud])
    return hud, X


def cluster(args):
    from sklearn.cluster import KMeans

    hud, X = load()
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(X)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "clusters.npz"), labels=km.labels_, keys=np.array([f"{c['run']}__{c['frame_id']}" for c in hud]))
    for k in range(args.k):
        idx = np.where(km.labels_ == k)[0]
        # Closest members first: the cleanest examples of the cluster.
        d = np.linalg.norm(X[idx] - km.cluster_centers_[k], axis=1)
        idx = idx[np.argsort(d)]
        picks = list(idx[:12]) + list(idx[len(idx) // 2 : len(idx) // 2 + 6]) + list(idx[-6:])
        sheet = Image.new("RGB", (8 * 72, 3 * 92), "white")
        draw = ImageDraw.Draw(sheet)
        for j, i in enumerate(picks[:24]):
            x, y = (j % 8) * 72, (j // 8) * 92
            sheet.paste(icon(os.path.join(ROOT, hud[i]["path"])).resize((64, 64), Image.NEAREST), (x + 4, y + 4))
            draw.text((x + 4, y + 72), "near" if j < 12 else ("mid" if j < 18 else "far"), fill="black")
        sheet.save(os.path.join(OUT, f"cluster_{k}.png"))
        print(f"cluster {k}: {len(idx)} frames, e.g. {hud[idx[0]]['path']}")


def export(args):
    names = json.loads(args.names)
    hud, X = load()
    z = np.load(os.path.join(OUT, "clusters.npz"))
    labels = z["labels"]
    entries = []
    for k, kind in names.items():
        idx = np.where(labels == int(k))[0]
        centre = X[idx].mean(0)
        best = idx[np.argmin(np.linalg.norm(X[idx] - centre, axis=1))]
        ref = os.path.join("calibration", f"hud_{kind.lower()}_{k}.jpg")
        shutil.copy(os.path.join(ROOT, hud[best]["path"]), os.path.join(ROOT, ref))
        # Grey mean-absolute difference of each member to the reference icon, as Hud::detect computes it.
        r = np.asarray(icon(os.path.join(ROOT, ref), grey=True), np.float32)
        diffs = [np.abs(np.asarray(icon(os.path.join(ROOT, hud[i]["path"]), grey=True), np.float32) - r).mean() for i in idx[:400]]
        entries.append((kind, ref, float(np.percentile(diffs, 95)), len(idx)))
    for kind, ref, p95, n in entries:
        print(f'[[hud.icons]]\npowerup = "{kind}"\nreference = "{ref}"\nmax_diff = {max(15.0, round(p95 * 1.2, 1))}  # {n} frames; 95th pct diff {p95:.1f}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cluster", "export"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--names", default="{}")
    args = ap.parse_args()
    {"cluster": cluster, "export": export}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
