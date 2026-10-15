"""Search recorded frames by example (SPEC-roofs-and-powerups §4.1, step 2). Offline only.

`ssbot mine-candidates` filters runs down to candidate frames; this embeds them with
EmbeddingGemma 2 and ranks them per concept by their closest seed frame:

  ../.venv/bin/python sidecar/frame_search.py embed                       # ~134 ms per frame on MPS
  ../.venv/bin/python sidecar/frame_search.py search --seeds data/mined/seeds.json --top 300

`seeds.json` maps each concept to a few example frames (paths) and, optionally, text queries:
  {"roof": {"frames": ["runs/.../frames/123.jpg"], "texts": ["runner on top of a train"]}, ...}
Search writes data/mined/<concept>/ (frame copies named <run>__<frame>.jpg) and
data/mined/manifest.jsonl, one line per frame: {run, frame_id, concept, score, filter}.
Embeddings are cached in data/mined/embeddings_t<tokens>.npz, keyed by run and frame id.
"""

import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINED = os.path.join(ROOT, "data", "mined")


def load_model(tokens):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("google/embeddinggemma-2", device=device, model_kwargs={"torch_dtype": torch.float32})
    # 280 image tokens by default; 70 is ~6x faster on whole frames (134 vs 785 ms, measured
    # 2026-10-15) and scene-level concepts survive it. Small icons don't: the HUD kind is read
    # from pixels instead (sidecar/hud_icons.py).
    model[0].processor.image_processor.max_soft_tokens = tokens
    return model


def key(c):
    return f"{c['run']}__{c['frame_id']}"


def cache_path(tokens):
    return os.path.join(MINED, f"embeddings_t{tokens}.npz")


def embed(args):
    from PIL import Image

    cands = [json.loads(line) for line in open(args.candidates)]
    path = cache_path(args.tokens)
    keys, E = [], np.zeros((0, 768), np.float32)
    if os.path.exists(path):
        z = np.load(path)
        keys, E = list(z["keys"]), z["E"]
    have = set(keys)
    todo = [c for c in cands if key(c) not in have]
    print(f"{len(cands)} candidates, {len(todo)} to embed", file=sys.stderr)
    if not todo:
        return
    model = load_model(args.tokens)
    t0 = time.perf_counter()
    for i in range(0, len(todo), 256):
        batch = todo[i : i + 256]
        imgs = [Image.open(os.path.join(ROOT, c["path"])).convert("RGB") for c in batch]
        E = np.concatenate([E, model.encode(imgs, batch_size=16, normalize_embeddings=True).astype(np.float32)])
        keys += [key(c) for c in batch]
        np.savez(path, keys=np.array(keys), E=E)  # resumable
        done = i + len(batch)
        print(f"  {done}/{len(todo)} ({(time.perf_counter() - t0) / done * 1000:.0f} ms each)", file=sys.stderr, flush=True)


def search(args):
    from PIL import Image

    cands = {key(c): c for c in (json.loads(line) for line in open(args.candidates))}
    z = np.load(cache_path(args.tokens))
    keys, E = list(z["keys"]), z["E"]
    seeds = json.load(open(args.seeds))
    model = None
    manifest = open(os.path.join(MINED, "manifest.jsonl"), "w")
    for concept, spec in seeds.items():
        queries = []
        frames = spec.get("frames", [])
        if frames or spec.get("texts"):
            model = model or load_model(args.tokens)
        if frames:
            queries.append(model.encode([Image.open(os.path.join(ROOT, f)).convert("RGB") for f in frames], normalize_embeddings=True))
        if spec.get("texts"):
            queries.append(model.encode(spec["texts"], prompt_name="SearchQuery", normalize_embeddings=True))
        Q = np.concatenate(queries)
        score = (E @ Q.T).max(1)
        # A filter can be required (e.g. HUD concepts only among frames with the timer showing).
        need = spec.get("filter")
        order = [i for i in np.argsort(-score) if need is None or need in cands[keys[i]]["concepts"]]
        seed_names = {os.path.basename(f) for f in frames}
        out_dir = os.path.join(MINED, concept)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        kept = 0
        for i in order:
            c = cands[keys[i]]
            if os.path.basename(c["path"]) in seed_names and any(c["run"] in f for f in frames):
                continue
            shutil.copy(os.path.join(ROOT, c["path"]), os.path.join(out_dir, f"{keys[i]}.jpg"))
            manifest.write(json.dumps({"run": c["run"], "frame_id": c["frame_id"], "concept": concept, "score": round(float(score[i]), 4), "filter": c["concepts"]}) + "\n")
            kept += 1
            if kept >= args.top:
                break
        print(f"{concept}: {kept} frames, scores {score[order[0]]:.3f} .. {score[order[min(kept, len(order)) - 1]]:.3f}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "search"])
    ap.add_argument("--candidates", default=os.path.join(MINED, "candidates.jsonl"))
    ap.add_argument("--tokens", type=int, default=70)
    ap.add_argument("--seeds", default=os.path.join(MINED, "seeds.json"))
    ap.add_argument("--top", type=int, default=300)
    args = ap.parse_args()
    os.chdir(ROOT)
    {"embed": embed, "search": search}[args.cmd](args)


if __name__ == "__main__":
    main()
