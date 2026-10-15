"""Advisor inference sidecar for ssbot (SPEC §4.6). JSON lines on stdin/stdout.

`--model` picks the advisor:
  0.8b / 2b / 4b   OpenJev, an NLI cross-encoder, on MLX or torch; zero-shot or with a head
  embeddinggemma   EmbeddingGemma 2, a bi-encoder (embeddinggemma.py); needs a trained head
  clef-flash       Cloudflare's Clef Flash decision model, zero-shot: the Hugging Face release on
                   MLX (mlx_clef.py), or Cloudflare's hosted one with --backend workers-ai

Requests (one per line):
  {"type":"decide","id":42,"premise":"Subway run. ...","options":{"left":"Left is the best move because it ...", ...}}
  {"type":"warm","items":[{"premise":"...","options":{...}}]}     optional; no reply
Replies:
  {"type":"ready","model":"OpenJev 0.8B","backend":"mlx"}
  {"type":"decision","id":42,"probs":{"left":0.61,"stay":0.22,"roll":0.17},"ms":231}
  {"type":"error","id":42,"message":"..."}

Each option is an NLI hypothesis scored against the premise; P(entailment) is normalised
across options. The premise is shared, so it is encoded once (shared-prefix scoring).
Decisions are cached by (premise, options); `warm` items are computed while idle, like the
tic-tac-toe bridge's pondering. Logging goes to stderr.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "AlexWortega/openjev"
MODELS = {"4b": "qwen3.5-4b-nli-v5", "2b": "qwen3.5-2b-nli-v5", "0.8b": "qwen3.5-0.8b-nli-v2s-long"}
ADVISORS = [*MODELS, "embeddinggemma", "clef-flash"]


def encoder(model: str, backend: str = "mlx"):
    """The frozen model a decision head is trained on: anything with `latents_hypotheses`."""
    if model == "embeddinggemma":
        from embeddinggemma import EmbeddingGemma

        return EmbeddingGemma()
    if backend == "mlx":
        from mlx_openjev import MlxOpenJev

        return MlxOpenJev(REPO, MODELS[model])
    import torch

    from modeling_openjev import OpenJevCrossEncoder

    device = "mps" if torch.backends.mps.is_available() else None
    return OpenJevCrossEncoder(REPO, subfolder=MODELS[model], device=device)


class Engine:
    def __init__(self, model: str, backend: str, head: str = ""):
        self.head = self.remote = None
        if model == "clef-flash":
            if head:
                print(f"clef-flash is zero-shot; ignoring head {head}", file=sys.stderr)
            if backend == "workers-ai":
                from clef_flash import ClefFlash

                self.name, self.backend, self.remote = "Clef Flash", "workers-ai", ClefFlash()
            else:
                from mlx_clef import MlxClef

                self.name, self.backend, self.remote = "Clef Flash (local, 4-bit)", "mlx", MlxClef()
            return
        if model == "embeddinggemma":
            if not head:
                sys.exit("embeddinggemma needs a trained head: train_head.py --model embeddinggemma")
            backend = "torch"  # sentence-transformers on MPS
        self.name = ("EmbeddingGemma 2" if model == "embeddinggemma" else f"OpenJev {model.upper()}") + (
            " + trained head" if head else ""
        )
        self.backend = backend
        self.ce = encoder(model, backend)
        if head:
            # Decision head trained by train_head.py on the frozen model's latents.
            from modeling_openjev import LatentMLPHead

            self.head = LatentMLPHead.load(head, device="cpu")
            d = self.ce.latents_hypotheses("Subway run.", ["Stay is the best move."]).shape[1]
            if self.head.cfg["d"] != d:
                sys.exit(f"head {head} takes {self.head.cfg['d']}-d latents but {model} gives {d}; train one for {model}")

    def probabilities(self, premise: str, options: dict) -> dict:
        if self.remote is not None:
            return self.remote.probabilities(premise, options)
        keys = sorted(options)
        hyps = [options[k] for k in keys]
        if self.head is not None:
            import numpy as np

            logits = self.head.predict(self.ce.latents_hypotheses(premise, hyps))
            e = np.exp(logits - logits.max())
            return {k: round(float(p), 4) for k, p in zip(keys, e / e.sum())}
        from modeling_openjev import ENT

        ent = self.ce.predict_hypotheses(premise, hyps)[:, ENT]
        total = max(float(ent.sum()), 1e-9)
        return {k: round(float(p) / total, 4) for k, p in zip(keys, ent)}


def reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def read_lines(stream, lines: queue.Queue) -> None:
    for line in stream:
        lines.put(line)
    lines.put(None)


def cache_key(premise: str, options: dict) -> tuple:
    return premise, tuple(sorted(options.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ADVISORS, default="0.8b")
    parser.add_argument("--backend", choices=["mlx", "torch", "workers-ai"], default="mlx", help="OpenJev: mlx or torch; clef-flash: mlx (local) or workers-ai")
    parser.add_argument("--head", default="", help="trained decision head directory (train_head.py)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    engine = Engine(args.model, args.backend, args.head)
    # Warm-up pass so the first real decision doesn't pay for kernel compilation.
    engine.probabilities("Subway run.", {"stay": "Stay is the best move.", "left": "Left is the best move."})
    print(f"loaded {engine.name} ({engine.backend}) in {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    reply({"type": "ready", "model": engine.name, "backend": engine.backend})

    lines = queue.Queue()
    threading.Thread(target=read_lines, args=(sys.stdin, lines), daemon=True).start()
    cache = {}
    warm = []  # (premise, options) still to precompute

    while True:
        # A waiting request always wins; warm one item at a time otherwise.
        if warm and lines.empty():
            premise, options = warm.pop(0)
            key = cache_key(premise, options)
            if key not in cache:
                try:
                    cache[key] = engine.probabilities(premise, options)
                except Exception as exc:
                    print(f"warm failed: {exc!r}", file=sys.stderr)
            continue
        line = lines.get()
        if line is None:
            break
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            reply({"type": "error", "id": None, "message": f"bad JSON: {exc}"})
            continue
        kind = req.get("type")
        if kind == "warm":
            warm.extend((it["premise"], it["options"]) for it in req.get("items", []))
            continue
        if kind != "decide":
            reply({"type": "error", "id": req.get("id"), "message": f"unknown request type {kind!r}"})
            continue
        rid, premise, options = req.get("id"), req.get("premise", ""), req.get("options") or {}
        if len(options) < 1:
            reply({"type": "error", "id": rid, "message": "no options"})
            continue
        try:
            start = time.perf_counter()
            key = cache_key(premise, options)
            if key not in cache:
                cache[key] = engine.probabilities(premise, options)
            ms = (time.perf_counter() - start) * 1000
            reply({"type": "decision", "id": rid, "probs": cache[key], "ms": round(ms, 1)})
        except Exception as exc:  # keep the bot alive on model errors
            print(f"decide failed: {exc!r}", file=sys.stderr)
            reply({"type": "error", "id": rid, "message": str(exc)})


if __name__ == "__main__":
    main()
