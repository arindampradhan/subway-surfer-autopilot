"""Cloudflare Clef Flash on Apple MLX, from the Hugging Face release (Cloudflare/clef-flash).

The release is a Qwen3.5 9B backbone plus a small joint-schema head (`joint_schema_model.py`
in the repo). In bf16 the backbone is 19 GB, too much next to Chrome on a 24 GB Mac, so it
runs on MLX quantized (4-bit by default, ~5 GB; converted once into CACHE). The head runs in
torch on CPU. Prompt building and scoring are the release's own `encode_record` and
`JointSchemaHead`, so only the backbone's arithmetic differs from the reference.

The head also reads lm_head rows for each option's tokens ("lexical" vectors). Those rows are
read in bf16 from the original shards, not from the quantized copy, and cached by token id.
"""

import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

from clef_flash import request

REPO = "Cloudflare/clef-flash"
CACHE = Path(os.path.expanduser("~/.cache/ssbot"))


def convert(src: Path, out: Path, bits: int) -> None:
    from mlx_lm.convert import convert as mlx_convert

    print(f"converting {src.name} to {bits}-bit MLX in {out} (once)", file=sys.stderr)
    mlx_convert(hf_path=str(src), mlx_path=str(out), quantize=True, q_bits=bits, q_group_size=64)


class MlxClef:
    def __init__(self, repo: str = REPO, bits: int = 4):
        from mlx_lm.utils import load_model

        src = Path(snapshot_download(repo))
        sys.path.insert(0, str(src))
        import joint_schema_model as jsm

        self.jsm = jsm
        out = CACHE / f"clef-flash-mlx-{bits}bit"
        if not (out / "config.json").exists():
            convert(src, out, bits)
        model, _ = load_model(out)
        self.backbone = model.language_model.model  # final-norm hidden states
        self.tok = AutoTokenizer.from_pretrained(src)

        self.head = jsm.JointSchemaHead(**json.loads((src / "joint_head_config.json").read_text()))
        from safetensors.torch import load_file

        self.head.load_state_dict(load_file(src / "joint_head.safetensors"), strict=True)
        self.head = self.head.float().eval()

        index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        self.lm_head = safe_open(src / index["lm_head.weight"], framework="pt").get_slice("lm_head.weight")
        self.rows = {}  # token id -> lm_head row (float32)

    def _rows(self, ids: list) -> torch.Tensor:
        for t in ids:
            if t not in self.rows:
                self.rows[t] = self.lm_head[t : t + 1].float()[0]
        return torch.stack([self.rows[t] for t in ids])

    @torch.inference_mode()
    def probabilities(self, premise: str, options: dict) -> dict:
        if len(options) == 1:
            return {k: 1.0 for k in options}
        enc = self.jsm.encode_record(self.tok, request(premise, options))
        hidden = self.backbone(mx.array([list(enc.input_ids)]))
        hidden = torch.from_numpy(np.array(hidden.astype(mx.float32)))
        # The head indexes the output embedding by token id; pass only the rows this input uses.
        ids = torch.tensor([enc.input_ids])
        uniq, local = torch.unique(ids, return_inverse=True)
        logits = self.head(hidden, local, torch.ones_like(ids), [enc], self._rows(uniq.tolist()))[0][0]
        probs = logits.softmax(-1).tolist()
        q = enc.questions[0]
        return {k: round(p, 4) for k, p in sorted(zip(q.option_ids, probs))}
