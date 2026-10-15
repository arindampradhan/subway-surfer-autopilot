"""Cloudflare Clef Flash as the advisor: a decision model that takes a state and typed questions
and returns probabilities, no text. Each decision is one `choice` question whose criteria are
the option sentences, so it scores the same wording OpenJev does. It is zero-shot: there is no
trained head of ours on top.

Two ways to run it:
  local (default)  the Hugging Face release (Cloudflare/clef-flash) on MLX, `mlx_clef.py`
  workers-ai       Cloudflare's hosted model, `ClefFlash` below; needs CLOUDFLARE_ACCOUNT_ID
                   and CLOUDFLARE_API_TOKEN (a token with Workers AI read access)
"""

import os

URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/clef-flash"
INSTRUCTIONS = (
    "The state describes a Subway Surfers runner and the three lanes ahead. "
    "Which move should the runner make now to survive and keep running?"
)


def request(premise: str, options: dict) -> dict:
    """The SystemOne request body both runners answer: one choice question named `move`."""
    return {
        "model": "clef-flash",
        "state": premise,
        "questions": {"move": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": options}},
    }


class ClefFlash:
    """Clef Flash on Workers AI. Every uncached decision is a network round trip."""

    def __init__(self, timeout_s: float = 2.0):
        import httpx

        account, token = os.environ.get("CLOUDFLARE_ACCOUNT_ID"), os.environ.get("CLOUDFLARE_API_TOKEN")
        if not account or not token:
            raise RuntimeError("Clef Flash on Workers AI needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")
        self.url = URL.format(account=account)
        # One kept-alive connection, so only the first decision pays for the TLS handshake.
        self.client = httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=timeout_s)

    def probabilities(self, premise: str, options: dict) -> dict:
        if len(options) == 1:  # a choice question needs at least 2 criteria
            return {k: 1.0 for k in options}
        r = self.client.post(self.url, json=request(premise, options))
        r.raise_for_status()
        payload = r.json()
        # The REST API wraps the model's reply in {"result": ..., "success": ...}.
        probs = payload.get("result", payload)["answers"]["move"]["probabilities"]
        return {k: round(float(probs[k]), 4) for k in sorted(options)}
