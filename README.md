# Subway Surfers bot

A bot that plays [Subway Surfers on Poki](https://poki.com/en/g/subway-surfers) in a real Chrome window. It reads the screen about 58 times a second, decides with a Rust reflex layer in milliseconds, and takes slower advice from a small language model (OpenJev 0.8B on MLX).

![The bot's best benchmarked run: score 2230, 47 s (sped up 2x)](docs/media/best-run.webp)

*Best run of a 47-run benchmark: **score 2230, 47.0 s**, sped up 2x. Full-quality video: [best-run-original.mp4](docs/media/best-run-original.mp4) (640×358, H.264, 48.5 s). Smaller copy: [best-run.mp4](docs/media/best-run.mp4).*

## How it plays

1. **Browser.** It drives Chrome over CDP ([chromiumoxide](https://crates.io/crates/chromiumoxide)), crops the game iframe and streams frames with the CDP screencast.
2. **Eyes.** A small CNN, written in Rust, classifies 9 zones (3 lanes × near/mid/far) on every frame: free, train, ramp, low barrier or high barrier. It also checks screen markers for menus, the revive prompt and the score screen.
3. **Reflexes.** A rule layer in Rust masks out the actions that would kill the runner and picks a safe one: stay, left, right, jump or roll.
4. **Advisor.** The scene is turned into a short text description. OpenJev, a Qwen3.5 NLI cross-encoder with a decision head trained on it, scores the options in a Python/MLX sidecar process. Its advice is only used when it is fresh and the reflexes agree it is safe. The reflex layer never waits for it.
5. **Training data.** Frames were labelled by Claude, through Claude Code subagents, and checked with a frozen held-out set. On that set the eyes score 95.8 % accuracy, miss 2.4 % of hazards and catch 97.3 % of barriers.

What the eyes see on four frames of the run above (green F = free, red T = train):

![Zone predictions on frames from the run](docs/media/perception.jpg)

## The run in numbers

From `runs/20261006-171732.138/summary.json`, run 16 of the `strict-eyes` benchmark:

| | |
|---|---|
| score | 2230 |
| survival | 47.0 s; it ended on a high barrier in the left lane |
| frames / fps | 3,238 frames at 58.5 fps |
| actions | 142 (181 per minute): 89 reflex, 42 default, 11 advisor |
| reflex latency | p50 19.0 ms, p95 25.5 ms per frame (CNN included) |
| advisor | 70 requests, 69 fresh, 75 % cache hits |

This is the best run, not a typical one. Across the 47 runs of that benchmark, median survival was 9.8 s and the median score was 343.

## Quick start

macOS on Apple Silicon (for MLX), Chrome and a recent Rust toolchain. Everything goes through one CLI, `ssbot` (short for Subway Surfers bot), built from `crates/cli`:

```sh
git clone https://github.com/arindampradhan/subwaysurfbot.git
cd subwaysurfbot
cargo build --release                        # builds target/release/ssbot
./target/release/ssbot run --runs 3          # play 3 runs, recorded under runs/
./target/release/ssbot run --no-advisor      # reflexes only, no Python needed
./target/release/ssbot bench --runs 20 --tag mine   # median survival and score
./target/release/ssbot run --human           # record yourself playing
```

`./target/release/ssbot --help` lists the other commands: labelling, training the zone CNN, replaying runs and benchmarking the advisor.

The advisor needs a Python environment with MLX (and `mlx-lm`). By default `ssbot` looks for it at `../.venv/bin/python`, one level above the repo; change `python` under `[advisor]` in `ssbot.toml` to point elsewhere. `ssbot.toml` holds the settings for the browser, capture, policy, advisor and recorder; SPEC.md §11 describes every field.

## Research

- **[Reflexes First, Language Second](docs/research-paper.md)**: the write-up, in the form of a paper. It covers the architecture, how the labelled data was made, the perception results (Table 4), live benchmarks (Table 6) and the negative results: imitation learning, local VLM labellers and lane-change fixes.
- [SPEC.md](SPEC.md): the original design document, with the latency budgets, milestones and the tic-tac-toe measurements behind the advisor choice (Appendix A).
- [labels/label_guide.md](labels/label_guide.md): the labelling guide the zone labels were made with, defining each obstacle class.
- [docs/media/](docs/media/): the best-run recordings and the perception still used above.

## More

- [CLAUDE.md](CLAUDE.md): the workspace layout (`engine` ← `runtime` ← `lab` ← `cli`) and the rule that anything realtime stays in Rust.

Poki's terms of use forbid automated play. This is a personal research project, not affiliated with SYBO, Kiloo or Poki.
