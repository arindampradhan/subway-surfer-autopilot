# Subway Surfers bot

## Layout

A Cargo workspace. Dependencies point one way only: `engine` ← `runtime` ← `lab` ← `cli`.

```
subway_surfers_bot/
├── Cargo.toml              # virtual workspace (members = crates/*, excludes .claude/)
├── crates/
│   ├── engine/             # ssbot-engine: pure, per-frame. No browser, network or labelling deps
│   │   └── src/            #   config, facts, il (inference), sidecar (protocol), perception/, policy/
│   ├── runtime/            # ssbot-runtime: the live loop's I/O around the engine
│   │   └── src/            #   bot, browser, capture/, recorder, sidecar (process)
│   ├── lab/                # ssbot-lab: offline tooling. May use reqwest, image_hasher, etc.
│   │   ├── src/            #   label/, train, fit, il_data, advisor_bench, advisor_data,
│   │   │                   #   bench, replay, see, calibrate, frames
│   │   └── tests/          #   perception + replay tests, fixtures/
│   └── cli/                # ssbot (bin `ssbot`): main.rs, cli.rs (clap), cmd/ handlers
├── sidecar/                # Python (MLX): OpenJev sidecar + training scripts
├── scripts/improve.py      # Typer glue for the improvement loop
├── labels/                 # Claude labels (jsonl) + label_guide.md (labelling prompt)
├── calibration/            # reference frames per screen
├── models/                 # OpenJev heads; archive/ and candidate/ are gitignored
├── data/                   # frames for labelling and training
├── runs/                   # recorded runs and bench reports (gitignored)
├── calibration.toml        # zones, markers, thresholds; points at zone_model.json
├── ssbot.toml              # settings (SPEC §11)
└── zone_model.json         # live zone CNN
```

Build with `cargo build --release` at the root; the binary is `target/release/ssbot`.

## Main rule: Rust for anything realtime

Prefer Rust for everything that runs while the bot plays: browser control, frame capture, perception, reflexes, the arbiter, input dispatch, the recorder and the benchmark. Keep that path in Rust and off Python, shell and other runtimes.

- **Realtime means anything on the per-frame path.** The reflex path has to stay in the low milliseconds (see SPEC.md §5), so don't add a subprocess, interpreter or network call to it.
- **The workspace boundary is the realtime boundary.** The per-frame path lives in `ssbot-engine` + `ssbot-runtime`, which must never depend on `ssbot-lab` (the compiler enforces it).
- **Training and offline work can use other tools.** `sidecar/` (Python, MLX) for the OpenJev head, labelling scripts, data wrangling and one-off analysis are fine. Rust is still preferred when it's no harder, e.g. the zone classifier trains in Rust.
- **Offline first, then ported.** If a realtime feature needs a model or algorithm prototyped in Python, prototype it there, then port the inference to Rust before it goes into the live loop.
- **The OpenJev sidecar is the one exception on the live path.** It runs out of process and is advisory only. The reflex layer must never wait on it.
- **Scripts are glue.** `scripts/*.py` (Typer) may chain `ssbot` commands but hold no logic the bot needs while playing.
