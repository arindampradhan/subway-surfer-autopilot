# Subway Surfers bot

## Main rule: Rust for anything realtime

Prefer Rust for everything that runs while the bot plays: browser control, frame capture, perception, reflexes, the arbiter, input dispatch, the recorder and the benchmark. Keep that path in Rust and off Python, shell and other runtimes.

- **Realtime means anything on the per-frame path.** The reflex path has to stay in the low milliseconds (see SPEC.md §5), so don't add a subprocess, interpreter or network call to it.
- **Training and offline work can use other tools.** `sidecar/` (Python, MLX) for the OpenJev head, labelling scripts, data wrangling and one-off analysis are fine. Rust is still preferred when it's no harder, e.g. the zone classifier trains in Rust.
- **Offline first, then ported.** If a realtime feature needs a model or algorithm prototyped in Python, prototype it there, then port the inference to Rust before it goes into the live loop.
- **The OpenJev sidecar is the one exception on the live path.** It runs out of process and is advisory only. The reflex layer must never wait on it.
- **Scripts are glue.** `scripts/*.sh` may chain Rust commands but shouldn't contain logic the bot needs while playing.
