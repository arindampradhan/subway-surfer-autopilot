# Subway Surfers bot: spec

Status: code implemented (all modules in §6, CLI in §7); M0 partly answered (§10.2); M1–M5 need live
runs, labels and the terms check, see [§12](#12-implementation-status). Owner: apradhan. Date: 2026-10-05.

A Rust program that opens [Subway Surfers on Poki](https://poki.com/en/g/subway-surfers) in Chrome,
watches the game frame by frame, and plays it. Fast Rust code does perception and reflexes, and the
OpenJev decision model picks moves when there's time. It builds on the tic-tac-toe experiment in
`../laya_tictactoe` (see [Appendix A](#appendix-a-what-the-tic-tac-toe-experiment-established)).

---

## 1. Goals and non-goals

**Goals**
1. Play a full run end to end with no human input: open page → start run → dodge → detect crash → restart.
2. Keep orchestration, browser control, capture, perception, reflexes and input **in Rust**.
3. Use OpenJev for move choice **where its latency allows**, and report how much it helps compared with rules alone.
4. Record every run (frames, facts, actions, timings) so perception and policy can be tuned offline.

**Non-goals**
- Topping leaderboards or submitting scores anywhere. This is a local, personal experiment. The page
  shows a "Top Run" leaderboard; the bot must not be used to post to it (see [§10](#10-risks-and-open-questions)).
- Reading raw pixels with a language model **while playing**. Earlier tests showed small models fail
  at spatial reasoning from grids. During play, perception is done in Rust and handed to OpenJev as
  **text facts**. Claude's vision is used **offline only**, to label recorded frames that train and
  check that Rust perception ([§4.8](#48-labelling-with-claude-offline)). Claude is never called during a run.
- Pure-Rust model inference in v1 (see [§4.6](#46-inference-sidecar)).
- Other games, mobile, or headless/cloud runs.

## 2. Is it feasible?

Subway Surfers is a **real-time** endless runner. Speed and obstacle density rise over a run.

| Constraint | Value | Source |
|---|---|---|
| Rendering | WebGL canvas (Unity port); **no page elements to read game state from** | to verify, Phase 0 |
| Controls | ← → change lane · ↑ jump · ↓ roll · Space hoverboard | Poki game page |
| Lanes | 3, viewed in perspective from behind the runner | game |
| OpenJev 0.8B decision (MLX, ~6 options, shared-prefix scoring) | **~250 ms** | measured, Appendix A |
| OpenJev 4B decision | ~880 ms | measured |
| Time from an obstacle becoming visible to impact | ~0.6–1.0 s early in a run, less later | **estimate, verify in Phase 0** |

**Consequence:** the model is too slow to be the only thing between the runner and a train. The
design uses **two layers** ([§4.5](#45-policy-two-layers)):
- **Reflex layer (Rust rules, <5 ms):** always on, and the only thing allowed to make emergency dodges.
- **Advisor layer (OpenJev, ~250 ms, async):** picks the *preferred* lane and action for the next
  stretch of track (coins, power-ups, avoiding dead ends). The reflex layer vetoes any advice that
  would cause a crash.

Expected outcome, stated honestly: rules alone should survive the early game. The model's
contribution will be modest and has to be measured ([§8](#8-milestones), M5). If it adds nothing
measurable, that is a valid result.

## 3. Architecture

```
┌────────────────────────── Rust: ssbot (one process, tokio) ──────────────────────────┐
│                                                                                      │
│  browser ──CDP──▶ Chrome (headful, fixed 1280×720 viewport, persistent profile)       │
│     ▲                    │                                                           │
│     │ key events         │ screencast frames (JPEG) or native capture                │
│     │                    ▼                                                           │
│  actuator ◀── arbiter ◀── policy::reflex ◀── facts ◀── perception ◀── capture        │
│                  ▲                                         │                         │
│                  └──── policy::advisor ◀── (facts text) ───┘                         │
│                              │  ▲                                                    │
│                              ▼  │ JSON lines (stdin/stdout)                          │
│                       inference sidecar                                              │
│                                                                                      │
│  recorder: frames + facts + actions + timings → runs/<timestamp>/                    │
└──────────────────────────────────────────────────────────────────────────────────────┘
                               │
                 Python sidecar: openjev_sidecar.py (MLX, OpenJev 0.8B by default)
```

Data flow per frame: **capture → perception → facts → reflex decision → (maybe) advisor result →
arbitrate → key press**, with the advisor running alongside rather than blocking.

## 4. Components

### 4.1 Browser control (`browser` module)
- **Crate:** `chromiumoxide 0.9` (CDP client; the Rust counterpart to Python's browser-use). Fallback: `headless_chrome 1.0`.
- **Chrome:** launch **headful**, because some games pause or throttle when hidden. Use a fixed window and viewport
  (1280×720, `deviceScaleFactor: 1`) so screen geometry stays stable, and a persistent `--user-data-dir`
  so the cookie consent is remembered.
- **Flags:** `--disable-background-timer-throttling --disable-renderer-backgrounding --autoplay-policy=no-user-gesture-required`.
- **Startup sequence:**
  1. Open the game URL and accept the cookie dialog if it appears.
  2. Wait for the game iframe. Record its origin and frame ID; the origin is **unknown, discover in Phase 0**.
  3. Wait out any pre-roll ad by watching for game-state `Menu` ([§4.3](#43-perception-perception-module)), not a fixed sleep.
  4. Click the canvas centre to give the iframe keyboard focus.
  5. Optional: enter page fullscreen or zoom so the canvas fills the viewport. Recalibrate if the size changes.
- **Inputs:** `Input.dispatchKeyEvent` with `keyDown` and `keyUp` for ArrowLeft, ArrowRight, ArrowUp, ArrowDown and Space.
  **Risk:** events must reach a cross-origin iframe. If page-level dispatch doesn't arrive, attach to the
  iframe's target (site isolation) and dispatch there. Phase 0 checks both.

### 4.2 Frame capture (`capture` module)
Two backends behind one trait, `trait FrameSource { async fn next(&mut self) -> Frame }`, where
`Frame { id: u64, t_capture: Instant, rgb: RgbImage }`:

| Backend | How | Expected | Notes |
|---|---|---|---|
| **A: CDP screencast** (default) | `Page.startScreencast {format: jpeg, quality: 60, maxWidth: 640}`, ack each frame | ~15–30 fps, ~20–40 ms delay | No extra permissions; measure in Phase 0 |
| **B: Native** | `xcap 0.9` window capture (ScreenCaptureKit on macOS) | 60 fps possible | Needs macOS Screen Recording permission; crop to canvas rect |

- Decode JPEG with `image 0.25`, then downscale to the working size (~320×180) with `fast_image_resize 6`.
- Always process **the newest frame** and drop stale ones. Never queue.

### 4.3 Perception (`perception` module)
Turns a frame into a typed `Observation`. Pure Rust, no ML in v1, **≤10 ms per frame**.

**Calibration (one-off, saved to `calibration.toml`):** with the run paused or at the start, record:
- **Lane look-ahead zones:** for each lane, three trapezoids along the track at `near`, `mid` and `far` distance.
- **Player zone:** the strip where the runner's sprite is, used to work out the current lane.
- **Screen markers:** for game-state detection (menu, "tap to play", crash/revive dialog, pause, ad).
  Store each as a small template plus position.

**Per-frame outputs:**
```rust
enum GameState { Loading, AdBreak, Ad, Menu, Running, Crashed, RevivePrompt, NewHighScore, ScoreScreen, Paused, Unknown }
enum Obstacle  { Free, TrainBody, TrainRamp, LowBarrier, HighBarrier, OverheadBar, Unknown }
struct LaneView { near: Obstacle, mid: Obstacle, far: Obstacle, coins: bool, powerup: bool }
struct Observation {
    frame_id: u64, state: GameState, player_lane: Option<Lane /* L|C|R */>,
    airborne: bool, rolling: bool, lanes: [LaneView; 3], confidence: f32,
}
```

**Classifier v1:** heuristics per zone:
- Colour histograms, mainly the hue and saturation of trains, barriers and coins.
- Edge density.
- Frame-to-frame change.
- Template matching for screen markers.
- Every threshold lives in `calibration.toml`.

**v2, only if v1 is under 95% accurate on the labelled set:** a small object detector exported to
ONNX and run with `ort 2.0` (CoreML execution provider), trained on frames saved by the recorder.

**Ground truth:** labels come from Claude, not from hand-labelling ([§4.8](#48-labelling-with-claude-offline)).
Targets: ≥300 zone labels per obstacle class and ≥50 per game state, with a small human spot-check.
Crash frames add labels automatically from what actually happened. Perception must reach the targets
in [§9](#9-testing) before moving past M2.

**Fitting the classifier to labels:**
- **v1 heuristics:** grid-search the thresholds in `calibration.toml` to maximise agreement with
  Claude's labels on the training split.
- **v2, if heuristics fall short:** a tiny CNN, about 32×32 input per zone crop and 7 classes.
  Train it in Rust with `candle-nn`, which handles ordinary conv nets; the Qwen3.5 gap noted in
  [§4.6](#46-inference-sidecar) doesn't apply. Run it with Candle, or export to ONNX for `ort`.

### 4.4 Facts (`facts` module)
Turns an `Observation` into short text, Hard-mode style: **plain facts only, never which move is best**.
Example:
```
Subway run. You are in the center lane, on the ground.
Left lane: train close ahead. Center lane: low barrier at mid distance. Right lane: free, coins ahead.
```
- Facts use a **closed vocabulary**, so the same situation always produces the same string. This
  makes the advisor cache ([§4.5](#45-policy-two-layers)) effective.
- Size: ≤60 tokens, keeping one OpenJev pass short.

### 4.5 Policy: two layers
**Actions:** `Stay, Left, Right, Jump, Roll, Hoverboard`. Only actions that are possible are offered;
for example, no `Left` from the left lane.

**Reflex layer (`policy::reflex`, Rust, per frame):**
- Works out time to impact for the current lane from obstacle distance and the run's current speed,
  estimated from how fast zone contents move.
- Must-act rules, in priority order:
  1. A train body ahead in the lane → switch to the nearest lane that is free, or has only a ramp, at `near` and `mid`.
  2. A low barrier → jump or roll.
  3. A high barrier or overhead bar → roll.
- Produces a **safety mask**: the set of actions that won't crash within the look-ahead window.
  Every action, including the advisor's picks, must pass it.
- Acts only within its emergency window, a configurable time to impact (default 250 ms). Outside it,
  the advisor's choice wins.

**Advisor layer (`policy::advisor`, OpenJev, async):**
- Whenever the facts string changes and no request is in flight, send
  `{facts, options}` to the sidecar, tagged with `frame_id`.
- **Option wording**, following Appendix A: *"&lt;Action&gt; is the best move because it &lt;effect&gt;."*,
  where the effect is a computed fact such as "moves into a free lane with coins ahead" or "keeps you
  in a lane with a low barrier at mid distance".
- **Results go stale:** discard any answer whose `frame_id` is older than `advisor_max_age_ms`
  (default 400 ms), or whose facts no longer match the current facts.
- **Cache:** `HashMap<facts_string, Decision>`. Discrete facts repeat often, so repeated states cost 0 ms.
  This is the same idea as tic-tac-toe pondering.
- **Pre-warm (optional):** at startup, run the advisor over the most frequent facts strings from earlier recordings.

**Arbiter:**
```
if state != Running        → handle game-state flow (start, restart, dismiss revive), no lane actions
mask = reflex.safety_mask(obs)
if reflex.emergency(obs)   → reflex.action (always in mask)
else if advisor fresh && advisor.action ∈ mask → advisor.action
else                       → reflex.default (best safe action by rule: prefer coins, stay centered)
```
- **Cooldowns:** after a lane change, jump or roll, block a new action for that move's animation time
  (calibrated in Phase 0, starting value 180 ms) so the game doesn't drop key presses.

### 4.6 Inference sidecar
- **v1:** Python + MLX, a trimmed version of `../laya_tictactoe/{bridge.py,mlx_openjev.py}`, run as
  a child process by Rust. Measured cost is ~250 ms per decision on the 0.8B and ~1 s to load.
- **Why not pure Rust yet:**
  - **Candle:** has no Qwen3.5 support, i.e. no Gated DeltaNet layer.
  - **`mlx-rs 0.32`:** has the MLX primitives but not the Qwen3.5 model code. Porting `mlx_lm/models/qwen3_5.py`
    plus the OpenJev score layer is about 600 lines of work, and gives the same speed as the Python
    sidecar. The bottleneck is the model's first pass over the input (about 1.8 ms per token), not Python.
- **v2 (optional, M6):** a Rust port with `mlx-rs` that loads the same safetensors. Ship it only if it
  makes the same choices as the sidecar on ≥99% of a recorded decision set.

**Protocol (JSON lines, one request per line, one reply per request):**
```jsonc
// Rust → sidecar
{"type":"decide","id":42,"premise":"Subway run. You are in ...","options":{"left":"Left is the best move because it ...", "stay":"..."}}
{"type":"warm","items":[{"premise":"...","options":{...}}]}   // optional, no reply
// sidecar → Rust
{"type":"ready","model":"OpenJev 0.8B","backend":"mlx"}
{"type":"decision","id":42,"probs":{"left":0.61,"stay":0.22,"roll":0.17},"ms":231}
{"type":"error","id":42,"message":"..."}
```
- **Startup:** sidecar flags are `--model 0.8b|2b|4b` (default 0.8b) and `--backend mlx|torch` (default mlx).
- **Errors:** if the sidecar crashes, Rust logs it and carries on with **reflex-only** play.

### 4.7 Recorder and replay (`recorder`, `tools/replay`)
- **Per run,** writes to `runs/<ts>/`:
  - `frames/<id>.jpg`, sampled at a configurable rate and always including the 2 s before a crash
  - `events.jsonl`, one line per frame: `{frame_id, t, obs, facts, reflex, advisor, chosen, latency_ms}`
  - `summary.json` (see the metrics in [§9](#9-testing))
- **Replay:** `ssbot replay runs/<ts>` re-runs perception and policy on saved frames with no browser,
  for tuning thresholds and checking for regressions.

### 4.8 Labelling with Claude (offline)
The user records gameplay but doesn't label it. Claude does the labelling instead, in two roles:

| Role | Who | Volume | When |
|---|---|---|---|
| **Calibration** | Claude Code, interactively in this repo, reading frames as images | ~10–20 frames | Once, then again whenever the game's look changes |
| **Bulk labelling** | Claude API (vision + structured JSON), called from Rust via the Batches API | hundreds to a few thousand frames | After each recording session |

#### Inputs
- **Before M1:** a QuickTime screen recording (`.mov`) of a human playing.
  - `ssbot frames video.mov` calls `ffmpeg` to pull frames at 10 fps and crops them to the canvas.
  - It doesn't capture key presses, which is fine for labelling.
- **From M1 on:** `runs/<ts>/frames` from the recorder, with exact key timestamps.
- **Frame selection, to keep cost down:**
  - Skip near-duplicates: drop a frame if its 64-bit perceptual hash is within 6 bits of the last kept frame.
  - Always keep ~1 s before every crash, plus every frame where the game state changes.
  - Then sample the rest evenly, up to a per-session cap (default 600).

#### Step 1: calibration with Claude Code
1. The user runs `ssbot frames` and asks Claude Code to calibrate.
2. Claude reads ~10 frames with different lane situations, plus menu, crash and ad screens, and
   proposes in `calibration.toml`:
   - the three lane trapezoids at `near`, `mid` and `far`
   - the player strip
   - screen-marker boxes
3. `ssbot calibrate --preview` draws the zones on those frames as PNGs. Claude checks them visually
   and adjusts until the zones sit on the tracks.
4. The user gives final approval by looking at the previews.

#### Step 2: bulk labelling with the Claude API (`ssbot label`, Rust)
- **Image sent:** each frame scaled to 640 px wide. The calibrated zones are drawn on it as thin
  outlines with IDs (`L-near`, `C-mid`, …), so Claude labels *those exact zones* rather than
  describing the scene loosely. The labels then line up one-to-one with Rust perception's outputs.
- **Transport:** raw HTTPS with `reqwest`. There is no official Rust SDK, so these are the plain REST endpoints:
  - `POST https://api.anthropic.com/v1/messages/batches` with
    `{"requests":[{"custom_id":"<run>/<frame_id>","params":{...}}]}`
  - poll `GET /v1/messages/batches/{id}` until `processing_status == "ended"`
  - download the JSONL results from the batch's `results_url`. Match results **by `custom_id`**,
    because they can arrive in any order.
  - Headers: `x-api-key: $ANTHROPIC_API_KEY`, `anthropic-version: 2023-06-01`.
  - For a quick check of a few frames, `--sync` sends the same `params` to `POST /v1/messages` instead.
- **Request params:**
  ```jsonc
  {
    "model": "claude-opus-5-5",
    "max_tokens": 4000,
    "output_config": {
      "effort": "low",                      // labelling is perception, not deep reasoning; raise if accuracy is low
      "format": {"type": "json_schema", "schema": { /* FrameLabel, below */ }}
    },
    "system": "<fixed labelling guide: obstacle definitions with example descriptions, zone meanings, 'unsure' rules>",
    "messages": [{"role": "user", "content": [
      {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "<frame with zone overlay>"}},
      {"type": "text", "text": "Label this Subway Surfers frame. Zones are outlined and named."}
    ]}]
  }
  ```
  - **Model:** Claude Opus 5.5 by default. `--model claude-sonnet-5-5` halves the price; compare its
    agreement with Opus on a 50-frame sample before switching.
  - **System prompt:** keep it byte-identical across requests so prompt caching applies.
- **`FrameLabel` schema:** every object has `additionalProperties: false`, and every field is required.
  ```jsonc
  {
    "game_state": "Loading|AdBreak|Ad|Menu|Running|Crashed|RevivePrompt|NewHighScore|ScoreScreen|Paused|Unknown",
    "player_lane": "L|C|R|unknown",
    "player_action": "running|jumping|rolling|switching|unknown",
    "zones": [ {"zone": "L-near", "obstacle": "Free|TrainBody|TrainRamp|LowBarrier|HighBarrier|OverheadBar|Unknown",
                "coins": true, "powerup": false, "sure": true} /* ×9 zones */ ],
    "notes": "short free text for anything odd"
  }
  ```
  - There's no numeric confidence, because the schema can't constrain a number's range.
    `sure: false` marks uncertain zones; they're left out of training and listed for review.
- **Stored as** `labels/<run>.jsonl`, one `FrameLabel` per frame, plus `model`, `request_id` and the prompt version.

#### Step 3: checks (no hand-labelling everything)
1. **Labelling twice:** 10% of frames are labelled a second time with the zone list in a different
   order. Agreement under 90% on any class means fixing the labelling guide before continuing.
2. **Consistency over time:** an obstacle labelled `far` at time t should show up in `mid` or `near`
   a fraction of a second later. Frames that break this get flagged automatically.
3. **What actually happened:** when the game goes to `Crashed`, the player's lane in the ~0.3–1.0 s
   before must hold something deadly. If Claude labelled it `Free`, flag it. Those crash frames also
   become **automatic danger labels**: ground truth for collision detection that doesn't depend on Claude.
4. **Human spot-check:** `ssbot review` writes a static HTML gallery: each frame with its zone overlay
   and labels, flagged frames first. The user checks ~100 frames and fixes any wrong labels in the
   gallery, which saves corrections to `labels/<run>.fixes.jsonl`.
   - Target: **≥95% of spot-checked zone labels correct** before labels are used for training.

#### Cost (estimate, check with `count_tokens` on real frames)
- **Image:** a 640×360 frame is roughly 300 image tokens. Model docs describe tokens growing with
  pixel area, capped at around 1.6k–4.8k for full-resolution images.
- **Text in:** the system prompt is ~1k tokens and mostly cached.
- **Output:** about 300–1,000 tokens including thinking.
- **Per frame:** roughly $0.005–0.015 on Opus 5.5 with the batch discount. **1,000 frames ≈ $5–15.**
  Print the actual total from `usage` after each batch.

#### What doesn't go to Claude
- **No live play:** latency and cost rule it out.
- **No other content:** only cropped game-canvas frames, never the rest of the screen, so desktops
  and notifications aren't sent.

## 5. Latency budget (per frame, target)

| Stage | Budget | Notes |
|---|---|---|
| Capture (screencast delay) | ≤40 ms | Backend B if A can't meet this |
| Decode + resize | ≤5 ms | |
| Perception | ≤10 ms | |
| Facts + reflex + arbiter | ≤2 ms | |
| Key dispatch over CDP | ≤10 ms | |
| **Reflex path total** | **≤70 ms** | Must hold at p95 |
| Advisor (async, off the critical path) | ~250 ms (0.8B), 0 ms if cached | Used only while fresh |

## 6. Crate layout
```
subway_surfers_bot/
  SPEC.md
  Cargo.toml                      # virtual workspace: members = ["crates/*"], shared [workspace.dependencies]
  calibration.toml                # written by `ssbot calibrate`
  crates/ssbot-core/              # pure: no browser, no network, no labelling deps (tokio only for sync channels)
    src/lib.rs
    src/config.rs                 # owns `Wording`; imports nothing from policy
    src/facts.rs
    src/il.rs                     # IL inference: IL_ACTIONS, class_of, IlNet
    src/sidecar.rs                # sidecar protocol: WarmItem, Request, Reply, SidecarLink
    src/perception/{mod,zones,classify,state,model,cnn}.rs
    src/policy/{mod,reflex,arbiter,advisor}.rs
    examples/perceive.rs
  crates/ssbot-live/              # the realtime loop's I/O: browser, capture, process, recorder
    src/lib.rs
    src/bot.rs  src/browser.rs  src/recorder.rs
    src/capture/{mod,screencast,native}.rs   # feature `xcap = ["dep:xcap"]`
    src/sidecar.rs                # sidecar process: `Sidecar::spawn`
    tests/sidecar.rs              # #[ignore]
    examples/keytest.rs
  crates/ssbot-tools/             # offline tooling: labelling, training, benches, replay
    src/lib.rs
    src/label/{mod,batch,boxes,checks,cmd,overlay,review,select}.rs   # §4.8
    src/train.rs  src/bench.rs  src/replay.rs  src/see.rs  src/calibrate.rs  src/frames.rs
    src/il_data.rs                # IL dataset build: IlOpts, build
    src/fit.rs  src/advisor_bench.rs  src/advisor_data.rs
    tests/perception.rs           # + tests/fixtures/session1/ (labelled frames)
    tests/replay.rs
  crates/ssbot/src/main.rs        # bin `ssbot` (CLI); feature `xcap = ["ssbot-live/xcap"]`
  sidecar/openjev_sidecar.py, sidecar/mlx_openjev.py
  scripts/improve.py              # Typer: the bench → label → train → verify loop (offline glue)
  labels/                         # Claude labels (jsonl) + human fixes, kept in git (small)
  prompts/label_guide.md          # versioned labelling system prompt
```
**Dependency direction:** `ssbot-core` ← `ssbot-live` ← `ssbot-tools` ← `ssbot`, enforced by the compiler.
The per-frame path is `ssbot-core` + `ssbot-live` only.
**Dependencies:** `chromiumoxide`, `tokio`, `image`, `fast_image_resize`, `serde`/`serde_json`, `toml`,
`reqwest` (rustls, json) + `base64` + `image_hasher` (perceptual hashes) for labelling, `candle-nn` (optional, perception v2),
`clap`, `anyhow`, `tracing`; optional `xcap` (backend B) and `ort` (perception v2).

## 7. CLI
```
ssbot calibrate                 # opens the game, guides zone/marker capture, writes calibration.toml
ssbot run [--model 0.8b] [--no-advisor] [--capture screencast|native] [--runs N]
ssbot replay runs/<ts> [--no-advisor]
ssbot frames video.mov          # ffmpeg → cropped frames at 10 fps (before the recorder exists)
ssbot label <frames-dir> [--model claude-opus-5-5] [--max 600] [--sync]   # Claude labelling (§4.8)
ssbot review <frames-dir>       # static HTML gallery for spot-checks and fixes
```

## 8. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **M0** Spike | Script: launch Chrome, open the page, read the iframe origin, send ← → ↑ ↓ and **see the runner react**; measure screencast fps and delay; measure obstacle-visible-to-impact time | Answers to Phase 0 questions 1–5 ([§10](#10-risks-and-open-questions)) written into this spec |
| **M1** Capture + recorder | `ssbot run --no-advisor` records frames with timing while a human plays (keys pass through) | ≥20 fps at p50, delay ≤40 ms at p95 |
| **M1.5** Labels | `frames`, `label`, `review` working on ≥2 recorded sessions; calibration done with Claude Code | ≥1,000 labelled frames; labelling-twice agreement ≥90%; spot-check ≥95% correct; cost logged |
| **M2** Perception | Zone classifier and game-state detector fitted to the Claude labels and tested on a held-out split | Accuracy targets in [§9](#9-testing) met |
| **M3** Reflex autopilot | Rules-only play with the full start → run → crash → restart loop | Median survival ≥45 s over 10 runs, with no human input |
| **M4** Advisor | Sidecar, facts, cache and arbiter wired in | Decisions arrive fresh ≥80% of the time; reflex p95 still ≤70 ms |
| **M5** Evaluation | A/B: `--no-advisor` vs advisor (0.8B), 20 runs each | Report survival, coins and crash causes. **Either result is fine; the point is to measure it** |
| **M6** (optional) | Rust `mlx-rs` port of the sidecar | Same choices on ≥99% of recorded decisions, latency ≤ sidecar's |

## 9. Testing
- **Perception (unit):** Claude-labelled fixture frames (held-out split, human spot-checked) → `Observation`. Targets:
  - Obstacle class per zone ≥95% (`near` ≥98%)
  - Player lane ≥99%
  - Game state ≥99%
- **Policy (unit):** hand-written `Observation`s → the expected safety mask and reflex action, especially
  the "train in two lanes" and "barrier while airborne" cases.
- **Advisor (offline):** like `../laya_tictactoe/benchmark.py`. Build a set of fact strings with a
  known best action, score how often OpenJev picks it, and compare option wordings. Choose the wording
  by score; never use wording that names the best move, as in Appendix A.
- **Replay regression:** a fixed set of recorded runs; perception or policy changes must not increase
  the number of predicted crashes.
- **Run metrics (`summary.json`):**
  - Survival time
  - Crash cause, from the last 2 s of events
  - Actions per minute
  - Advisor freshness rate, cache hit rate
  - Reflex p50/p95 latency

## 10. Risks and open questions

| # | Question / risk | How it's settled |
|---|---|---|
| 1 | Iframe origin; do CDP key events reach the Unity canvas inside it? | M0: try page-level dispatch first, then the iframe target |
| 2 | Real screencast fps and delay on this Mac | M0 measurement; switch to native capture if too slow |
| 3 | Obstacle-visible-to-impact time over a run (sets the emergency window and whether the advisor is ever usable) | M0: frame-step through a recording |
| 4 | Pre-roll or mid-run ads and the revive dialog | Game-state detector plus handling for each state |
| 5 | Does the game pause on window blur, or run slower when CPU-throttled? | M0; keep the window visible and in front |
| 6 | Game or Poki visual updates break calibration | Calibration lives in a file; `ssbot calibrate` reruns in minutes |
| 7 | **Terms of use / fair play.** Automated play may break Poki's or SYBO's terms, and the page has a "Top Run" leaderboard | Personal, local use only. Never submit or share scores. Read the Poki terms before M3 and stop if they forbid it |
| 8 | The advisor adds nothing over rules | M5 measures it; a negative result is acceptable and gets documented |
| 9 | macOS Screen Recording permission (native capture only) | Granted once by the user; screencast needs none |
| 10 | Claude labels are wrong in a consistent way (e.g. ramp vs train body at a distance) | Labelling twice, time-consistency and crash checks, human spot-check, `sure:false` excluded; fix the guide and relabel only affected frames |
| 11 | Labelling cost grows with recording volume | Near-duplicate removal, per-session cap, batch discount, cost printed per batch; Sonnet 5.5 option after an agreement check |
| 12 | Sending frames to an external API | Only cropped game-canvas frames are sent; never full-screen captures |

### 10.1 Findings from recordings
**`recordings/session1.mov`** (2026-10-05, 12 s). Full-screen macOS capture, 3024×1788 at ~57 fps,
covering the screens after a run, with **no gameplay**. Frames are at `recordings/session1_frames/`.

**What happens between runs:**
1. **End of run:** a "New High Score! 1646" banner with **"Press Space to continue"**, shown in the
   normal page layout (the game embedded in the page, next to Poki's ads).
2. **Score screen:** score, coins, a **"Free Double Up! Get N coins"** button, a leaderboard
   (Yutani 999999, Spike 810312, …), and **Menu / Boosts / PLAY** buttons.
   - The game was in Poki's **fullscreen** mode here, so the layout differs from step 1.
3. **After PLAY: an ad break before the next run.**
   - Overlay "We'll be back after this short break", with a black loading screen and
     an animated icon plus progress bar for ~7 s.
   - Then a video ad with a countdown ("Playing Ad 0:13").
   - Then back to the page with the ad still showing.

**Effect on the spec:**
- **New screen types for `GameState`:** `NewHighScore` (press Space), `ScoreScreen` (click PLAY),
  `AdBreak` (wait; never click). The existing `Ad` covers the video stage.
- **Never click "Free Double Up"** or any other rewarded-ad button. Only PLAY and Space are allowed.
  The restart flow is: Space → wait for `ScoreScreen` → click the PLAY button → wait out
  `AdBreak`/`Ad` → `Menu`/`Running`.
- **Ads can take 15–20 s between runs.** That counts as downtime, not survival time, in the metrics.
- **Page vs fullscreen layout:** the game appears in both, with different geometry. Pick one mode
  for the bot (fullscreen preferred: bigger canvas, no side ads) and calibrate for it.
  Recordings for labelling must use the same mode.
- **Capture size:** retina full-screen frames are 3024×1788. `ssbot frames` must crop to the game
  canvas and scale down to 640 px wide before anything else.

**Still needed:** a recording of **actual running**, 1–2 minutes over 2–3 runs, in fullscreen mode.

### 10.2 M0 spike results (`ssbot spike`, 2026-10-05)
| # | Question | Answer |
|---|---|---|
| 1 | Iframe origin; do CDP key events reach the canvas? | Origin **`https://games.poki.com`**. **Page-level `Input.dispatchKeyEvent` reaches the game** after one click on the canvas: the runner jumped, rolled and switched to both side lanes. No iframe-target attach needed (`direct` mode kept as a fallback). Poki adds `bot=1` to the iframe URL, so it detects the CDP-driven browser. |
| 2 | Screencast fps and delay | **59–65 fps**, delay p50 **1.7–2.0 ms**, p95 **2.3–2.4 ms** (Chrome frame timestamp → Rust, so excludes compositing). Well inside ≥20 fps and ≤40 ms; backend B isn't needed. |
| 3 | Obstacle-visible-to-impact time | **Open.** Needs a normal run; a fresh profile starts in the tutorial (below). |
| 4 | Ads, revive dialog | **Open** for the bot's layout. Ads weren't seen in the spike. |
| 5 | Pause on blur / throttling | Not seen with the throttling flags in §4.1. |

**Other findings:**
- **A fresh Chrome profile starts in the first-run tutorial.** It shows "Press Arrow Key Up" and rewinds to the same barrier until the requested key comes at the right moment, so the score stays put (43). Clear it once by hand in the bot's profile (`~/.ssbot/chrome`) before measuring survival.
- **"Fullscreen" layout:** the bot stretches the game iframe to the viewport and hides every other page element. Without that, Poki's semi-transparent "Advertisement" boxes cover the track.
- **Clicking the canvas centre on the menu starts a run** ("PRESS TO PLAY"), so `Menu` needs no marker click.
- **Lane geometry in this layout** (Claude Code calibration from live frames, now in `calibration.toml`): vanishing point (0.455, 0.24); lane boundaries at y = 0.95 are 0.1275 | 0.3725 | 0.6175 | 0.8625; runner from y ≈ 0.66 to 0.86. Bands: near 0.56–0.67, mid 0.47–0.56, far 0.39–0.47.
- **The v1 colour heuristics misread this world with default thresholds.** The orange track tiles are saturated, so the centre lane reads `TrainBody` and the player lane is unreliable. As planned, fit thresholds to labels (M2) before M3; expect to need the v2 zone CNN if fitting can't separate orange tiles from trains.

## 11. Configuration (`ssbot.toml`)
```toml
[browser]   viewport = [1280, 720]; profile_dir = "~/.ssbot/chrome"
[capture]   backend = "screencast"; jpeg_quality = 60; max_width = 640
[policy]    emergency_ms = 250; cooldown_ms = 180; advisor_max_age_ms = 400
[advisor]   enabled = true; model = "0.8b"; backend = "mlx"; python = "../.venv/bin/python"
[recorder]  frame_sample_every = 3; keep_pre_crash_ms = 2000
```

## 12. Implementation status
**Built and tested** (`cargo test`: 52 unit + 4 integration tests; `cargo test -- --ignored` runs the model):
- **Every module in §6.** Plus `bot.rs` (run loop and spike), `calibrate.rs`, `frames.rs`, `replay.rs`, `label/cmd.rs`, `perception/fit.rs`, `policy/bench.rs`.
- **Extra CLI commands:**
  - `spike` (M0)
  - `fit` (M2: grid search plus held-out §9 report)
  - `advisor-bench` (§9 advisor offline)
  - `calibrate --init / --preview`
  - `run --human` (M1 recording)
  - `label --dry-run` (overlays and cost estimate without API calls)
  - `review --manual` (hand labelling in the gallery, no API needed)
  - `import` (bounding boxes from CVAT, Label Studio, makesense.ai or Roboflow, as COCO JSON or YOLO, turned into zone labels: a box labels a zone it covers ≥40%; 5–40% marks the zone unsure; the box lowest on screen wins)
- **Perception speed:** 0.38 ms p95 per frame at 320×180 (budget 10 ms).
- **Real-frame state test** (`tests/perception.rs`, frames from `recordings/session1.mov`): AdBreak and Ad fallbacks work. Marker templates detect NewHighScore and ScoreScreen and give the PLAY click point. Markers never match the page layout, so the bot never clicks there.
- **Sidecar** (`sidecar/openjev_sidecar.py`, end to end from Rust): OpenJev 0.8B loads in 3.5–4 s and decides in **~230–330 ms** with 5 options.

**Advisor benchmark** (`ssbot advisor-bench`, 27 situations with a known best move, 2026-10-05):

| Model | Wording | Raw | After reflex mask | ms |
|---|---|---|---|---|
| (rules only) | | 27/27 | 27/27 | 0 |
| 0.8B | position / "X is the best move because it …" | 8 | 24 | 295 |
| 0.8B | consequence / "The best move is x, because it …" | **14** | 25 | 218 |
| 4B | position | 22 | 26 | ~980 |
| 4B | consequence | **26** | 26 | 742–811 |

- **0.8B is close to chance on its own** and strongly biased toward Stay. The 4B with consequence wording nearly matches the rules but is too slow except through the cache. The default wording is now consequence/1.
- **Caveat:** the same author wrote the cases and the reflex rules, so rules-only = 27/27 is partly by construction. M5's live A/B is the real test.

**Deviations from the text above:**
- **Batch `custom_id`** is `<run>__<frame_id>`, because the API allows only `[A-Za-z0-9_-]`.
- **Refusals:** `fallbacks: "default"` is sent only on `--sync`, since the Batches API rejects it. Batch refusals are recorded as `status: "refused"`.
- **Effort and thinking:** `effort: "low"` is explicit because Opus 5.5 defaults to medium. Thinking is left at its default; disabling it returns a 400.
- **Facts wording:** "something unclear" became "unclear object" and "power-up ahead" became "power-up". That keeps the worst case at 60 tokens (OpenJev tokenizer).
- **Review fixes** download from the gallery as `<run>.fixes.jsonl`; save the file into `labels/`.

**Blocked on the owner:**
1. **Anthropic credentials** (`ANTHROPIC_API_KEY` or `ant auth login`) for `ssbot label` (M1.5). A dry run on session1 estimated about $0.27 for 39 requests.
2. **Read the Poki and SYBO terms** before letting the bot play on its own (§10 risk 7, M3).
3. **Clear the tutorial once** in `~/.ssbot/chrome`. Then record 2–3 runs with `ssbot run --human --no-advisor` and capture the screens with `ssbot calibrate` (markers for NewHighScore, ScoreScreen and RevivePrompt in this layout).
4. **M1.5 → M2 → M3 → M5 in order:** `label` → `review` → `fit --write` → `run --no-advisor --runs 10` → A/B with the advisor.

### 12.1 Labelling and OpenJev training (2026-10-05)
**Terms check (§10 risk 7):** Poki's Terms of Use forbid "any robot, spider or other automatic device, process or means to access the Website". Per this spec, the bot doesn't play on Poki. The M0 spike runs above were made before this check. More frames must come from a human playing, recorded with QuickTime, then `ssbot frames`.

**Labels:**
- **`labels/live1.jsonl`:** 17 frames from the M0 spike (`data/live1`), labelled zone by zone by Claude Code. 16 are Running, all in the centre lane during the tutorial; 1 is Menu.
- **`labels/session1.jsonl`:** all 122 frames of `recordings/session1.mov` (`data/session1`), labelled by game state: NewHighScore 15, ScoreScreen 11, Loading 2, AdBreak 64, Ad 30.
- Far short of §4.3's targets (≥300 zones per obstacle class; ≥50 per game state for the states not listed above). Only 3 low-barrier zones and no ramps, high barriers or overhead bars.

**Perception:**

| Part | Before | After | How |
|---|---|---|---|
| Zones | threshold rules: 56% | **94.9%**, near 95.7% (leave one frame out) | Learned zone classifier, `perception/model.rs` (logistic regression on HSV histograms and edges; `ssbot train-zones`). The rules couldn't tell orange track tiles from train paint. Frames are highly correlated, so this flatters generalisation. |
| Game state | 86% | 120/124 = 96.8% in the bot's layout | A Running marker (the in-run pause button) and Menu, NewHighScore and ScoreScreen markers. Any screen nothing matches is now Unknown (wait), never Running. All 15 page-layout frames and the 4 misses read as Unknown. |
| Player lane | not usable | tracked from the bot's own lane changes (`policy::LaneTracker`) | The camera follows the runner sideways, so screen position can't give the lane. |

- **Facts OpenJev receives** (held-out frames): exactly right on 10/16. The implied reflex move is right on 13/16. All 3 move errors involve low barriers or the tutorial overlay.

**OpenJev decision head** (`sidecar/train_head.py`, the OpenJev card's `LatentMLPHead` on frozen 0.8B latents):
- **Training data** (`ssbot advisor-data`):
  - 2,000 random situations; the correct move is the reflex policy's.
  - Two held-out test sets: the 27 benchmark cases, and 9 distinct situations from the labelled frames.
- **Results:**

| | Zero-shot 0.8B | Trained head |
|---|---|---|
| Training set | 746/2000 | 1870/2000 (validation 90%) |
| Benchmark (held out) | 14/27 | **26/27** |
| Real situations (held out) | 4/9 | **9/9** |
| Through the Rust sidecar | 14/27, 218 ms | **26/27, 233 ms** |

- **Enabled** with `advisor.head = "models/head-0.8b"`; `openjev_sidecar.py --head`.
- **What "correct" means here:** the teacher is the rule policy, so the head learns to rank the safe, coin-seeking move given correct facts. It doesn't learn from survival outcomes. M5's A/B is still the test of whether it beats rules, and that needs a legitimate place to run the bot.

### 12.2 Human-trained data and live check (2026-10-05)
**Data:**
- Recorded with `ssbot run --human`: a 3.5-minute human session (4,122 saved frames).
- 120 of those frames labelled (`labels/human_20261005-041856_952.jsonl`).
- `ssbot train-by-human` then retrained the lane classifier and OpenJev's head.

**Classifier check:**
- **Lanes:** 63 frames, 479 zones, 10-fold accuracy 89.6%.
  - Free 92%, train 86%, low barrier 2/5. Barriers remain the weak class.
  - The implied move matches the labelled scene on 45/56 frames.
  - The earlier 94% came from near-identical tutorial frames; this test spans five worlds.
- **Game state:** 230/259 = 88.8%, or 94.3% excluding page-layout frames the bot never sees. Every error is now "Unknown", where the bot waits.
- **Fixed along the way:**
  - The pause-button marker accepted the menu (running frames match at <1, the menu at 25; limit cut from 30 to 10).
  - The menu marker now uses the top-right buttons, because the bottom buttons animate in.
  - Added Loading, ScoreScreen (bot layout, PLAY at 0.58, 0.915) and hoverboard-shop markers.

**Live findings and fixes:**
1. **The camera follows the runner,** so the middle zones are the runner's own lane. Zones are now mapped to absolute lanes from the tracked lane (`perception::to_absolute`), and the lane two away reads "not visible". Before this, the bot looked at off-track scenery for its own lane.
2. **Space opens a "Need hoverboards? 300 coins" dialog that pauses the game.**
   - The hoverboard is now blocked on every decision path.
   - The dialog is recognised, and its ✕ is clicked (never the buy button).
3. **The bot dithered left and right every 0.2 s.** Fixed with:
   - a 3-frame majority vote per zone;
   - a 0.25 score margin before the default move changes lane;
   - no reversing a lane change within 600 ms unless in an emergency.
4. **OpenJev had never seen an off-screen lane:** 0/6000 training situations; all 23 vetoed live picks contained one.
   - Training data now includes them.
   - A held-out `test_offscreen` split guards the head comparison.
   - Retrained head: 89/104 on off-screen situations vs 85/104 for the previous head.
5. **Robustness:**
   - 15 s with no frames → save and stop.
   - A closed window → save and stop.
   - A leftover bot browser is closed at startup.
   - A truncated events log is still readable.

**Live session after the fixes** (3 runs):
- Running time per run: 5.8 s, 1.8 s, 3.9 s.
- OpenJev:
  - advice available on 157/679 running frames, used to decide 126;
  - 12 picks vetoed by the safety mask (8%, down from 17%);
  - advice fresh 86–93% (M4 target ≥80% met).
- Decision path p95 ≈ 4 ms.
- **OpenJev's picks equal the rule default whenever used,** as expected with the rule policy as its teacher. M5 still has to show whether it adds anything.
- **What limits survival:** barriers misread as trains, and jumps timed too early. That needs more labelled barrier frames.

---

## Appendix A: what the tic-tac-toe experiment established
Source: `../laya_tictactoe` (`benchmark.py`, `probe.py`, `bridge.py`, `mlx_openjev.py`).

1. **Models can't reason about space from raw layouts.**
   - On 200 forced-move positions, raw OpenJev 4B found the right move 55% of the time and Laya 36%, against 40% for random play.
   - Probes showed OpenJev reads stated facts reliably, but agrees with almost any claim about lines or wins.
   - So all geometry is computed in code, and only facts are given to the model.
2. **Labelling what each move does is the big win.**
   - Hard-mode labels took OpenJev 4B from 55% to **80%**, and the 0.8B reached 64%.
   - Labels in the claim beat labels in the premise.
   - *"X is O's best move because it &lt;effect&gt;."* was the best wording of 7 tried, spanning 54–86%.
   - Wording that says outright which move is best was ruled out, because then the labels make the decision.
3. **Latency on an Apple M4:**
   - Each pass costs about 75–250 ms of fixed overhead from the Gated DeltaNet layers.
   - Scoring a shared premise once (shared prefix) gives about 1.6× speed.
   - MLX matches PyTorch per move (0.8B ≈ 250 ms, 4B ≈ 880 ms) but loads 5–15× faster.
   - Running on the CPU is far slower (~4 s).
4. **Hiding latency works.** Precomputing replies for likely next states (pondering) gave 0 ms
   replies in play. Here, the facts-string cache plays that role.
5. **Model files:** `AlexWortega/openjev`: `qwen3.5-0.8b-nli-v2s-long` (1.7 GB),
   `qwen3.5-2b-nli-v5` (4.5 GB), `qwen3.5-4b-nli-v5` (9 GB), all cached locally. The labels are
   contradiction/entailment/neutral; the input template is `Premise: {premise}\nHypothesis: {hypothesis}`;
   the score is the entailment probability, normalised across options.
6. **Vision checkpoint:** only `qwen3.5-4b-nli-v2` accepts images. It isn't downloaded and isn't used
   here; text facts did as well as pixels in the model card's Doom results (about 11 vs 10.4 kills per episode).

## Appendix B: references
- **Game page:** https://poki.com/en/g/subway-surfers (developer: SYBO Games)
- **OpenJev:** https://huggingface.co/AlexWortega/openjev (MIT)
- **mlx-lm Qwen3.5:** `mlx_lm/models/qwen3_5.py`, `gated_delta.py` in https://github.com/ml-explore/mlx-lm
- **Claude API (labelling):**
  - `POST /v1/messages` and `/v1/messages/batches` (50% price; up to 100k requests or 256 MB per
    batch; most finish within 1 h, 24 h max; results kept 29 days)
  - Structured outputs via `output_config.format` (`json_schema`; all objects `additionalProperties: false`;
    no numeric range or string-length constraints)
  - Models: `claude-opus-5-5` ($4/$20 per M tokens), `claude-sonnet-5-5` ($2/$10)
  - Docs: https://platform.claude.com/docs
- **Crates:**
  - `chromiumoxide` 0.9.1, `xcap` 0.9.8, `ort` 2.0.0-rc.13
  - `image` 0.25.10, `fast_image_resize` 6.1.0, `tokio` 1.53
  - `mlx-rs` 0.32.0, `candle-core` 0.11 (no Qwen3.5)
