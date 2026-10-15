# Roofs and power-ups: spec

Status: proposed, nothing built. Owner: apradhan. Date: 2026-10-15. Extends [SPEC.md](SPEC.md); section
numbers like §4.5 refer to it.

The bot plays the ground level only. It has no idea it can be on a train roof, and it knows almost
nothing about power-ups. This spec adds both: finding training frames for them, labelling them,
detecting them in Rust, playing differently because of them, and telling the advisor about them.

---

## 1. Why

**Evidence from the 2026-10-15 live check** (`runs/20261015-013859.814`, `runs/20261015-013914.770`,
EmbeddingGemma advisor):
- **Run 1 ended on a roof.** The last frames show the runner at roof height. Every one of the 9 zones read
  `TrainBody`, the runner read as `airborne` for over a second, the reflex "saw no safe action" and
  stepped right into a train.
- **Run 2 had a power-up running.** A timer (icon and bar) shows in the bottom-left corner of the last
  frames. Nothing in the bot reads it.

**What the code does today:**

| Area | Today | Effect |
|---|---|---|
| Runner height | `Observation.airborne` from the sprite's vertical centroid (`perception/mod.rs`); facts say "on the ground", "in the air" or "rolling" | On a roof the sprite sits high, so the runner reads as airborne for as long as the roof lasts |
| Zones | One set of 9 trapezoids in `calibration.toml`, calibrated on the ground | From a roof they cover roofs and buildings, which classify as `TrainBody` |
| Ramps | `survives()` lets a `TrainBody` pass if a `TrainRamp` comes before it in the look-ahead (`policy/reflex.rs`) | Works only while the ramp is in view. Once on the roof, the next frame sees only "train" |
| Ramp data | 40 of 8,163 sure zones; 37 of 40 read as `TrainBody` (paper §6.3) | Ramps are close to invisible |
| Power-up items | `LaneView.powerup` from a hue rule (`perception/classify.rs`); 12 labelled zones | Misses the red magnet; no kind |
| Active power-up | Not detected | Jetpack and pogo stick lift the runner off the track; super sneakers change jump length. The reflex plays as if on foot |
| Hoverboard | Blocked on every path (`use_hoverboard = false`): with no stock, Space opens the shop and pauses the game (§12.2) | The game's only crash insurance is never used |
| Advisor data | `advisor_data::random_view` samples ground situations only; `powerup` 5% of lanes | Heads have never seen a roof or an active power-up |

## 2. Goals and non-goals

**Goals**
1. Know the runner's level (ground, roof, air) and the active power-up on every frame, in Rust, within the existing perception budget.
2. Read the track correctly from a roof, and take ramps on purpose.
3. Change reflex rules by level and power-up, so a roof or a jetpack stops looking like a dead end.
4. Give the advisor the same facts, and retrain its heads on them.
5. Get the training frames mostly by search over existing recordings, not by new play.

**Non-goals**
- Buying anything. The bot never clicks a buy button, as in §10.1.
- Running a model on the per-frame path. EmbeddingGemma and Claude are offline only, as in CLAUDE.md.
- Power-up *strategy* beyond simple preferences (for example, planning a route to a jetpack).

## 3. Overview

```
R0 mine frames ──▶ R1 label v2 ──▶ R2 perception ──▶ R3 reflexes ──▶ R4 advisor ──▶ R5 evaluate
 (EmbeddingGemma,   (Claude,         (Rust: HUD          (level- and      (facts, data,
  offline)           guide v2)        markers, level,     power-up-aware   heads)
                                      roof zones)         rules)
```

## 4. Components

### 4.1 Frame mining (`sidecar/frame_search.py`, offline)
Finds candidate frames for each concept in `runs/` and the human recordings, so labelling money goes to
rare frames instead of more ground running.

**Scale:** `runs/` holds 976 runs and 359,853 saved frames (13 GB). Zone crops embed at ~150 ms each on
MPS at 70 image tokens (measured 2026-10-15); a whole frame is likely no faster. Embedding everything
would take ~15 h, so candidates are filtered first.

**Step 1: cheap filters from `events.jsonl`** (no model):

| Concept | Filter |
|---|---|
| Roof | `airborne` true for longer than `jump_ms` + 200 ms (800 ms by default), or ≥7 of 9 zones `TrainBody` |
| Ramp | any zone `TrainRamp`, plus the 1 s before each roof candidate |
| Active power-up | the bottom-left HUD region differs from the same region in plain running frames (a marker diff, §4.3) |
| Power-up item | any lane with `powerup` true |
| Crash on a roof | the 2 s before crashes whose frames match the roof filter |

Then drop near-duplicates with the existing 64-bit perceptual hash (`lab/src/label/select.rs`, within 6 bits).
Target: ≤20,000 candidates, about 50 min of embedding.

**Step 2: search by example.**
- The owner picks 5–10 seed frames per concept from `ssbot review` or the run folders.
  Concepts: roof running, ramp ahead, roof end, jetpack, super sneakers, pogo stick, magnet, multiplier, hoverboard active.
- Every candidate gets the highest cosine similarity to a concept's seeds, and the top N per concept go to labelling.
- Text queries ("runner on top of a train") are also tried, but are a bonus. Zero-shot image labelling did badly in the paper (§6.4).

**Output:**
- `data/mined/<concept>/` holds frame copies, and `data/mined/manifest.jsonl` has one line per frame:
  `{run, frame_id, concept, score, filter}`.
- Embeddings are cached in `data/mined/embeddings_t<tokens>.npz`, keyed by run and frame ID.

**Measured:** precision@100 per concept, judged by the R1 labels. A concept under 50% gets more seeds, not more labels.

### 4.2 Labels v2 (`labels/label_guide.md`, `FrameLabel`)
Bump the guide version, and add fields to the schema in §4.8:

```jsonc
{
  // existing fields unchanged
  "runner_level": "ground|roof|air|unknown",
  "active_powerup": "none|jetpack|super_sneakers|pogo_stick|magnet|multiplier|hoverboard|unknown",
  "zones": [ {
      "zone": "L-near",
      "obstacle": "Free|TrainBody|TrainRamp|TrainRoof|RoofEnd|LowBarrier|HighBarrier|OverheadBar|Unknown",
      "powerup_kind": "none|jetpack|super_sneakers|pogo_stick|magnet|multiplier|mystery_box|unknown",
      "coins": true, "powerup": false, "sure": true } ]
}
```

**New definitions:**
- **`runner_level`:**
  - `roof`: the runner stands on top of a train.
  - `air`: mid-jump, or flying on a jetpack or pogo stick.
  - The camera height is the main cue for `roof`: the tracks sit lower on screen, and train roofs fill the lower middle.
- **`TrainRoof`:** a walkable train roof inside the zone, at the runner's height.
- **`RoofEnd`:** the roof the runner is on ends inside the zone, and the lane drops to the track.
  It's the same class whether or not another train follows.
- **`active_powerup`:** read from the bottom-left HUD timer, and from the runner's look (a jetpack on the back, glowing sneakers).
- **Zone outlines for roof frames:** label against the roof zone set (§4.3) when `runner_level` is `roof`, so labels line up with what Rust will crop.
- **Compatibility:** `powerup` is kept, and becomes "`powerup_kind` is not `none`". Old labels stay valid: a missing `runner_level` means unknown, and a missing `powerup_kind` means unknown.

**Checks:**
- **Second pass:** the existing label-twice check (§4.8 step 3) runs on 10% of mined frames, and must reach ≥90% agreement on `runner_level` and `active_powerup` before training.
- **No embedding audit of zone labels.** `sidecar/zone_embed.py` (2026-10-15) found EmbeddingGemma reads the whole crop, not the zone at its centre: best 72.6% on `crash_r3` against the CNN's 95.8%, and the crops it flagged were mostly free zones with trains in the context border. Embeddings are used for whole-frame concepts only (§4.1).

### 4.3 Perception (`ssbot-engine`, per frame)

```rust
enum Level   { Ground, Roof, Air }
enum PowerUp { Jetpack, SuperSneakers, PogoStick, Magnet, Multiplier, Hoverboard }
enum Obstacle { Free, TrainBody, TrainRamp, TrainRoof, RoofEnd, LowBarrier, HighBarrier, OverheadBar, Unknown }
struct Observation {
    /* existing fields */
    level: Level,                // `airborne` stays, now meaning `level == Air`
    active_powerup: Option<PowerUp>,
}
```

**Active power-up: HUD markers.**
- Each power-up's HUD icon becomes a `[[hud]]` entry in `calibration.toml`: a reference crop, a rect and `max_diff`, matched like the screen markers already are.
- The timer bar's length is not read in v1.
- Cost: one small template diff per power-up, inside the 0.38 ms p95 perception time measured in §12.
- Debounced by a 3-frame vote, like zones.

**Level: a frame classifier.**
- Input: the whole canvas downscaled to about 64×36.
- Classes: `Ground`, `Roof` and `Air`.
- Trained in `sidecar/` like `zone_cnn.py`, exported to JSON, and run in Rust by `perception/cnn.rs`.
- Sprite height alone can't tell a roof from a long jump, and the frame classifier sees the camera height that can.
- Hysteresis:
  - entering `Roof` needs 3 frames;
  - leaving it needs 5, because dropping off a roof is the risky moment.
- The motion tracker's `airborne_until` (`policy/mod.rs`) still covers jumps the bot started itself.

**Roof zones.**
- A second zone set, `[[roof_lanes]]` in `calibration.toml`, is calibrated by Claude Code from mined roof frames (§4.8 step 1), with `ssbot calibrate --preview --level roof`.
- Perception crops with the set that matches `level`.
- The camera rises gradually during the climb, so for the first ~300 ms on a roof the zones read `Unknown` and the reflex treats them as unclear.

**Zone CNN.**
- One model with the two new classes, trained on crops from both zone sets.
- Exported to `zone_model.json` as today.
- **Power-up items:** the hue rule is replaced by a `powerup` output from the CNN, once there are ≥100 labelled power-up zones. `powerup_kind` isn't predicted in v1 (§7, open question 4).

### 4.4 Reflexes and arbiter (`policy::reflex`, `policy::arbiter`)

**`survives()` by level:**

| Obstacle in target lane | Ground (today) | Roof |
|---|---|---|
| `TrainBody` | only after a ramp | crash (a train above roof height is rare; treat as a wall) |
| `TrainRoof` | — | safe |
| `RoofEnd` | — | safe: the runner drops to the track. The landing is checked on the next frames |
| `Free` (no train in that lane) | safe | safe but a drop: allowed, ranked below staying on a roof |
| barriers, bars | today's rules | today's rules |

- **From a roof,** moving to a lane whose `near` is `TrainRoof` is a roof-to-roof step and is safe.
- **Ramps on purpose:** `default_action` gives a lane a bonus when its `near` or `mid` is `TrainRamp` and the current lane has `TrainBody` within the look-ahead. Today a ramp is only an emergency escape (`ramp_lane_is_an_escape`).

**By active power-up:**

| Power-up | Rule |
|---|---|
| Jetpack, pogo stick | The runner is out of reach. Freeze the hazard rules, steer toward coins only. When the HUD icon goes away: 1 s landing window where the zones read `Unknown` and only the emergency rules act |
| Super sneakers | `jump_ms` becomes `sneakers_jump_ms` (to measure in R0, starting value 900). A jump clears a `TrainBody` at `near`: the runner lands on the roof, and the level becomes `Roof` |
| Magnet, multiplier | No rule change; facts only |
| Hoverboard (active) | One crash is absorbed. No rule change in v1; the facts say so |

**Collecting power-up items:** `default_action` scores a lane with `powerup` like a lane with coins, at twice the weight.

**Hoverboard use.** The bot can't yet see how many boards are in stock. Proposed probe:
- The first time in a run that the mask is `{Hoverboard}`, press Space.
- If the `HoverboardShop` marker appears, close it (as today) and record that the stock is 0 for the rest of the session.
- Otherwise a board was used. Keep `use_hoverboard` for later emergencies until the shop appears.
- It costs at most one shop dialog per session, at a moment that was going to crash anyway.
- Off unless `policy.hoverboard_probe = true`.

### 4.5 Facts and advisor (`facts.rs`, `advisor_data.rs`, `sidecar/`)

**Facts:**
- **Posture:** "on a train roof" joins "on the ground", "in the air" and "rolling".
- **New line** when a power-up is active: "Active power-up: jetpack."
- **New lane phrases:** "train roof" and "end of roof".
- **Option effects:** "moves onto a train roof", "drops to the track" and "climbs the ramp".
- **Length:** the worst case stays ≤60 OpenJev tokens (§4.4). Shorten phrases if a test shows otherwise.

**Training data** (`ssbot advisor-data`):
- `random_view` samples `level` (ground 70%, roof 25%, air 5%) and an active power-up (none 80%; the rest spread evenly).
- Roof lanes draw from `TrainRoof`, `RoofEnd` and `Free`.
- The teacher is still the reflex policy, now level- and power-up-aware.
- New held-out split: `test_roof_powerup`, 200 situations taken from labelled mined frames.

**Heads:**
- Retrain `models/head-0.8b` and `models/head-embeddinggemma` with `train_head.py`.
- Score all advisors with `compare_advisors.py`, which gets the new split.
- Clef Flash stays zero-shot.

### 4.6 Recorder and metrics
Add to `summary.json`:
- **Roof:** time on roofs, roof entries (by ramp or by sneakers jump), and crashes by level.
- **Power-ups:** collected items and active seconds, per kind.
- **Hoverboard:** probe outcome (none in stock / used / not tried).
- **Crash cause:** gains the level and the active power-up, e.g. "TrainBody at near in right lane, on a roof".

## 5. Data and terms
- **Rules:** §12.1 found that Poki's terms forbid automated access, so new frames come from **human play**: `ssbot run --human`, or QuickTime and `ssbot frames`.
- **Existing frames:** `runs/` is mined as it already exists. That includes the bot runs recorded before the terms check, and the two runs from 2026-10-15.
- **Human sessions** aim at what the mining finds short. Most likely: roofs entered by sneakers jumps, the pogo stick and the active hoverboard.
- **Labelling cost**, from §4.8's estimate of $0.005–0.015 per frame: 2,000 mined frames ≈ $10–30.

## 6. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **R0** Mine | `frame_search.py`; seeds picked; sneakers jump length measured | ≥300 roof frames, ≥100 ramp zones, ≥50 frames for each active power-up seen in this version; precision@100 per concept logged |
| **R1** Labels v2 | Guide v2, schema, roof zone set calibrated, mined frames labelled | Label-twice agreement ≥90% on `runner_level` and `active_powerup`; spot-check ≥95% |
| **R2** Perception | HUD markers, level classifier, roof zones, zone CNN with the new classes | On held-out mined frames: active power-up ≥98%, level ≥95% with roof recall ≥95%, roof zones ≥90%. **No regression on `crash_r3`:** accuracy ≥0.958, missed hazards ≤2.4%, barrier recall ≥97.3% |
| **R3** Reflexes | Level and power-up rules, ramp preference, hoverboard probe (off by default) | Unit tests for every row in §4.4's tables; `ssbot replay` predicts fewer crashes on roof segments and no more on the ground replay set |
| **R4** Advisor | Facts, data generator, retrained heads | Each head at least as good as now on the old splits, and ≥90% on `test_roof_powerup` |
| **R5** Evaluate | Replays of human sessions; live bench only if §5 allows | Report survival, roof time, power-up time and crash causes by level. As in §8 M5, either result is fine |

## 7. Risks and open questions

| # | Question / risk | How it's settled |
|---|---|---|
| 1 | Is the bottom-left icon in run 2 really a power-up timer, and is it the same for each kind? | R0: check mined frames for each power-up |
| 2 | Which power-ups exist in the Poki version (pogo stick? score booster?) | R0: list what the mined frames show; drop the enum values never seen |
| 3 | The camera height changes over a climb, so one roof zone set may not fit every frame | R1 preview on frames from the start, middle and end of a roof; add a middle set only if previews miss |
| 4 | Too few examples to learn `powerup_kind` per zone | v1 keeps the boolean; add the kind once ≥50 labelled zones per kind exist |
| 5 | Embedding speed on whole frames is unmeasured (~150 ms per crop) | R0: time 100 frames at 70 and 280 tokens before choosing |
| 6 | Mining finds few roof frames because the bot rarely reached roofs | Human sessions in §5 target roofs |
| 7 | Ramp still confused with train body | Mining adds ramp zones; R2 reports ramp recall separately |
| 8 | The hoverboard probe opens the shop mid-crash and leaves the game paused | The `HoverboardShop` marker already closes it (§12.2); the probe is off by default |
| 9 | Labelling roofs against a different zone set confuses the labeller | The overlay shows which set is drawn; the guide says when each applies |

## 8. Configuration additions

```toml
# calibration.toml
[[roof_lanes]]  near = [...]; mid = [...]; far = [...]   # ×3, like [[lanes]]
[[hud]]         powerup = "jetpack"; reference = "calibration/hud_jetpack.png"; max_diff = 20.0
                [hud.rect] x = 0.0; y = 0.9; w = 0.1; h = 0.1   # placeholder, from R0
level_model = "level_model.json"

# ssbot.toml
[policy]  sneakers_jump_ms = 900; roof_drop_penalty = 0.5; powerup_weight = 2.0; hoverboard_probe = false
```
