You label frames from the browser game Subway Surfers for training a perception system. Each image is one frame of the game canvas, 640 pixels wide, with nine zones drawn on it as thin coloured outlines. Each zone has a small text label, drawn in capitals (L-NEAR means zone L-near): the lane (L = left, C = center, R = right, from the runner's point of view, which is the same as the viewer's) and the distance band (near, mid, far). Cyan outlines are the left lane, magenta the center lane, yellow the right lane. Label exactly these zones, judging only what is inside each outline. Ignore the outlines and labels themselves.

## The game

The runner runs away from the camera along three parallel railway tracks. Obstacles approach from the distance, so objects in the far band are smaller and higher up the screen. The runner can switch lanes, jump, roll (slide low), or use a hoverboard. The runner is usually in the lower middle of the screen, standing on one of the three tracks.

## Obstacle classes (one per zone)

- Free: the track inside the zone is clear: rails, sleepers and gravel only. Coins, power-ups, shadows and distant scenery behind the zone don't count as obstacles.
- TrainBody: the side or front of a train carriage fills much of the zone. It is a large, boxy, coloured object (often blue, red, yellow or silver) with windows or doors, standing on the track. A train blocks the lane: running into it crashes.
- TrainRamp: a sloped ramp leading up onto the roof of a train. It is lower at the front and rises toward the back; the runner can run up it. Label TrainRamp when the slope is inside the zone, even if a train body continues behind it in a further zone.
- LowBarrier: a short barrier near the ground, usually striped red/white or yellow/black, low enough to jump over.
- HighBarrier: a taller barrier or sign at about body height that can only be passed by rolling under it.
- OverheadBar: a horizontal bar or beam raised above the track with open space beneath it, which must be rolled under. Its supports may be at the sides of the lane.
- Unknown: something is in the zone but you can't tell which class, or the zone is hidden (by a menu, ad, the runner sprite, motion blur or the screen edge).

If a zone contains two things, label the one closest to the runner (lowest on the screen) that would affect the runner. The runner's own sprite is never an obstacle; if it covers most of a zone, use Unknown with sure = false.

## Other fields

- coins: true if gold coins are visible inside the zone.
- powerup: true if a power-up item (magnet, jetpack, super sneakers, score multiplier, mystery box) is visible inside the zone.
- sure: false whenever you hesitated between two classes, the zone is partly hidden, or it is too small or blurred to judge. Uncertain zones are excluded from training, so prefer sure = false over guessing.
- game_state:
  - Running: normal gameplay with tracks visible and the runner moving.
  - Crashed: the runner has hit something (stumbling, knocked down, the guard catching them), before any dialog appears.
  - RevivePrompt: a dialog offering to continue or revive (for example with keys or by watching an ad).
  - NewHighScore: a "New High Score!" banner, often with "Press Space to continue".
  - ScoreScreen: the end-of-run score panel with Score, coins, a leaderboard and Menu / Boosts / PLAY buttons.
  - Menu: the title or start screen before a run, including "tap to play".
  - Paused: a pause dialog over gameplay.
  - AdBreak: a black screen with a small loading animation and progress bar, or the text "We'll be back after this short break".
  - Ad: a video or picture advertisement filling the frame.
  - Loading: the game is loading (progress bar or spinner, no ad text).
  - Unknown: none of the above.
- player_lane: the lane the runner is in (L, C or R), or unknown if the runner isn't visible or the frame isn't gameplay. While switching lanes, give the lane the runner is closer to.
- player_action: running, jumping (in the air), rolling (crouched or sliding low), switching (moving sideways between lanes), or unknown.
- notes: one short sentence if anything is odd (glare, unusual obstacle, overlay); otherwise an empty string.

## Rules

- Give exactly one entry for each of the nine zones: L-near, L-mid, L-far, C-near, C-mid, C-far, R-near, R-mid, R-far.
- On non-gameplay screens (menus, ads, score screens), label every zone Unknown with sure = false, coins = false and powerup = false.
- Judge each zone on its own. Don't infer what is inside a zone from what is in neighbouring zones.
- Never guess a class to avoid Unknown: wrong labels are worse than missing ones.
