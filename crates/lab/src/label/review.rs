//! `ssbot review`: a static HTML gallery for the human spot-check (SPEC §4.8 step 3.4).
//! Each frame shows its zone overlay and labels, flagged frames first. Corrections are made
//! in place and downloaded as `<run>.fixes.jsonl`, to be saved under `labels/`.
//!
//! Manual mode (`--manual`) labels from scratch: every frame starts from perception's guess,
//! a click on a zone cycles its class, and the checked frames download as `<run>.jsonl` in the
//! same record format Claude's labels use, so `ssbot fit` reads them unchanged.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde_json::json;

use ssbot_engine::perception::zones::Calibration;
use ssbot_engine::perception::{GameState, Obstacle};

use super::FrameLabel;
use super::checks::Flag;

pub fn render(
    run: &str,
    frames: &[(u64, String)],
    labels: &HashMap<u64, FrameLabel>,
    flags: &[Flag],
    calib: &Calibration,
    manual: bool,
) -> String {
    let mut flagged: HashMap<u64, Vec<String>> = HashMap::new();
    for f in flags {
        flagged.entry(f.frame_id).or_default().push(f.reason.clone());
    }
    let mut items: Vec<_> = frames
        .iter()
        .filter_map(|(id, file)| {
            let l = labels.get(id)?;
            Some(json!({"id": id, "file": file, "label": l, "flags": flagged.get(id).cloned().unwrap_or_default()}))
        })
        .collect();
    items.sort_by_key(|v| (v["flags"].as_array().map(|a| a.is_empty()).unwrap_or(true), v["id"].as_u64()));
    let zones: Vec<_> = calib
        .lanes
        .iter()
        .enumerate()
        .flat_map(|(l, lz)| lz.bands().into_iter().enumerate().map(move |(b, q)| json!({"lane": l, "band": b, "pts": q.0})))
        .collect();
    let data = json!({
        "run": run,
        "manual": manual,
        "items": items,
        "zones": zones,
        "obstacles": Obstacle::ALL.iter().map(|o| format!("{o:?}")).collect::<Vec<_>>(),
        "states": GameState::ALL.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>(),
    });
    // `</` can't appear inside the inline script.
    let data = data.to_string().replace("</", "<\\/");
    TEMPLATE.replace("__RUN__", run).replace("__DATA__", &data)
}

pub fn write(path: &Path, html: &str) -> Result<()> {
    std::fs::write(path, html)?;
    Ok(())
}

const TEMPLATE: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label Review</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddd;--flag:#b3261e;--ok:#1b6e3a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--card:#21211f;--fg:#ecebe6;--muted:#a3a29b;--line:#3a3a37;--flag:#f2827a;--ok:#7dd69a}}
:root[data-theme=dark]{--bg:#161615;--card:#21211f;--fg:#ecebe6;--muted:#a3a29b;--line:#3a3a37;--flag:#f2827a;--ok:#7dd69a}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;flex-wrap:wrap;align-items:center;z-index:2}
h1{font-size:16px;margin:0}
.stat{color:var(--muted)} button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
main{padding:16px;display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.card.checked{outline:2px solid var(--ok)}
.pic{position:relative;width:100%} .pic img{width:100%;display:block;border-radius:4px} .pic svg{position:absolute;inset:0;width:100%;height:100%}
.flags{color:var(--flag);margin:6px 0} table{width:100%;border-collapse:collapse;margin-top:6px} td{padding:2px 4px;border-top:1px solid var(--line)}
select{font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:4px}
.changed{background:color-mix(in srgb,var(--flag) 18%,transparent)} .unsure{color:var(--muted);font-style:italic}
</style></head><body>
<header><h1>Label review · __RUN__</h1><span class="stat" id="stat"></span>
<button id="dl">Download</button><span class="stat" id="saveas"></span></header>
<p class="stat" id="help" style="padding:0 16px"></p>
<main id="grid"></main>
<script>
const D = __DATA__;
const LANES = ["L","C","R"], BANDS = ["near","mid","far"], COLOURS = ["#00e6ff","#ff3cff","#fff000"];
const fixes = new Map(); const checked = new Set(); const selects = new Map(); const polys = new Map();
const FILL = {Free: "none", TrainBody: "#ff3b30", TrainRamp: "#ff9500", LowBarrier: "#ffcc00", HighBarrier: "#af52de", OverheadBar: "#5856d6", Unknown: "#8e8e93"};
function paint(id, zone, obstacle){
  const p = polys.get(key(id, zone)); if (!p) return;
  p.setAttribute("fill", FILL[obstacle] || "none"); p.setAttribute("fill-opacity", "0.45");
  p.innerHTML = ""; const t = document.createElementNS(p.namespaceURI, "title"); t.textContent = zone + ": " + obstacle; p.appendChild(t);
}
document.getElementById("saveas").textContent = D.manual ? `Save as labels/${D.run}.jsonl (checked frames only)` : `Save as labels/${D.run}.fixes.jsonl`;
document.getElementById("help").textContent = D.manual
  ? "Click a zone on the picture to cycle its class (red train, orange ramp, yellow low barrier, purple high barrier, indigo overhead bar, grey unknown, clear = free). Set game state and lane, then tick checked."
  : "Change any wrong label, tick checked on frames you've reviewed, then download the fixes.";
function key(id, what){ return id + "|" + what; }
function stat(){
  let zc = 0; for (const k of fixes.keys()) if (!k.endsWith("|state") && !k.endsWith("|lane")) zc++;
  let total = 0; for (const it of D.items) if (checked.has(it.id)) total += it.label.zones.length;
  const acc = total ? (100 * (1 - zc / total)).toFixed(1) + "%" : "–";
  document.getElementById("stat").textContent = `${checked.size}/${D.items.length} frames checked · ${zc} zone fixes · spot-check accuracy ${acc} (target ≥95%)`;
}
function sel(options, value, onchange){
  const s = document.createElement("select");
  for (const o of options){ const e = document.createElement("option"); e.value = e.textContent = o; if (o === value) e.selected = true; s.appendChild(e); }
  s.onchange = () => onchange(s.value, s); return s;
}
const grid = document.getElementById("grid");
for (const it of D.items){
  const card = document.createElement("div"); card.className = "card";
  const head = document.createElement("div");
  head.innerHTML = `<b>#${it.id}</b> · ${it.file} · ${it.label.player_action} `;
  head.appendChild(sel(D.states, it.label.game_state, (v, s) => { if (D.manual) { it.label.game_state = v; return; } fixes.set(key(it.id,"state"), {frame_id: it.id, game_state: v}); s.classList.add("changed"); stat(); }));
  head.append(" lane ");
  head.appendChild(sel(["L","C","R","unknown"], it.label.player_lane, (v, s) => { if (D.manual) { it.label.player_lane = v; return; } fixes.set(key(it.id,"lane"), {frame_id: it.id, player_lane: v}); s.classList.add("changed"); stat(); }));
  const cb = document.createElement("label"); cb.innerHTML = ` <input type="checkbox"> checked`;
  cb.querySelector("input").onchange = e => { e.target.checked ? checked.add(it.id) : checked.delete(it.id); card.classList.toggle("checked", e.target.checked); stat(); };
  head.appendChild(cb); card.appendChild(head);
  if (it.flags.length){ const f = document.createElement("div"); f.className = "flags"; f.textContent = "⚑ " + it.flags.join(" · "); card.appendChild(f); }
  const pic = document.createElement("div"); pic.className = "pic";
  const img = document.createElement("img"); img.src = it.file; img.loading = "lazy"; pic.appendChild(img);
  const svg = document.createElementNS("http://www.w3.org/2000/svg","svg"); svg.setAttribute("viewBox","0 0 1 1"); svg.setAttribute("preserveAspectRatio","none");
  for (const z of D.zones){
    const p = document.createElementNS(svg.namespaceURI,"polygon");
    const zid = LANES[z.lane] + "-" + BANDS[z.band];
    p.setAttribute("points", z.pts.map(q => q.join(",")).join(" "));
    p.setAttribute("stroke", COLOURS[z.lane]); p.setAttribute("stroke-width","0.003");
    polys.set(key(it.id, zid), p);
    if (D.manual) {
      p.style.cursor = "pointer";
      p.onclick = () => {
        const zl = it.label.zones.find(x => x.zone === zid); if (!zl) return;
        zl.obstacle = D.obstacles[(D.obstacles.indexOf(zl.obstacle) + 1) % D.obstacles.length]; zl.sure = true;
        const s = selects.get(key(it.id, zid)); if (s) s.value = zl.obstacle;
        paint(it.id, zid, zl.obstacle);
      };
    }
    svg.appendChild(p);
  }
  pic.appendChild(svg); card.appendChild(pic);
  const t = document.createElement("table");
  for (const z of it.label.zones){
    const tr = document.createElement("tr");
    const td0 = document.createElement("td"); td0.textContent = z.zone; if (!z.sure){ td0.className = "unsure"; td0.textContent += " (unsure)"; }
    const td1 = document.createElement("td");
    const orig = z.obstacle;
    const zs = sel(D.obstacles, z.obstacle, (v, s) => {
      if (D.manual) { z.obstacle = v; z.sure = true; paint(it.id, z.zone, v); return; }
      const k = key(it.id, z.zone);
      if (v === orig) { fixes.delete(k); s.classList.remove("changed"); } else { fixes.set(k, {frame_id: it.id, zone: z.zone, obstacle: v}); s.classList.add("changed"); }
      stat();
    });
    selects.set(key(it.id, z.zone), zs);
    td1.appendChild(zs);
    const sc = document.createElement("label"); sc.innerHTML = ' <input type="checkbox"> unsure';
    sc.querySelector("input").checked = !z.sure; sc.querySelector("input").onchange = e => { z.sure = !e.target.checked; };
    if (D.manual) td1.appendChild(sc);
    const cc = document.createElement("label"); cc.innerHTML = ' <input type="checkbox"> coins';
    cc.querySelector("input").checked = z.coins; cc.querySelector("input").onchange = e => { z.coins = e.target.checked; };
    if (D.manual) td1.appendChild(cc);
    const td2 = document.createElement("td"); td2.textContent = (z.coins ? "coins " : "") + (z.powerup ? "power-up" : "");
    tr.append(td0, td1, td2); t.appendChild(tr);
  }
  card.appendChild(t); grid.appendChild(card);
  for (const z of it.label.zones) paint(it.id, z.zone, z.obstacle);
}
document.getElementById("dl").onclick = () => {
  if (D.manual) {
    const recs = D.items.filter(it => checked.has(it.id)).map(it => JSON.stringify({
      custom_id: D.run + "__" + it.id, run: D.run, frame_id: it.id, frame: it.file, repeat: false,
      label: it.label, status: "succeeded", error: null, model: "human", request_id: null,
      prompt_version: "manual", usage: null }));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([recs.join("\n") + (recs.length ? "\n" : "")], {type: "application/x-ndjson"}));
    a.download = D.run + ".jsonl"; a.click(); return;
  }
  const lines = [...fixes.values()].map(f => JSON.stringify(Object.assign({zone:null, obstacle:null, game_state:null, player_lane:null}, f))).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines + (lines ? "\n" : "")], {type: "application/x-ndjson"}));
  a.download = D.run + ".fixes.jsonl"; a.click();
};
stat();
</script></body></html>
"##;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::{ZoneLabel, all_zone_ids};

    #[test]
    fn flagged_frames_come_first_and_data_is_embedded() {
        let label = FrameLabel {
            game_state: GameState::Running,
            player_lane: "C".into(),
            player_action: "running".into(),
            zones: all_zone_ids().into_iter().map(|z| ZoneLabel { zone: z, obstacle: Obstacle::Free, coins: false, powerup: false, sure: true }).collect(),
            notes: "</script> attempt".into(),
        };
        let labels: HashMap<u64, FrameLabel> = [(1, label.clone()), (2, label)].into();
        let frames = vec![(1, "1.jpg".to_string()), (2, "2.jpg".to_string())];
        let flags = vec![Flag { frame_id: 2, reason: "odd".into() }];
        let html = render("r", &frames, &labels, &flags, &Calibration::default(), false);
        assert!(!html.contains("</script> attempt"));
        let first2 = html.find("\"id\":2").unwrap();
        let first1 = html.find("\"id\":1").unwrap();
        assert!(first2 < first1);
    }
}
