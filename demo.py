"""F4: the demo surface - one static HTML file built from out/alerts.json.

Static and self-contained on purpose: nothing here can break on stage. Click
a row, the evidence clip seeks and plays. The "why" beside each alert is the
VLM's own sentence, quoting the same facts the geometry stage measured.

    python run.py --video theirs.mp4 --preset day
    python demo.py --alerts out/alerts.json --out out/demo.html
"""
import argparse, html, json, os

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Alert Timeline</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0d1214;color:#e6edf0;margin:0}}
.wrap{{max-width:900px;margin:0 auto;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}}
.meta{{color:#8fa3ab;font-size:13px;margin-bottom:16px;font-family:monospace}}
video{{width:100%;border-radius:6px;background:#000;margin-bottom:20px}}
.row{{border:1px solid #2a3439;border-radius:6px;padding:12px 16px;margin-bottom:8px;cursor:pointer}}
.row:hover{{border-color:#54d6c6}}
.top{{display:flex;justify-content:space-between;font-family:monospace;font-size:12.5px;color:#8fa3ab}}
.kind{{color:#54d6c6;text-transform:uppercase;letter-spacing:.05em}}
.why{{margin-top:6px;font-size:14.5px}}
.facts{{margin-top:6px;font-family:monospace;font-size:11.5px;color:#75858d}}
.empty{{color:#8fa3ab;font-size:14px}}
</style></head>
<body><div class="wrap">
<h1>Alert Timeline</h1>
<div class="meta">{n_alerts} alerts &middot; {n_frames} frames sampled @ {eff_fps:.1f} fps &middot; {video}</div>
<video id="player" src="{video_src}" controls></video>
<div id="rows"></div>
<script>
const alerts = {alerts_json};
const video = document.getElementById('player');
const rows = document.getElementById('rows');
if (alerts.length === 0) {{
  rows.innerHTML = '<div class="empty">No alerts above threshold on this run.</div>';
}}
// #14: build DOM nodes with textContent, not innerHTML string concatenation.
// `why` and `facts` come from the VLM's own output - a hallucinated or
// adversarially-prompted verdict containing "<img onerror=...>" would
// otherwise execute in this page. textContent never interprets its input
// as markup, so this is safe regardless of what the VLM ever returns.
function el(tag, cls, text) {{
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}}

alerts.forEach((a) => {{
  const div = el('div', 'row');
  const top = el('div', 'top');
  top.appendChild(el('span', 'kind', a.kind));
  top.appendChild(el('span', '',
    a.t_start.toFixed(1) + 's - ' + a.t_end.toFixed(1) + 's · score ' + a.score.toFixed(2)));
  div.appendChild(top);

  const why = a.why ? a.why : '(no VLM verdict - geometric score only)';
  div.appendChild(el('div', 'why', why));

  const facts = Object.entries(a.facts || {{}}).map(([k, v]) => k + '=' + JSON.stringify(v)).join('  ');
  div.appendChild(el('div', 'facts', facts));

  div.onclick = () => {{ video.currentTime = a.t_start; video.play(); }};
  rows.appendChild(div);
}});
</script>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="out/alerts.json")
    ap.add_argument("--out", default="out/demo.html")
    a = ap.parse_args()

    data = json.load(open(a.alerts, encoding="utf-8"))
    alerts = data["alerts"]
    video_path = data["config"]["video"]["path"]
    out_dir = os.path.dirname(a.out) or "."
    video_src = os.path.relpath(video_path, start=out_dir)

    page = TEMPLATE.format(
        n_alerts=len(alerts), n_frames=data["n_frames"], eff_fps=data["eff_fps"],
        video=html.escape(video_path), video_src=html.escape(video_src),
        alerts_json=json.dumps(alerts))
    os.makedirs(out_dir, exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(page)
    print(f"[demo] wrote {a.out} - open in a browser, click a row to play the evidence clip")


if __name__ == "__main__":
    main()
