"""Render docs/DESIGN.md into a styled standalone HTML page.

    python docs/build_page.py      # from the repo root -> out/design.html

Markdown to HTML at build time, so the page is static: no client-side
markdown, and mermaid fences become <pre class="mermaid"> blocks, which the
artifact runtime renders natively. Stdlib only - nothing to install.
"""
import html as H
import re
import os

SRC = "docs/DESIGN.md"
OUT = "out/design.html"

CHIP = {"TARGET": "chip-target", "MEASURED": "chip-measured",
        "DERIVED": "chip-derived"}


def slug(text):
    t = re.sub(r"<[^>]+>", "", text)
    t = t.replace("§", "").replace("—", "-")
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    return t or "s"


def inline(text):
    """Escape, then apply code spans, bold, italic, links. Code spans are
    tokenised first so nothing inside them is reinterpreted."""
    spans = []

    def stash(m):
        spans.append(m.group(1))
        return "\x00%d\x00" % (len(spans) - 1)

    text = re.sub(r"`([^`]+)`", stash, text)
    text = H.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: '<a href="%s">%s</a>' % (m.group(2), m.group(1)), text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)

    def restore(m):
        raw = spans[int(m.group(1))]
        cls = CHIP.get(raw.strip())
        esc = H.escape(raw, quote=False)
        if cls:
            return '<span class="chip %s">%s</span>' % (cls, esc)
        return "<code>%s</code>" % esc

    return re.sub(r"\x00(\d+)\x00", restore, text)


def split_row(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    out, cur, i = [], "", 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur += "|"
            i += 2
            continue
        if c == "|":
            out.append(cur)
            cur = ""
            i += 1
            continue
        cur += c
        i += 1
    out.append(cur)
    return [c.strip() for c in out]


def build(md):
    lines = md.split("\n")
    out, nav = [], []
    i, n = 0, len(lines)

    def close_list(stack):
        while stack:
            out.append("</%s>" % stack.pop())

    list_stack = []

    while i < n:
        line = lines[i]

        # fenced code
        if line.startswith("```"):
            lang = line[3:].strip()
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            close_list(list_stack)
            body = H.escape("\n".join(buf), quote=False)
            if lang == "mermaid":
                out.append('<figure class="diagram"><pre class="mermaid">%s</pre></figure>' % body)
            else:
                label = lang if lang else "text"
                out.append('<div class="codewrap"><span class="codelang">%s</span>'
                           '<pre class="code"><code>%s</code></pre></div>' % (label, body))
            continue

        # table
        if line.startswith("|") and i + 1 < n and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1]):
            close_list(list_stack)
            head = split_row(line)
            i += 2
            rows = []
            while i < n and lines[i].startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            t = ['<div class="tablewrap"><table><thead><tr>']
            for c in head:
                t.append("<th>%s</th>" % inline(c))
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>")
                for c in r:
                    t.append("<td>%s</td>" % inline(c))
                t.append("</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            close_list(list_stack)
            level = len(m.group(1))
            text = inline(m.group(2))
            sid = slug(m.group(2))
            if level == 1:
                if i == 0:
                    out.append('<h1 id="%s">%s</h1>' % (sid, text))
                else:
                    nav.append(("part", m.group(2), sid))
                    out.append('<h2 class="part" id="%s">%s</h2>' % (sid, text))
            elif level == 2:
                nav.append(("sec", m.group(2), sid))
                out.append('<h3 id="%s">%s</h3>' % (sid, text))
            elif level == 3:
                out.append('<h4 id="%s">%s</h4>' % (sid, text))
            else:
                out.append('<h5 id="%s">%s</h5>' % (sid, text))
            i += 1
            continue

        # hr
        if re.match(r"^-{3,}\s*$", line):
            close_list(list_stack)
            out.append('<hr/>')
            i += 1
            continue

        # blockquote
        if line.startswith(">"):
            close_list(list_stack)
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i].lstrip(">").strip())
                i += 1
            out.append("<blockquote><p>%s</p></blockquote>" % inline(" ".join(buf)))
            continue

        # lists
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            indent = len(m.group(1))
            ordered = m.group(2)[0].isdigit()
            tag = "ol" if ordered else "ul"
            depth = 1 + indent // 2
            while len(list_stack) > depth:
                out.append("</%s>" % list_stack.pop())
            if len(list_stack) < depth:
                out.append('<%s>' % tag)
                list_stack.append(tag)
            buf = [m.group(3)]
            i += 1
            while i < n and lines[i].strip() and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]) \
                    and not lines[i].startswith(("#", "|", "```", ">")) \
                    and lines[i].startswith(" "):
                buf.append(lines[i].strip())
                i += 1
            out.append("<li>%s</li>" % inline(" ".join(buf)))
            continue

        if not line.strip():
            close_list(list_stack)
            i += 1
            continue

        # paragraph
        close_list(list_stack)
        buf = []
        while i < n and lines[i].strip() and not lines[i].startswith(("#", "|", "```", ">", "---")) \
                and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        out.append("<p>%s</p>" % inline(" ".join(buf)))

    close_list(list_stack)
    return "\n".join(out), nav


def nav_html(nav):
    parts = []
    for kind, text, sid in nav:
        label = re.sub(r"^#+\s*", "", text)
        if kind == "part":
            label = label.replace("Part ", "").split("—")[-1].strip()
            parts.append('<a class="nav-part" href="#%s">%s</a>' % (sid, H.escape(label)))
        else:
            m = re.match(r"^(\d+)\.\s*(.*)$", label)
            if m:
                num, rest = m.group(1), m.group(2)
            else:
                num, rest = "", label
            rest = rest.split("—")[0].strip()
            parts.append('<a class="nav-sec" href="#%s"><span class="nav-num">%s</span>'
                         '<span>%s</span></a>' % (sid, num, H.escape(rest)))
    return "\n".join(parts)


CSS = """
:root{
  --ground:#eef2f4; --surface:#ffffff; --surface-2:#e4eaed; --sunken:#e9eef0;
  --ink:#111a20; --ink-2:#3c4d57; --ink-3:#687c86; --ink-4:#8b9ca4;
  --rule:#c9d4d9; --rule-2:#dce4e8;
  --accent:#a75f13; --accent-2:#8c4f10; --accent-soft:#f7e8d5;
  --verify:#0c6963; --verify-soft:#d6eae8;
  --risk:#9d3527; --risk-soft:#f6dfdb;
  --code-bg:#e9eef1; --code-ink:#1d2b33;
  --shadow:0 1px 2px rgba(17,26,32,.06);
  --f-disp:"Chivo","Helvetica Neue",Arial,sans-serif;
  --f-body:"Source Serif 4",Georgia,"Times New Roman",serif;
  --f-mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0d1418; --surface:#131c22; --surface-2:#1a252c; --sunken:#101a1f;
    --ink:#e5edf0; --ink-2:#aebdc5; --ink-3:#83959e; --ink-4:#6a7c85;
    --rule:#28373f; --rule-2:#1f2c33;
    --accent:#e2a460; --accent-2:#f0b878; --accent-soft:#382718;
    --verify:#5cbcb3; --verify-soft:#11302e;
    --risk:#e28a7e; --risk-soft:#331915;
    --code-bg:#0b1317; --code-ink:#cfdde3;
    --shadow:0 1px 2px rgba(0,0,0,.4);
  }
}
:root[data-theme="dark"]{
  --ground:#0d1418; --surface:#131c22; --surface-2:#1a252c; --sunken:#101a1f;
  --ink:#e5edf0; --ink-2:#aebdc5; --ink-3:#83959e; --ink-4:#6a7c85;
  --rule:#28373f; --rule-2:#1f2c33;
  --accent:#e2a460; --accent-2:#f0b878; --accent-soft:#382718;
  --verify:#5cbcb3; --verify-soft:#11302e;
  --risk:#e28a7e; --risk-soft:#331915;
  --code-bg:#0b1317; --code-ink:#cfdde3;
  --shadow:0 1px 2px rgba(0,0,0,.4);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--f-body); font-size:16.5px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
a{color:var(--accent-2); text-decoration:none; border-bottom:1px solid var(--rule)}
a:hover{border-bottom-color:var(--accent)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}

/* ---------- masthead ---------- */
.masthead{
  border-bottom:1px solid var(--rule); background:var(--surface);
}
.mast-in{max-width:1180px; margin:0 auto; padding:38px 32px 0}
.eyebrow{
  font-family:var(--f-mono); font-size:11px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-3); margin:0 0 14px;
}
h1{
  font-family:var(--f-disp); font-weight:900; font-size:clamp(30px,4.6vw,50px);
  line-height:1.02; letter-spacing:-.022em; margin:0 0 6px; text-wrap:balance;
  max-width:22ch;
}
.subtitle{
  font-family:var(--f-disp); font-weight:600; font-size:clamp(15px,1.7vw,18px);
  color:var(--ink-2); margin:0 0 26px; letter-spacing:-.005em;
}
.thesis{
  font-size:19px; line-height:1.5; max-width:60ch; color:var(--ink);
  border-left:3px solid var(--accent); padding:2px 0 2px 18px; margin:0 0 28px;
}
.thesis em{font-style:italic; color:var(--ink-2)}
.metastrip{
  display:flex; flex-wrap:wrap; gap:0; border-top:1px solid var(--rule-2);
  margin-bottom:0;
}
.metacell{
  flex:1 1 150px; padding:13px 18px 15px; border-right:1px solid var(--rule-2);
}
.metacell:last-child{border-right:0}
.metacell dt{
  font-family:var(--f-mono); font-size:10px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-4); margin:0 0 4px;
}
.metacell dd{
  margin:0; font-family:var(--f-disp); font-weight:600; font-size:14.5px;
  color:var(--ink); font-variant-numeric:tabular-nums;
}
/* the pipeline's own stage trace, as the masthead rule */
.trace{
  display:flex; gap:0; overflow-x:auto; border-top:1px solid var(--rule);
  background:var(--sunken); font-family:var(--f-mono); font-size:10.5px;
  scrollbar-width:thin;
}
.trace span{
  white-space:nowrap; padding:9px 14px; color:var(--ink-3);
  border-right:1px solid var(--rule-2);
}
.trace span b{color:var(--accent); font-weight:500}

/* ---------- shell ---------- */
.shell{
  max-width:1180px; margin:0 auto; padding:0 32px 90px;
  display:grid; grid-template-columns:212px minmax(0,1fr); gap:52px;
  align-items:start;
}
nav.rail{
  position:sticky; top:0; max-height:100vh; overflow-y:auto;
  padding:34px 0 40px; scrollbar-width:thin;
}
.rail-title{
  font-family:var(--f-mono); font-size:10px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--ink-4);
  padding-bottom:10px; border-bottom:1px solid var(--rule); margin-bottom:12px;
}
.nav-part{
  display:block; font-family:var(--f-disp); font-weight:700; font-size:11px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3);
  margin:20px 0 8px; border:0;
}
.nav-part:hover{color:var(--accent)}
.nav-sec{
  display:grid; grid-template-columns:22px 1fr; gap:6px; border:0;
  font-family:var(--f-disp); font-weight:500; font-size:13px; line-height:1.35;
  color:var(--ink-2); padding:4px 0;
}
.nav-sec:hover{color:var(--accent)}
.nav-num{
  font-family:var(--f-mono); font-size:11px; color:var(--ink-4);
  font-variant-numeric:tabular-nums;
}

main{padding-top:34px; min-width:0}
main > p, main > ul, main > ol, main > blockquote, main > h4, main > h5{max-width:68ch}

/* ---------- headings ---------- */
h2.part{
  font-family:var(--f-disp); font-weight:900; font-size:13px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--accent); margin:64px 0 6px;
  padding-bottom:12px; border-bottom:2px solid var(--accent);
}
h3{
  font-family:var(--f-disp); font-weight:700; font-size:26px; line-height:1.14;
  letter-spacing:-.016em; margin:52px 0 16px; text-wrap:balance; color:var(--ink);
  padding-top:6px;
}
h4{
  font-family:var(--f-disp); font-weight:700; font-size:17.5px; line-height:1.25;
  letter-spacing:-.008em; margin:38px 0 12px; color:var(--ink); text-wrap:balance;
}
h5{
  font-family:var(--f-disp); font-weight:600; font-size:14px; margin:26px 0 10px;
  color:var(--ink-2); letter-spacing:.01em;
}
p{margin:0 0 16px}
hr{border:0; border-top:1px solid var(--rule-2); margin:44px 0}
strong{font-weight:600; color:var(--ink)}
em{font-style:italic}

ul,ol{margin:0 0 18px; padding-left:22px}
li{margin-bottom:8px}
li::marker{color:var(--ink-4)}

blockquote{
  margin:26px 0; padding:18px 22px; background:var(--surface);
  border-left:3px solid var(--verify); border-radius:0 3px 3px 0;
  box-shadow:var(--shadow);
}
blockquote p{margin:0; font-size:17.5px; line-height:1.55; color:var(--ink)}

/* ---------- code ---------- */
code{
  font-family:var(--f-mono); font-size:.855em; background:var(--code-bg);
  color:var(--code-ink); padding:1.5px 5px; border-radius:2px;
  word-break:break-word;
}
.codewrap{position:relative; margin:24px 0}
.codelang{
  position:absolute; top:0; right:0; font-family:var(--f-mono); font-size:9.5px;
  letter-spacing:.14em; text-transform:uppercase; color:var(--ink-4);
  background:var(--surface-2); padding:4px 9px; border-radius:0 3px 0 3px;
}
pre.code{
  margin:0; background:var(--code-bg); border:1px solid var(--rule-2);
  border-radius:3px; padding:18px 20px; overflow-x:auto;
  font-family:var(--f-mono); font-size:12.5px; line-height:1.62;
  color:var(--code-ink); scrollbar-width:thin;
}
pre.code code{background:none; padding:0; font-size:inherit; color:inherit}

/* ---------- diagrams ---------- */
figure.diagram{
  margin:30px 0; padding:22px 18px; background:var(--surface);
  border:1px solid var(--rule-2); border-radius:3px; overflow-x:auto;
  text-align:center; scrollbar-width:thin;
}
figure.diagram pre.mermaid{margin:0; background:none; border:0; text-align:center}

/* ---------- tables ---------- */
.tablewrap{
  margin:26px 0; overflow-x:auto; border:1px solid var(--rule-2);
  border-radius:3px; background:var(--surface); box-shadow:var(--shadow);
  scrollbar-width:thin;
}
table{border-collapse:collapse; width:100%; font-family:var(--f-disp); font-size:13.5px}
thead th{
  text-align:left; font-weight:700; font-size:10.5px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--ink-3); background:var(--surface-2);
  padding:11px 14px; border-bottom:1px solid var(--rule); white-space:nowrap;
}
tbody td{
  padding:11px 14px; border-bottom:1px solid var(--rule-2); vertical-align:top;
  line-height:1.5; color:var(--ink-2);
}
tbody tr:last-child td{border-bottom:0}
tbody td:first-child{color:var(--ink); font-weight:500}
tbody td code{font-size:11.8px}
tbody tr:hover td{background:var(--sunken)}

/* ---------- claim chips ---------- */
.chip{
  display:inline-block; font-family:var(--f-mono); font-size:10px;
  letter-spacing:.1em; text-transform:uppercase; padding:2px 7px;
  border-radius:2px; white-space:nowrap; vertical-align:1px; font-weight:500;
}
.chip-target{background:var(--accent-soft); color:var(--accent)}
.chip-measured{background:var(--verify-soft); color:var(--verify)}
.chip-derived{background:var(--surface-2); color:var(--ink-3)}

/* ---------- footer ---------- */
footer{
  max-width:1180px; margin:0 auto; padding:26px 32px 60px;
  border-top:1px solid var(--rule); font-family:var(--f-mono);
  font-size:11px; letter-spacing:.06em; color:var(--ink-4);
  display:flex; flex-wrap:wrap; gap:8px 26px;
}

@media (max-width:940px){
  .shell{grid-template-columns:1fr; gap:0; padding:0 22px 70px}
  nav.rail{
    position:static; max-height:none; padding:26px 0 8px;
    border-bottom:1px solid var(--rule); margin-bottom:8px;
  }
  .rail-inner{
    display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:2px 18px;
  }
  .nav-part{margin:14px 0 4px; grid-column:1/-1}
  .mast-in{padding:30px 22px 0}
  footer{padding:24px 22px 50px}
  h3{font-size:22px; margin-top:42px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""

TRACE = [
    ("[0]", "config &middot; scene fit"),
    ("[1-2]", "detect &middot; track"),
    ("[2b]", "re-id"),
    ("[3]", "candidate events"),
    ("[3b]", "open-vocab"),
    ("[4]", "vlm judge"),
    ("[5]", "fuse &middot; suppress"),
    ("[5b]", "first alert"),
    ("[6]", "alerts.json"),
    ("[7]", "fps sustained"),
]

META = [
    ("Version", "1.0"),
    ("Date", "4 Sept 2026"),
    ("Codebase", "2,224 LOC &middot; 24 modules"),
    ("Requirements", "21 FR &middot; 38 NFR"),
    ("Open gaps", "10, enumerated"),
    ("Verified", "0 &mdash; nothing has run"),
]


def main():
    md = open(SRC, encoding="utf-8").read()
    # the h1 and the intro table are replaced by the designed masthead
    body_md = md.split("### How to read this document", 1)[1]
    body_md = "### How to read this document" + body_md
    body, nav = build(body_md)

    trace = "".join('<span><b>%s</b> %s</span>' % (k, v) for k, v in TRACE)
    meta = "".join('<div class="metacell"><dt>%s</dt><dd>%s</dd></div>' % (k, v)
                   for k, v in META)

    page = """<title>Aerial Anomaly Intelligence</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chivo:wght@500;600;700;900&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap"/>
<style>%s</style>

<header class="masthead">
  <div class="mast-in">
    <p class="eyebrow">Design of record &middot; HLD + LLD &middot; FlytBase Visual Intelligence Hackathon</p>
    <h1>Aerial Anomaly Intelligence</h1>
    <p class="subtitle">Small-VLM anomaly detection on long-form drone and CCTV video</p>
    <p class="thesis">A small VLM cannot watch every frame, so it is the judge at the end of a
      cheap cascade &mdash; <em>eight calls instead of eighteen thousand.</em></p>
  </div>
  <div class="mast-in" style="padding-bottom:0">
    <dl class="metastrip">%s</dl>
  </div>
  <div class="trace">%s</div>
</header>

<div class="shell">
  <nav class="rail">
    <div class="rail-title">Contents</div>
    <div class="rail-inner">%s</div>
  </nav>
  <main>%s</main>
</div>

<footer>
  <span>docs/DESIGN.md</span>
  <span>flytbase-prep</span>
  <span>Every NFR target is a budget, not a measurement</span>
</footer>
""" % (CSS, meta, trace, nav_html(nav), body)

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(page)
    print("wrote", OUT, len(page), "bytes")
    print("nav entries:", len(nav))
    print("tables:", body.count("<table>"), "diagrams:", body.count('class="mermaid"'),
          "code blocks:", body.count("pre class=\"code\""))
    print("chips:", body.count('class="chip'))


if __name__ == "__main__":
    main()
