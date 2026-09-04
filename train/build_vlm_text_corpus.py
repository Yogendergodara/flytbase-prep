"""Build Pool 2 (DATASET_PLAN.md): a small curated text corpus for VLM judge
few-shot prompting (and, if Phase 18 is attempted, distillation validation).

Deliberately small - see DATASET_PLAN.md's Pool 2 section. This is NOT a
bulk fine-tuning corpus; it seeds `pipeline/vlm_judge.PROMPT`.

    python train/build_vlm_text_corpus.py \\
        --a2seek datasets/A2Seek/annotations.jsonl \\
        --uca datasets/UCA/captions.json \\
        --tar datasets/NVIDIA_TAR/tar.jsonl \\
        --cuva datasets/CUVA/cuva.json \\
        --out train/vlm_text_corpus.jsonl

UNVERIFIED ASSUMPTIONS: each source is read as JSON or JSONL (whichever the
file extension implies) with per-entry keys drawn from
{video, video_id, clip, source_video} / {caption, text, description} /
{reasoning, rationale, chain_of_thought} / {start, t_start} / {end, t_end}.
Any source whose actual schema doesn't overlap this set will yield 0 rows -
the printed per-source count is the check; a 0 means go look at that file,
don't assume the sample size was just small.
"""
import argparse
import json
import random
from pathlib import Path

FIELD_ALIASES = {
    "video_ref": ("video", "video_id", "clip", "source_video", "video_name"),
    "caption": ("caption", "text", "description", "sentence"),
    "reasoning": ("reasoning", "rationale", "chain_of_thought", "cot",
                  "cause", "consequence", "cause_consequence", "why"),
    "t_start": ("t_start", "start", "start_time"),
    "t_end": ("t_end", "end", "end_time"),
}
NIGHT_KEYWORDS = ("night", "dark", "twilight", "dusk", "low-light", "low light")


def _get(entry, field):
    for alias in FIELD_ALIASES[field]:
        if alias in entry and entry[alias] not in (None, ""):
            return entry[alias]
    return None


def _load_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("annotations", "data", "items", "captions"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return list(data.values())
    return data


def normalize(entry, source):
    if not isinstance(entry, dict):
        return None  # a dict-of-lists source can hand us bare strings/lists
    caption = _get(entry, "caption")
    if isinstance(caption, (list, tuple)):
        caption = " ".join(str(c) for c in caption)  # some releases ship a list of sentences
    if not isinstance(caption, str) or not caption.strip():
        return None
    reasoning = _get(entry, "reasoning")
    return {
        "source": source,
        "video_ref": _get(entry, "video_ref"),
        "t_start": _get(entry, "t_start"),
        "t_end": _get(entry, "t_end"),
        "caption": caption.strip(),
        "reasoning": reasoning if isinstance(reasoning, str) else (
            " ".join(str(x) for x in reasoning) if isinstance(reasoning, (list, tuple)) else None),
    }


def sample_source(path, source, n, seed, weight_night=False):
    if not path:
        return []
    try:
        records = _load_records(path)
    except Exception as e:
        print(f"[vlm-corpus] {source}: could not read {path} ({e}), skipping")
        return []
    normalized = [r for r in (normalize(e, source) for e in records) if r]
    if not normalized:
        print(f"[vlm-corpus] {source}: 0 usable rows out of {len(records)} raw entries - "
              f"schema mismatch, check {path} by hand")
        return []
    rng = random.Random(seed)
    if weight_night:
        # partition in one pass - `r not in night` is a linear dict-compare
        # scan per row, which on A2Seek's 42k entries is ~10^8 comparisons
        night, rest = [], []
        for r in normalized:
            is_night = any(k in r["caption"].lower() for k in NIGHT_KEYWORDS)
            (night if is_night else rest).append(r)
        rng.shuffle(night)
        rng.shuffle(rest)
        picked = (night + rest)[:n]
        print(f"[vlm-corpus] {source}: {len(picked)} rows ({min(len(night), n)} night/twilight-weighted)")
    else:
        rng.shuffle(normalized)
        picked = normalized[:n]
        print(f"[vlm-corpus] {source}: {len(picked)} of {len(normalized)} usable rows")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2seek", default=None)
    ap.add_argument("--uca", default=None)
    ap.add_argument("--tar", default=None)
    ap.add_argument("--cuva", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="train/vlm_text_corpus.jsonl")
    a = ap.parse_args()

    rows = []
    rows += sample_source(a.a2seek, "A2Seek", 2000, a.seed, weight_night=True)
    rows += sample_source(a.uca, "UCA", 500, a.seed)
    rows += sample_source(a.tar, "NVIDIA_TAR", 800, a.seed)
    rows += sample_source(a.cuva, "CUVA", 370, a.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[vlm-corpus] {len(rows)} total rows -> {out}")

    candidates = sorted(
        (r for r in rows if r["source"] == "A2Seek" and r["reasoning"]),
        key=lambda r: len(r["caption"]), reverse=True,
    )[:5]
    print(f"[vlm-corpus] {len(candidates)} candidate few-shot examples (longest A2Seek "
          f"captions with reasoning) - hand-pick 3-5 of these into pipeline/vlm_judge.PROMPT:")
    for c in candidates:
        print(f"  - [{c['video_ref']}] {c['caption'][:100]}")


if __name__ == "__main__":
    main()
