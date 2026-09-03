"""Level 3: ask the footage a question in words.

Reuses the SAME normal-bank embeddings fit.py already built - no extra model,
no extra pass over the video.

    python query.py "person carrying a bag near the gate"
    python query.py "vehicle stopped on the footpath" --top-k 3
"""
import argparse
import yaml
from pipeline.retrieve import search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fit-json", default="out/scene_fit.json")
    ap.add_argument("--bank-npy", default="out/normal_bank.npy")
    ap.add_argument("--top-k", type=int, default=None,
                     help="default: retrieve.top_k from config.yaml")
    a = ap.parse_args()

    if a.top_k is None:
        cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
        a.top_k = cfg.get("retrieve", {}).get("top_k", 5)

    r = search(a.query, fit_json=a.fit_json, bank_npy=a.bank_npy, top_k=a.top_k)
    if r["reason"]:
        print("refused:", r["reason"])
        return
    for h in r["results"]:
        secs = h.get("approx_seconds")
        when = f"{secs:>7.1f}s" if secs is not None else f"frame {h['frame_index']:>7}"
        print(f"  #{h['rank']}  {when}  sim={h['similarity']:.3f}")


if __name__ == "__main__":
    main()
