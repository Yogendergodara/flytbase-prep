# Pre-flight checklist — do this the night before (Phase 19)

Organizers explicitly ask teams to arrive with setup and model access already
tested. This is procedure, not code — check items off in order, the night
before, not Saturday morning.

- [ ] `git push` everything. Kaggle re-clones every session — uncommitted
      local edits do not exist there (`IMPLEMENTATION.md` Part 0).
- [ ] Cache every model weight so nothing depends on venue wifi:
  - [ ] `yolo11s.pt` / fine-tuned `weights/aerial_night/weights/best.pt`
  - [ ] `Qwen/Qwen2.5-VL-3B-Instruct`
  - [ ] `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (Phase 16 fallback)
  - [ ] `yolov8l-worldv2.pt` — only if `open_vocab.backend=yoloworld` is planned
  - [ ] OSNet re-ID weights — only if `reid.backend=osnet` is planned
  - [ ] the distilled LoRA adapter directory — only if Phase 18 produced one
        and won its A/B
- [ ] Confirm HF auth token works from a fresh shell. If Phase 18 was
      attempted, confirm `TEACHER_API_BASE`/`TEACHER_API_KEY`/`TEACHER_MODEL`
      still work too — don't discover an expired key at 9:05 AM.
- [ ] Run all four presets once each from a cold shell:
      `python run.py --video <clip> --preset day`
      `python run.py --video <clip> --preset night`
      `python run.py --video <clip> --preset fast`
      `python run.py --video <clip> --preset accurate`
      (this is `IMPLEMENTATION.md` Phase 13 step 1 — do it tonight, not as
      part of Saturday morning)
- [ ] Run `python scripts/compare_backends.py --video <clip> --cpu-fallback`
      (Phase 16) — confirm the SmolVLM2 fallback and the forced-CPU path
      both actually produce `alerts.json`.
- [ ] Run `python scripts/economics.py --alerts out/alerts.json` (Phase 14) —
      confirm the three headline numbers print cleanly from a real run.
- [ ] **Final check — disconnect wifi, run one preset end to end.** Anything
      that fails here is a silent network dependency you didn't know about.
      Fix it before the day, not during it.

Every box above needs a checkmark from *last night*, not from memory. If the
wifi-off check fails, that is the single most valuable bug you'll find all
week — it fails silently in a live room otherwise.
