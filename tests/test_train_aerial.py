"""Pure-logic tests for train/train_aerial.py and scripts/make_night_yaml.py.

Neither module imports anything heavier than argparse/pathlib at module
scope (train_aerial.py imports ultralytics and huggingface_hub lazily,
inside function bodies) so, like tests/test_pipeline.py, this runs
anywhere including CI (see .github/workflows/ci.yml). make_night.py's
tests live separately in test_make_night.py because it imports cv2 at
module scope, which CI does not install.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.train_aerial import PRESETS, _resolve_pretrained, _lr0_for, PRETRAINED
from scripts.make_night_yaml import NAMES as VISDRONE_NAMES


# ------------------------------------------------------------- train_aerial --

def test_presets_have_sane_ranges():
    for name, (model, imgsz, epochs, batch, freeze) in PRESETS.items():
        assert model.endswith(".pt"), name
        assert 0 < imgsz <= 2048, name
        assert 0 < epochs <= 200, name
        assert 0 < batch <= 64, name
        assert 0 <= freeze <= 24, name


def test_lr0_lower_for_two_stage_than_stock():
    # continuing from an already-converged VisDrone checkpoint must use a
    # lower lr0 than training up from stock COCO weights, or the full
    # unfrozen fine-tune risks unlearning the aerial adaptation
    assert _lr0_for(two_stage=True) < _lr0_for(two_stage=False)


def test_resolve_pretrained_falls_back_to_stock_on_download_failure(monkeypatch):
    class _FakeHFHub:
        @staticmethod
        def hf_hub_download(repo_id, filename):
            raise OSError("offline")

    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHFHub)
    base, two_stage = _resolve_pretrained("yolo11s.pt")
    assert base == "yolo11s.pt"
    assert two_stage is False


def test_resolve_pretrained_uses_checkpoint_on_success(monkeypatch, tmp_path):
    fake_path = str(tmp_path / "best.pt")

    class _FakeHFHub:
        @staticmethod
        def hf_hub_download(repo_id, filename):
            assert (repo_id, filename) == PRETRAINED["yolo11n.pt"]
            return fake_path

    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHFHub)
    base, two_stage = _resolve_pretrained("yolo11n.pt")
    assert base == fake_path
    assert two_stage is True


def test_resolve_pretrained_unknown_model_skips_lookup():
    base, two_stage = _resolve_pretrained("yolo11m.pt")
    assert base == "yolo11m.pt"
    assert two_stage is False


# ---------------------------------------------------------- make_night_yaml --

def test_make_night_yaml_train_split_is_a_broken_placeholder(tmp_path, monkeypatch):
    import scripts.make_night_yaml as make_night_yaml

    images_dir = tmp_path / "datasets" / "VisDrone" / "images" / "train"
    images_dir.mkdir(parents=True)
    (images_dir / "img0_night.jpg").write_bytes(b"")
    (images_dir / "img1_night.jpg").write_bytes(b"")
    (images_dir / "img2.jpg").write_bytes(b"")  # not a night twin - excluded

    out_yaml = tmp_path / "VisDroneNight.yaml"
    list_file = tmp_path / "night_val.txt"
    argv = ["make_night_yaml.py", "--images", str(images_dir),
            "--out", str(out_yaml), "--list", str(list_file)]
    monkeypatch.setattr(sys, "argv", argv)
    make_night_yaml.main()

    text = out_yaml.read_text(encoding="utf-8")
    # #13: `train:` must never point at a real, loadable split - it exists
    # only so `yolo val` works and `yolo train` against this file fails loudly
    assert "train: DO_NOT_TRAIN_ON_THIS_FILE_val_only.txt" in text
    assert f"val: {list_file}" in text
    assert len(list_file.read_text(encoding="utf-8").splitlines()) == 2
    for i, n in enumerate(VISDRONE_NAMES):
        assert f"{i}: {n}" in text
