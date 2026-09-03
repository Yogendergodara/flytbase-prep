"""Tests for train/make_night.py.

Skipped in CI (see .github/workflows/ci.yml) because it imports cv2 at
module scope and CI only installs numpy/pytest/pyyaml - run these locally
or on Kaggle where opencv-python is installed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

cv2 = pytest.importorskip("cv2")
from train.make_night import to_night  # noqa: E402  (needs cv2, see module docstring)


def _fake_image(rng, w=32, h=32):
    return rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)


def test_to_night_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = _fake_image(rng)
    out = to_night(img, rng)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_to_night_darkens_on_average():
    rng = np.random.default_rng(0)
    img = _fake_image(rng)
    out = to_night(img, np.random.default_rng(0))
    assert out.astype(np.float32).mean() < img.astype(np.float32).mean()


def _write_visdrone_pair(images_dir, labels_dir, stem, rng):
    img = _fake_image(rng)
    cv2.imwrite(str(images_dir / f"{stem}.jpg"), img)
    (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")


def _run_make_night(monkeypatch, images_dir, labels_dir, fraction=0.5, seed=0):
    import importlib
    import train.make_night as make_night
    importlib.reload(make_night)
    argv = ["make_night.py", "--images", str(images_dir), "--labels", str(labels_dir),
            "--fraction", str(fraction), "--seed", str(seed)]
    monkeypatch.setattr(sys, "argv", argv)
    make_night.main()
    return make_night


def test_make_night_creates_expected_fraction(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(); labels_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        _write_visdrone_pair(images_dir, labels_dir, f"img{i}", rng)

    _run_make_night(monkeypatch, images_dir, labels_dir, fraction=0.4)

    night_imgs = sorted(images_dir.glob("*_night.jpg"))
    night_labels = sorted(labels_dir.glob("*_night.txt"))
    assert len(night_imgs) == 4
    assert len(night_labels) == 4


def test_make_night_is_idempotent_on_rerun(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(); labels_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(10):
        _write_visdrone_pair(images_dir, labels_dir, f"img{i}", rng)

    _run_make_night(monkeypatch, images_dir, labels_dir, fraction=0.4)
    first_run_images = sorted(p.name for p in images_dir.glob("*_night.jpg"))

    # re-running (e.g. an interrupted-session retry) must not double-darken
    # its own output, and must not count synthetic frames toward the pool
    # that --fraction is computed against
    _run_make_night(monkeypatch, images_dir, labels_dir, fraction=0.4)
    second_run_images = sorted(p.name for p in images_dir.glob("*_night.jpg"))

    assert second_run_images == first_run_images
    assert not any("_night_night" in p.name for p in images_dir.iterdir())


def test_make_night_skips_images_with_no_label(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(); labels_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(5):
        _write_visdrone_pair(images_dir, labels_dir, f"img{i}", rng)
    # an image with no label file at all - to_night must never be asked to
    # invent labels for a synthetic twin
    cv2.imwrite(str(images_dir / "unlabeled.jpg"), _fake_image(rng))

    _run_make_night(monkeypatch, images_dir, labels_dir, fraction=1.0)

    assert not (images_dir / "unlabeled_night.jpg").exists()
