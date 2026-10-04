"""
Tests for the PatchCore engine that do NOT need PyTorch: a tiny NumPy
feature extractor stands in for the CNN so the memory-bank, scoring,
threshold and API-integration logic can be verified anywhere.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.services import patchcore as pc


def fake_feature_fn(images):
    """Stand-in for the CNN: per-8x8-block colour mean + std -> (N,32,32,6)."""
    out = []
    for img in images:
        g = img.shape[0] // 8
        blocks = img.reshape(g, 8, g, 8, 3)
        out.append(np.concatenate([blocks.mean(axis=(1, 3)), blocks.std(axis=(1, 3))], axis=-1))
    return np.stack(out).astype(np.float32)


def quantize(img):
    """Same 8-bit rounding an image goes through when saved as PNG and re-read."""
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8).astype(np.float32) / 255.0


def safe_threshold(val_scores):
    """Test-only threshold with a wide margin: synthetic images are almost noise-free,
    so a tight threshold would flip on tiny numeric differences between machines."""
    return 2.0 * max(val_scores)


def make_good(seed):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 0.5, dtype=np.float32)
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return quantize(img)


def make_defect(seed):
    img = make_good(seed)
    img[100:116, 120:160, :] = 0.95   # bright blob
    return quantize(img)


class TestAuroc(unittest.TestCase):
    def test_perfect_and_random_and_inverted(self):
        self.assertEqual(pc.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(pc.auroc([0, 0, 1, 1], [0.8, 0.9, 0.1, 0.2]), 0.0)
        self.assertAlmostEqual(pc.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_matches_pairwise_definition(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 2, 60)
        scores = np.round(rng.random(60), 1)  # many ties
        pos, neg = scores[labels == 1], scores[labels == 0]
        expected = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
        self.assertAlmostEqual(pc.auroc(labels, scores), expected, places=9)


class TestCoreset(unittest.TestCase):
    def test_size_and_coverage(self):
        rng = np.random.default_rng(0)
        # three well-separated clusters; a 3% coreset must still touch all of them
        feats = np.concatenate([rng.normal(c, 0.1, (500, 8)) for c in (0, 10, 20)]).astype(np.float32)
        idx = pc.greedy_coreset(feats, 0.03)
        self.assertEqual(len(idx), 45)
        self.assertEqual(len(set(idx.tolist())), 45)
        self.assertEqual({int(i // 500) for i in idx}, {0, 1, 2})


class TestScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = pc.build_bank([make_good(i) for i in range(8)], fake_feature_fn, ratio=0.2)

    def test_defect_scores_higher_than_good(self):
        good = max(pc.score_image(make_good(100 + i), self.bank, fake_feature_fn)[0] for i in range(5))
        defect, amap = pc.score_image(make_defect(200), self.bank, fake_feature_fn)
        self.assertGreater(defect, good * 3)
        y, x = np.unravel_index(np.argmax(amap), amap.shape)   # peak is on the blob
        self.assertTrue(95 <= y <= 120 and 115 <= x <= 165, (y, x))

    def test_threshold_rule(self):
        t = pc.choose_threshold([1.0, 1.1, 0.9, 1.2])
        self.assertGreaterEqual(t, 1.2)


class TestInputSize(unittest.TestCase):
    def test_larger_input_size_gives_matching_map(self):
        big = np.full((320, 320, 3), 0.5, dtype=np.float32)
        bank = pc.build_bank([big], fake_feature_fn, ratio=0.5)
        score, amap = pc.score_image(big, bank, fake_feature_fn)
        self.assertEqual(amap.shape, (320, 320))


class TestEndToEnd(unittest.TestCase):
    """predict_defect() with a saved model, using the stand-in extractor."""

    def test_predict_uses_patchcore_and_flags_defect(self):
        from app.config import settings
        from app.services import defect_detection

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pcdir = tmp / "pc"
            pcdir.mkdir()
            bank = pc.build_bank([make_good(i) for i in range(8)], fake_feature_fn, ratio=0.2)
            val = [pc.score_image(make_good(50 + i), bank.astype(np.float32), fake_feature_fn)[0] for i in range(12)]
            np.save(pcdir / "widget_bank.npy", bank)
            (pcdir / "widget_meta.json").write_text(json.dumps({"threshold": safe_threshold(val)}))
            (pcdir / pc.BACKBONE_FILE).write_bytes(b"x")  # presence check only

            uploads = tmp / "uploads"
            uploads.mkdir()
            good_path, bad_path = str(tmp / "good.png"), str(tmp / "bad.png")
            cv2.imwrite(good_path, cv2.cvtColor((make_good(999) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(bad_path, cv2.cvtColor((make_defect(999) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            old = (settings.PATCHCORE_DIR, settings.UPLOAD_DIR, pc.get_feature_fn)
            settings.PATCHCORE_DIR, settings.UPLOAD_DIR = str(pcdir), str(uploads)
            pc.get_feature_fn = lambda _d: fake_feature_fn
            pc._bank_cache.clear()
            try:
                good = defect_detection.predict_defect(good_path, "MVTec - widget")
                bad = defect_detection.predict_defect(bad_path, "MVTec - widget")
            finally:
                settings.PATCHCORE_DIR, settings.UPLOAD_DIR, pc.get_feature_fn = old
                pc._bank_cache.clear()

            self.assertEqual(good["model_used"], "patchcore:widget")
            self.assertEqual(good["status"], "pass")
            self.assertEqual(bad["status"], "fail")
            self.assertTrue(bad["bounding_boxes"])
            self.assertTrue((uploads / bad["heatmap_filename"]).exists())

    def test_unknown_category_falls_back(self):
        from app.services import defect_detection
        self.assertEqual(defect_detection.get_category_key("Some Random Part"), "generic")


if __name__ == "__main__":
    unittest.main()