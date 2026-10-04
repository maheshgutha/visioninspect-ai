"""
Tests for product recognition routing.  No PyTorch and no API key needed:
a NumPy feature extractor stands in for the CNN.

After removing AI vision, unsupported photos must raise RuntimeError with
a clear message listing the supported products.
"""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.services import defect_detection, identifier
from app.services import patchcore as pc
from tests.test_patchcore import fake_feature_fn, make_defect, make_good, quantize, safe_threshold


def save_png(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))


def make_other_setup(seed):
    """A 'different camera/background' photo: green-ish, not like the trained product."""
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), (0.1, 0.8, 0.2), dtype=np.float32)
    img[:, :128] = (0.9, 0.1, 0.6)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return quantize(img)


class TestIdentifier(unittest.TestCase):
    def test_embed_is_unit_length_and_matches_self(self):
        feats = fake_feature_fn([make_good(1), make_other_setup(2)])   # two clearly different images
        emb = identifier.embed(feats)
        self.assertTrue(np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5))
        for i, name in enumerate(["a", "b"]):
            label, sim = identifier.best_match(emb[i], emb, np.array(["a", "b"]))
            self.assertEqual((label, round(sim, 4)), (name, 1.0))

    def test_cutoff_between_groups(self):
        c = identifier.choose_cutoff([0.95, 0.96, 0.97, 0.99], [0.60, 0.70, 0.75])
        self.assertTrue(c["separated"])
        self.assertTrue(0.75 < c["cutoff"] < 0.95)
        self.assertEqual(c["known_accepted"], 1.0)
        self.assertEqual(c["unknown_rejected"], 1.0)


class TestRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.pcdir = cls.tmp / "pc"
        cls.pcdir.mkdir()
        cls.uploads = cls.tmp / "uploads"
        cls.uploads.mkdir()

        # a trained "widget" PatchCore model
        bank = pc.build_bank([make_good(i) for i in range(8)], fake_feature_fn, ratio=0.2)
        val = [pc.score_image(make_good(50 + i), bank.astype(np.float32), fake_feature_fn)[0] for i in range(12)]
        np.save(cls.pcdir / "widget_bank.npy", bank)
        (cls.pcdir / "widget_meta.json").write_text(json.dumps(
            {"threshold": safe_threshold(val), "input_size": 256}))
        (cls.pcdir / pc.BACKBONE_FILE).write_bytes(b"x")

        # recognizer: widget fingerprints, cutoff calibrated vs 'other setup' photos
        train = identifier.embed(fake_feature_fn([make_good(i) for i in range(8)]))
        pos = [identifier.best_match(e, train, np.array(["widget"] * 8))[1]
               for e in identifier.embed(fake_feature_fn([make_good(100 + i) for i in range(6)]))]
        neg = [identifier.best_match(e, train, np.array(["widget"] * 8))[1]
               for e in identifier.embed(fake_feature_fn([make_other_setup(i) for i in range(6)]))]
        calib = identifier.choose_cutoff(pos, neg)
        identifier.save(cls.pcdir, train, ["widget"] * 8, {"input_size": 256, "cutoff": calib["cutoff"]})

        cls.good, cls.defect, cls.other = cls.tmp / "g.png", cls.tmp / "d.png", cls.tmp / "o.png"
        save_png(cls.good, make_good(999))
        save_png(cls.defect, make_defect(999))
        save_png(cls.other, make_other_setup(999))

    def setUp(self):
        self.saved = (settings.PATCHCORE_DIR, settings.UPLOAD_DIR, settings.PATCHCORE_MAX_ANOMALY_AREA,
                      pc.get_feature_fn)
        settings.PATCHCORE_DIR, settings.UPLOAD_DIR = str(self.pcdir), str(self.uploads)
        pc.get_feature_fn = lambda _d: fake_feature_fn
        pc._bank_cache.clear()
        identifier._cache.clear()

    def tearDown(self):
        (settings.PATCHCORE_DIR, settings.UPLOAD_DIR, settings.PATCHCORE_MAX_ANOMALY_AREA,
         pc.get_feature_fn) = self.saved
        pc._bank_cache.clear()
        identifier._cache.clear()

    def test_known_product_detected_without_a_name(self):
        good = defect_detection.predict_defect(str(self.good), None)
        bad = defect_detection.predict_defect(str(self.defect), "")
        self.assertEqual((good["model_used"], good["status"], good["detected_product"]),
                         ("patchcore:widget", "pass", "widget"))
        self.assertEqual((bad["model_used"], bad["status"]), ("patchcore:widget", "fail"))

    def test_wrong_typed_name_does_not_fool_recognition(self):
        r = defect_detection.predict_defect(str(self.good), "Aluminum Bracket")
        self.assertEqual(r["model_used"], "patchcore:widget")

    def test_unknown_setup_raises_clear_error(self):
        """An unrecognised photo must raise RuntimeError with the supported-products message."""
        with self.assertRaises(RuntimeError) as ctx:
            defect_detection.predict_defect(str(self.other), "widget")
        msg = str(ctx.exception)
        self.assertIn("Unsupported product", msg)
        self.assertIn("bottle", msg)

    def test_huge_flagged_area_raises_clear_error(self):
        """If PatchCore flags >MAX_ANOMALY_AREA it means wrong setup — must raise, not hallucinate."""
        settings.PATCHCORE_MAX_ANOMALY_AREA = 0.001
        with self.assertRaises(RuntimeError) as ctx:
            defect_detection.predict_defect(str(self.defect), None)
        self.assertIn("Unsupported product", str(ctx.exception))

    def test_unknown_product_no_model_raises_clear_error(self):
        """No trained model + unrecognised photo -> clear RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            defect_detection.predict_defect(str(self.other), None)
        self.assertIn("Unsupported product", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()