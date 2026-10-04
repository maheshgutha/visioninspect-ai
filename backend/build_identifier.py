"""
Build the product recognizer ("which trained product is this?") and calibrate it.

Run ONCE after train_patchcore.py, from the backend/ folder with the venv active:

    python build_identifier.py --dataset-root dataset

What it does:
  1. Fingerprints every good training image of each trained product.
  2. Measures how similar held-out TEST photos of those products are to the fingerprints
     (these must be accepted), and how similar photos of OTHER MVTec products you did not
     train are (these must be rejected as "unknown").
  3. Picks the accept/reject cutoff between the two groups and saves it.
  4. Prints how much of the image real defects cover for each product (helps judge
     PATCHCORE_MAX_ANOMALY_AREA).

Needs: model_cache/patchcore/ from train_patchcore.py, and the other MVTec categories
in the dataset folder to act as "unknown" examples (without them the cutoff is less reliable).
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.services import identifier
from app.services import patchcore as pc

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def images_in(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT) if folder.exists() else []


def fingerprints(paths, feature_fn, size):
    out = []
    for start in range(0, len(paths), 8):
        batch = [pc.load_rgb(str(p), size) for p in paths[start:start + 8]]
        out.append(identifier.embed(feature_fn(batch)))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 384), dtype=np.float32)


def sample(paths, n, seed=0):
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    return paths[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="dataset")
    ap.add_argument("--out-dir", default=settings.PATCHCORE_DIR)
    ap.add_argument("--input-size", type=int, default=320)
    ap.add_argument("--max-test-per-category", type=int, default=40)
    ap.add_argument("--max-unknown-per-category", type=int, default=25)
    ap.add_argument("--skip-area-stats", action="store_true")
    args = ap.parse_args()

    root, out_dir = Path(args.dataset_root), Path(args.out_dir)
    trained = sorted(p.name.replace("_meta.json", "") for p in out_dir.glob("*_meta.json"))
    if not trained:
        sys.exit(f"No trained models in {out_dir}. Run train_patchcore.py first.")
    others = sorted(c.name for c in root.iterdir() if c.is_dir() and c.name not in trained
                    and images_in(c / "test"))
    print(f"Trained products: {trained}")
    print(f"Untrained products used as 'unknown' examples: {others or 'NONE (cutoff will be less reliable)'}")

    feature_fn = pc.torch_feature_fn(out_dir)
    size = args.input_size

    # 1) fingerprints from training images
    bank, labels = [], []
    for cat in trained:
        emb = fingerprints(images_in(root / cat / "train" / "good"), feature_fn, size)
        bank.append(emb)
        labels += [cat] * len(emb)
        print(f"  fingerprinted {cat}: {len(emb)} training images")
    bank = np.concatenate(bank, axis=0)
    labels_arr = np.asarray(labels)

    # 2) held-out positives and unknown negatives
    pos_sims, correct, total = [], 0, 0
    for cat in trained:
        paths = sample(images_in(root / cat / "test"), args.max_test_per_category)
        for emb in fingerprints(paths, feature_fn, size):
            label, sim = identifier.best_match(emb, bank, labels_arr)
            pos_sims.append(sim)
            correct += int(label == cat)
            total += 1
    neg_sims = []
    for cat in others:
        paths = sample(images_in(root / cat / "test"), args.max_unknown_per_category)
        for emb in fingerprints(paths, feature_fn, size):
            neg_sims.append(identifier.best_match(emb, bank, labels_arr)[1])

    calib = identifier.choose_cutoff(pos_sims, neg_sims)
    print(f"\nProduct identification accuracy on held-out test photos: {correct}/{total} = {correct / max(total, 1):.1%}")
    print(f"Known-product similarity   : min={min(pos_sims):.3f}  1st percentile={calib['pos_p01']:.3f}")
    if neg_sims:
        print(f"Unknown-product similarity : max={calib['neg_max']:.3f}  (over {len(neg_sims)} photos)")
        print("  -> cleanly separated" if calib["separated"] else
              "  -> OVERLAP: some unknown products look like trained ones; cutoff is a compromise")
    print(f"CUTOFF = {calib['cutoff']:.3f}   known accepted {calib['known_accepted']:.1%}"
          + (f", unknown rejected {calib['unknown_rejected']:.1%}" if neg_sims else ""))

    # 3) how much of the image do real defects cover?
    area_stats = {}
    if not args.skip_area_stats:
        print("\nShare of the image flagged on real defect test photos (guide for PATCHCORE_MAX_ANOMALY_AREA):")
        for cat in trained:
            bank_c, meta = pc.load_model(out_dir, cat)
            bank32 = bank_c.astype(np.float32)
            csize = int(meta.get("input_size", pc.INPUT_SIZE))
            areas = []
            for sub in sorted(p for p in (root / cat / "test").iterdir() if p.is_dir() and p.name != "good"):
                for img in images_in(sub)[:10]:
                    score, amap = pc.score_image(pc.load_rgb(str(img), csize), bank32, feature_fn)
                    areas.append(float((amap > meta["threshold"]).mean()))
            if areas:
                area_stats[cat] = {"median": float(np.median(areas)), "p95": float(np.percentile(areas, 95)),
                                   "max": float(max(areas))}
                a = area_stats[cat]
                print(f"  {cat:12s} median={a['median']:.1%}  95th pct={a['p95']:.1%}  max={a['max']:.1%}")
        print(f"  current PATCHCORE_MAX_ANOMALY_AREA = {settings.PATCHCORE_MAX_ANOMALY_AREA:.0%}")

    meta = {"input_size": size, "cutoff": calib["cutoff"], "categories": trained,
            "n_fingerprints": int(len(labels)), "identification_accuracy": correct / max(total, 1),
            "calibration": calib, "area_stats": area_stats}
    identifier.save(out_dir, bank, labels, meta)
    print(f"\nSaved {identifier.IDENT_FILE} and {identifier.IDENT_META} to {out_dir}. Commit them with the models.")


if __name__ == "__main__":
    main()
