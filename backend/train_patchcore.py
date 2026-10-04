"""
Train PatchCore models for MVTec AD categories and EVALUATE them.

Usage (run from the backend/ folder, with the venv active):

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python train_patchcore.py                      # all categories found in the dataset
    python train_patchcore.py --categories screw bottle
    python train_patchcore.py --coreset-ratio 0.1  # bigger bank = slightly better, bigger files

What it does per category:
  1. Splits train/good into FIT (80%) and VALIDATION (20%, never trained on).
  2. Builds the memory bank from FIT images only.
  3. Sets the fail threshold from the VALIDATION good images (no defects used).
  4. Evaluates on the official test split: image AUROC, pass/fail accuracy,
     and pixel AUROC when ground-truth masks exist.
  5. Saves <category>_bank.npy and <category>_meta.json into model_cache/patchcore/
  6. Writes model_cache/patchcore/metrics.json with every number printed.

Training only needs 'good' images - exactly like a real factory line.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.services import patchcore as pc

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def find_categories(dataset_root: Path):
    cats = []
    for c in sorted(dataset_root.iterdir()):
        if c.is_dir() and list_images(c / "train" / "good"):
            cats.append(c.name)
    return cats


def evaluate(category, dataset_root, bank, threshold, feature_fn, size, max_pixels_per_img=2000):
    test_dir = dataset_root / category / "test"
    gt_dir = dataset_root / category / "ground_truth"
    labels, scores, types = [], [], []
    pix_labels, pix_scores = [], []
    rng = np.random.default_rng(0)

    for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for img_path in list_images(sub):
            rgb = pc.load_rgb(str(img_path), size)
            score, amap = pc.score_image(rgb, bank, feature_fn)
            is_defect = 0 if sub.name == "good" else 1
            labels.append(is_defect)
            scores.append(score)
            types.append(sub.name)

            if gt_dir.exists():
                if is_defect:
                    mask_path = gt_dir / sub.name / f"{img_path.stem}_mask.png"
                    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
                    if m is None:
                        continue
                    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127
                else:
                    m = np.zeros((size, size), dtype=bool)
                idx = rng.choice(m.size, size=min(max_pixels_per_img, m.size), replace=False)
                pix_labels.append(m.ravel()[idx].astype(np.int8))
                pix_scores.append(amap.ravel()[idx])

    labels, scores = np.array(labels), np.array(scores)
    pred = scores > threshold
    result = {
        "n_test_good": int((labels == 0).sum()),
        "n_test_defect": int((labels == 1).sum()),
        "image_auroc": round(pc.auroc(labels, scores), 4),
        "accuracy_at_threshold": round(float((pred == labels.astype(bool)).mean()), 4),
        "good_pass_rate": round(float((~pred[labels == 0]).mean()), 4),
        "defect_catch_rate": round(float(pred[labels == 1].mean()), 4),
    }
    if pix_labels:
        result["pixel_auroc"] = round(pc.auroc(np.concatenate(pix_labels), np.concatenate(pix_scores)), 4)

    result["_labels"], result["_scores"] = labels, scores

    per_type = {}
    for t in sorted(set(types)):
        sel = [i for i, x in enumerate(types) if x == t]
        flagged = float(pred[sel].mean())
        per_type[t] = {"n": len(sel), "flagged_rate": round(flagged, 3)}
    result["per_type_flagged_rate"] = per_type
    return result


def threshold_table(val_scores, labels, scores):
    """Show what each threshold rule would do on the test set (for choosing a rule)."""
    rows = {}
    for rule in pc.THRESHOLD_RULES:
        t = pc.choose_threshold(val_scores, rule)
        pred = scores > t
        rows[rule] = (t, float((~pred[labels == 0]).mean()), float(pred[labels == 1].mean()))
    # best possible threshold on the test set itself - OPTIMISTIC, for diagnosis only
    best = max(np.unique(scores), key=lambda t: ((scores[labels == 1] > t).mean()
                                                + (scores[labels == 0] <= t).mean()))
    pred = scores > best
    rows["oracle(test)"] = (float(best), float((~pred[labels == 0]).mean()), float(pred[labels == 1].mean()))
    return rows


def train_category(category, dataset_root, out_dir, ratio, val_fraction, feature_fn, seed,
                   rotations=1, rule="strict", size=pc.INPUT_SIZE):
    t0 = time.time()
    train_imgs = list_images(dataset_root / category / "train" / "good")
    if not train_imgs:
        sys.exit(f"\n[{category}] no images found in {dataset_root / category / 'train' / 'good'}\n"
                 f"Copy the MVTec AD '{category}' folder (with train/ test/ ground_truth/) into {dataset_root} first.")
    random.Random(seed).shuffle(train_imgs)
    n_val = max(5, int(len(train_imgs) * val_fraction))
    val_paths, fit_paths = train_imgs[:n_val], train_imgs[n_val:]
    print(f"[{category}] fit={len(fit_paths)} val={len(val_paths)} good images")

    fit_images = [pc.load_rgb(str(p), size) for p in fit_paths]
    bank = pc.build_bank(fit_images, feature_fn, ratio=ratio, seed=seed, rotations=rotations)
    del fit_images
    print(f"[{category}] memory bank: {bank.shape[0]} patches x {bank.shape[1]} dims "
          f"({bank.nbytes / 1e6:.1f} MB)")

    bank32 = bank.astype(np.float32)
    val_scores = [pc.score_image(pc.load_rgb(str(p), size), bank32, feature_fn)[0] for p in val_paths]
    threshold = pc.choose_threshold(val_scores, rule)
    print(f"[{category}] val good scores: min={min(val_scores):.3f} max={max(val_scores):.3f} "
          f"-> threshold ({rule})={threshold:.3f}")

    metrics = evaluate(category, dataset_root, bank32, threshold, feature_fn, size)
    labels, scores = metrics.pop("_labels"), metrics.pop("_scores")
    print(f"[{category}] test score range: good {scores[labels == 0].min():.2f}-{scores[labels == 0].max():.2f}"
          f" | defect {scores[labels == 1].min():.2f}-{scores[labels == 1].max():.2f}")
    print(f"[{category}] threshold options on the test set:")
    for name, (t, gp, dc) in threshold_table(val_scores, labels, scores).items():
        print(f"      {name:13s} thr={t:7.3f}  good passed={gp:4.0%}  defects caught={dc:4.0%}")
    print(f"[{category}] TEST  image AUROC={metrics['image_auroc']}  "
          f"good passed={metrics['good_pass_rate']:.0%}  defects caught={metrics['defect_catch_rate']:.0%}"
          + (f"  pixel AUROC={metrics['pixel_auroc']}" if "pixel_auroc" in metrics else ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path, meta_path = pc.bank_paths(out_dir, category)
    np.save(bank_path, bank)
    meta = {
        "category": category,
        "threshold": threshold,
        "val_good_scores": {"min": float(min(val_scores)), "max": float(max(val_scores)),
                            "mean": float(np.mean(val_scores))},
        "bank_size": int(bank.shape[0]),
        "feature_dim": int(bank.shape[1]),
        "coreset_ratio": ratio,
        "rotations": rotations,
        "threshold_rule": rule,
        "input_size": size,
        "n_fit_images": len(fit_paths),
        "metrics": metrics,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate PatchCore models")
    parser.add_argument("--dataset-root", default=settings.MVTEC_DATASET_PATH)
    parser.add_argument("--out-dir", default=settings.PATCHCORE_DIR)
    parser.add_argument("--categories", nargs="*", help="default: every category with train/good images")
    parser.add_argument("--coreset-ratio", type=float, default=settings.PATCHCORE_CORESET_RATIO)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--rotations", type=int, default=1, choices=[1, 2, 4],
                        help="4 = also train on 90/180/270 degree turns (use for screw, metal_nut; ~4x RAM)")
    parser.add_argument("--threshold-rule", default="strict", choices=list(pc.THRESHOLD_RULES))
    parser.add_argument("--input-size", type=int, default=pc.INPUT_SIZE,
                        help="image side in pixels, multiple of 8 (256 default; 320/384 see smaller defects but are slower)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.input_size % 8:
        sys.exit("--input-size must be a multiple of 8")
    dataset_root, out_dir = Path(args.dataset_root), Path(args.out_dir)
    categories = args.categories or find_categories(dataset_root)
    if not categories:
        sys.exit(f"No categories with train/good images found in {dataset_root}")
    print(f"Categories: {categories}")

    if not (out_dir / pc.BACKBONE_FILE).exists():
        print("Downloading ImageNet ResNet18 (one time, ~45 MB)...")
        pc.save_backbone(out_dir)
    feature_fn = pc.torch_feature_fn(out_dir)

    summary = {}
    for cat in categories:
        summary[cat] = train_category(cat, dataset_root, out_dir, args.coreset_ratio,
                                      args.val_fraction, feature_fn, args.seed,
                                      args.rotations, args.threshold_rule, args.input_size)

    (out_dir / "metrics.json").write_text(json.dumps(
        {c: s["metrics"] | {"threshold": s["threshold"]} for c, s in summary.items()}, indent=2))

    print("\n=== SUMMARY ===")
    print(f"{'category':14s} {'imgAUROC':>9s} {'pixAUROC':>9s} {'goodPass':>9s} {'defCaught':>10s}")
    for c, s in summary.items():
        m = s["metrics"]
        print(f"{c:14s} {m['image_auroc']:>9} {m.get('pixel_auroc', '-'):>9} "
              f"{m['good_pass_rate']:>9.0%} {m['defect_catch_rate']:>10.0%}")
    print(f"\nSaved to {out_dir}. Commit that folder and redeploy.")


if __name__ == "__main__":
    main()