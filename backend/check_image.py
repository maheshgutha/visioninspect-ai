"""
Show what the app will do with ONE image - without starting the server and
without calling any AI service.

    python check_image.py path\\to\\image.png

Prints which trained product the image looks like, how close the match is
compared with the accept/reject cutoff, and (if it matches) the PatchCore
score, threshold and how much of the picture it flags.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.services import defect_detection as dd
from app.services import identifier, patchcore


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python check_image.py <image path>")
    path = sys.argv[1]
    if not Path(path).is_file():
        sys.exit(f"File not found: {path}\nUse the real location of an image, e.g.  "
                 f"python check_image.py dataset\\screw\\test\\good\\000.png")
    pc_dir = Path(settings.PATCHCORE_DIR)
    if not identifier.has_identifier(pc_dir):
        sys.exit("No product recognizer yet. Run build_identifier.py first.")

    bank, labels, meta = identifier.load(pc_dir)
    cutoff = settings.IDENTIFIER_CUTOFF_OVERRIDE if settings.IDENTIFIER_CUTOFF_OVERRIDE is not None else meta["cutoff"]
    size = int(meta["input_size"])
    rgb = patchcore.load_rgb(path, size)
    feats = patchcore.get_feature_fn(pc_dir)([rgb])
    label, sim = identifier.best_match(identifier.embed(feats)[0], bank, labels)

    print(f"Closest trained product : {label}")
    print(f"Similarity              : {sim:.4f}   (cutoff {cutoff:.4f}; gap {sim - cutoff:+.4f})")
    if sim < cutoff:
        print("DECISION: unknown product / different setup -> AI vision would inspect it")
        return

    result = dd._predict_patchcore(path, label, (rgb, feats[0], size))
    if result is None:
        sys.exit(f"No usable PatchCore model for {label}")
    mask, ratio, conf, model_used, amap, threshold, score = result
    limit = settings.PATCHCORE_MAX_ANOMALY_AREA
    print(f"PatchCore score         : {score:.3f}   (threshold {threshold:.3f})")
    print(f"Image area flagged      : {ratio:.1%}   (limit {limit:.0%})")
    if ratio > limit:
        print("DECISION: flagged area too large -> treated as 'not the trained setup' -> AI vision would inspect it")
    else:
        print(f"DECISION: inspected by {model_used} -> {'FAIL (defect)' if score > threshold else 'PASS'}")


if __name__ == "__main__":
    main()