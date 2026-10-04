"""
Set the pass/fail threshold of each trained PatchCore model WITHOUT retraining.

    python set_thresholds.py                      # apply the recommended values below
    python set_thresholds.py --show               # just print current thresholds
    python set_thresholds.py --set screw=1.60     # override one category

Values come from the "threshold options" tables printed by train_patchcore.py
(320 px models, full MVTec data). Raise a number -> fewer false rejects but more
missed defects; lower it -> the opposite.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app.config import settings

RECOMMENDED = {
    "bottle": 1.795,      # strict:   100% good passed / 100% defects caught
    "cable": 2.401,       # balanced:  93% / 90%
    "metal_nut": 1.990,   # balanced:  95% / 99%
    "screw": 1.618,       # balanced:  98% / 84%
    "toothbrush": 1.811,  # strict:    75% / 100%  (only 12 good test images - see notes)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=settings.PATCHCORE_DIR)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="CATEGORY=VALUE")
    args = ap.parse_args()

    if args.set:   # explicit overrides: only the categories named
        values = {}
        for item in args.set:
            name, _, val = item.partition("=")
            values[name] = float(val)
    else:
        values = dict(RECOMMENDED)

    folder = Path(args.dir)
    for meta_path in sorted(folder.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())
        cat = meta_path.name.replace("_meta.json", "")
        if args.show or cat not in values:
            print(f"{cat:12s} threshold={meta['threshold']:.3f}  (trained rule: {meta.get('threshold_rule', '?')})")
            continue
        meta.setdefault("threshold_trained", meta["threshold"])
        old, meta["threshold"] = meta["threshold"], values[cat]
        meta["threshold_rule"] = "manual"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"{cat:12s} {old:.3f} -> {meta['threshold']:.3f}")


if __name__ == "__main__":
    main()
