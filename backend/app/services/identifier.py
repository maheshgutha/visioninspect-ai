"""
Product recognizer: "which trained product is this a photo of, if any?"

Every trained product's good training images are turned into one 384-number
fingerprint each (the average of the CNN patch features). A new photo is
matched to its most similar fingerprint (cosine similarity). If even the best
match is below a calibrated cutoff, the photo is "unknown" - a different
product, camera, background or lighting than the trained setup.

Built by build_identifier.py (needs torch, run once on your PC); loading and
matching here is NumPy only.

Files in <PATCHCORE_DIR>:
    identifier.npz         fingerprints (float16) + category labels
    identifier_meta.json   cutoff, input size, calibration numbers
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np

IDENT_FILE = "identifier.npz"
IDENT_META = "identifier_meta.json"

_cache = {}


def embed(feats: np.ndarray) -> np.ndarray:
    """(N, g, g, D) patch features -> (N, D) unit-length fingerprints."""
    v = feats.reshape(feats.shape[0], -1, feats.shape[-1]).mean(axis=1).astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


def best_match(query: np.ndarray, bank: np.ndarray, labels: np.ndarray):
    """query: (D,) unit vector; bank: (M, D) unit vectors. -> (label, similarity)."""
    sims = bank.astype(np.float32) @ query.astype(np.float32)
    i = int(np.argmax(sims))
    return str(labels[i]), float(sims[i])


def choose_cutoff(pos_sims, neg_sims) -> dict:
    """
    pos_sims: best-match similarity of held-out photos of trained products.
    neg_sims: same for photos of products the model was NOT trained on.
    Cutoff goes just under the 1st percentile of the positives, and between the
    two groups when they are cleanly separated.
    """
    pos = np.asarray(pos_sims, dtype=np.float64)
    neg = np.asarray(neg_sims, dtype=np.float64)
    pos_low = float(np.percentile(pos, 1))
    result = {"pos_p01": pos_low, "pos_min": float(pos.min())}
    if len(neg):
        neg_high = float(neg.max())
        result["neg_max"] = neg_high
        result["separated"] = bool(neg_high < pos_low)
        result["cutoff"] = float((neg_high + pos_low) / 2.0) if neg_high < pos_low else pos_low
        result["unknown_rejected"] = float((neg < result["cutoff"]).mean())
    else:
        result["separated"] = False
        result["cutoff"] = pos_low
    result["known_accepted"] = float((pos >= result["cutoff"]).mean())
    return result


def save(folder: Path, embeddings: np.ndarray, labels: list, meta: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / IDENT_FILE, embeddings=embeddings.astype(np.float16),
                        labels=np.asarray(labels))
    (folder / IDENT_META).write_text(json.dumps(meta, indent=2))
    _cache.pop(str(folder), None)


def has_identifier(folder: Path) -> bool:
    return (folder / IDENT_FILE).exists() and (folder / IDENT_META).exists()


def load(folder: Path):
    """-> (embeddings float32, labels array, meta dict)"""
    key = str(folder)
    if key not in _cache:
        data = np.load(folder / IDENT_FILE)
        meta = json.loads((folder / IDENT_META).read_text())
        _cache[key] = (data["embeddings"].astype(np.float32), data["labels"], meta)
    return _cache[key]
