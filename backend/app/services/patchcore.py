"""
PatchCore anomaly-detection engine for VisionInspect AI.

How it works (this is what "training" means here):

  1. A CNN pretrained on ImageNet (ResNet18, layers 1-3) turns every image
     into a grid of 384-dim "patch features" (one per 8x8 pixel block).
  2. Training = collect the patch features of GOOD images only and keep a
     small, representative subset (greedy "coreset") as the memory bank.
  3. Inference = for each patch of a new image, find the distance to its
     nearest neighbour in the memory bank. Patches that look like nothing
     seen in good products get a high score -> that is the defect.
     The image score is the maximum patch score.

Only the CNN needs PyTorch. Everything else is NumPy/OpenCV, so the
backend can load a trained bank and run inference on small servers.

Files per category (written by train_patchcore.py, read here):
    <PATCHCORE_DIR>/backbone_resnet18.pt   shared CNN weights (~11 MB)
    <PATCHCORE_DIR>/<category>_bank.npy    memory bank, float16
    <PATCHCORE_DIR>/<category>_meta.json   threshold, metrics, settings
"""
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

INPUT_SIZE = 256            # images are resized (not cropped) to 256x256
FEATURE_GRID = INPUT_SIZE // 8   # layer2 has stride 8 -> 32x32 patches
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BACKBONE_FILE = "backbone_resnet18.pt"

# Feature extractor signature: list of RGB float32 images in [0,1], HxWx3
# (already resized to INPUT_SIZE) -> array (N, FEATURE_GRID, FEATURE_GRID, D)
FeatureFn = Callable[[list], np.ndarray]

_backbone_cache = {}
_bank_cache = {}  # category -> (bank float32, meta); only the latest is kept


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------
def load_rgb(path: str, size: Optional[int] = None) -> np.ndarray:
    """Read an image as RGB float32 in [0,1], resized to size x size (default INPUT_SIZE)."""
    size = size or INPUT_SIZE
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# CNN backbone (the only torch-dependent part)
# --------------------------------------------------------------------------
def save_backbone(patchcore_dir: Path) -> Path:
    """Download ImageNet ResNet18 once and store layers conv1..layer3 locally."""
    import torch
    import torchvision

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    keep = ("conv1", "bn1", "layer1", "layer2", "layer3")
    state = {k: v for k, v in model.state_dict().items() if k.split(".")[0] in keep}
    patchcore_dir.mkdir(parents=True, exist_ok=True)
    out = patchcore_dir / BACKBONE_FILE
    torch.save(state, out)
    return out


def _get_backbone(patchcore_dir: Path):
    key = str(patchcore_dir)
    if key in _backbone_cache:
        return _backbone_cache[key]

    import torch
    import torchvision

    weights_path = patchcore_dir / BACKBONE_FILE
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{weights_path} not found. Run `python train_patchcore.py` first."
        )

    model = torchvision.models.resnet18(weights=None)
    result = model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=False
    )
    # fc.* and layer4.* are intentionally absent
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in backbone weights: {unexpected}")
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    _backbone_cache[key] = model
    return model


def torch_feature_fn(patchcore_dir: Path) -> FeatureFn:
    """Build the real ResNet18 feature extractor."""
    import torch
    import torch.nn.functional as F

    model = _get_backbone(patchcore_dir)
    mean = torch.from_numpy(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.from_numpy(IMAGENET_STD).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract(images: list) -> np.ndarray:
        outputs = []
        for start in range(0, len(images), 16):  # small batches keep RAM low
            batch = np.stack(images[start:start + 16])
            x = torch.from_numpy(batch).permute(0, 3, 1, 2)
            x = (x - mean) / std
            x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            x = model.layer1(x)
            f2 = model.layer2(x)           # (B,128,32,32)
            f3 = model.layer3(f2)          # (B,256,16,16)
            # local neighbourhood aggregation, as in the PatchCore paper
            f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
            f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            feats = torch.cat([f2, f3], dim=1).permute(0, 2, 3, 1)  # (B,32,32,384)
            outputs.append(feats.numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    return extract


# --------------------------------------------------------------------------
# Memory bank (training side)
# --------------------------------------------------------------------------
def greedy_coreset(features: np.ndarray, ratio: float, seed: int = 0, proj_dim: int = 128) -> np.ndarray:
    """
    Greedy k-center subsampling: repeatedly pick the patch that is farthest
    from everything already picked, so the small bank still covers the
    whole variety of good patches. Returns the selected row indices.
    """
    n = features.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    if features.shape[1] > proj_dim:  # random projection only for speed of selection
        proj = rng.normal(size=(features.shape[1], proj_dim)).astype(np.float32) / np.sqrt(proj_dim)
        x = features @ proj
    else:
        x = features.astype(np.float32)

    sq = (x ** 2).sum(axis=1)
    selected = np.empty(k, dtype=np.int64)
    current = int(rng.integers(n))
    min_d = np.full(n, np.inf, dtype=np.float32)
    for i in range(k):
        selected[i] = current
        d = sq - 2.0 * (x @ x[current]) + sq[current]
        np.minimum(min_d, d, out=min_d)
        current = int(np.argmax(min_d))
    return selected


def build_bank(images: list, feature_fn: FeatureFn, ratio: float = 0.05, seed: int = 0,
               rotations: int = 1) -> np.ndarray:
    """
    Extract patch features from good images and reduce them to a coreset bank.
    rotations=4 also adds the 90/180/270-degree turns of every good image, so
    parts that sit at random angles (screws, nuts) are not flagged just for
    being rotated. Uses more RAM while training (about rotations x).
    """
    chunks = []
    for k in range(max(1, rotations)):
        turned = images if k == 0 else [np.ascontiguousarray(np.rot90(im, k)) for im in images]
        for start in range(0, len(turned), 16):
            feats = feature_fn(turned[start:start + 16])
            chunks.append(feats.reshape(-1, feats.shape[-1]).astype(np.float32))
    all_feats = np.concatenate(chunks, axis=0)
    idx = greedy_coreset(all_feats, ratio, seed=seed)
    return all_feats[idx].astype(np.float16)


# --------------------------------------------------------------------------
# Scoring (inference side, NumPy only)
# --------------------------------------------------------------------------
def _nn_distance(queries: np.ndarray, bank: np.ndarray, bank_sq: np.ndarray) -> np.ndarray:
    """Distance from each query vector to its nearest bank vector."""
    q_sq = (queries ** 2).sum(axis=1, keepdims=True)
    best = np.full(queries.shape[0], np.inf, dtype=np.float32)
    for start in range(0, bank.shape[0], 8192):  # chunk bank so the matrix stays small
        chunk = bank[start:start + 8192]
        d2 = q_sq - 2.0 * (queries @ chunk.T) + bank_sq[start:start + 8192][None, :]
        np.minimum(best, d2.min(axis=1), out=best)
    return np.sqrt(np.clip(best, 0, None))


def score_image(rgb: np.ndarray, bank: np.ndarray, feature_fn: FeatureFn, blur_sigma: float = 4.0,
                feats: Optional[np.ndarray] = None):
    """
    Returns (image_score, anomaly_map). anomaly_map has the same height/width
    as the input image (any multiple of 8); image_score is its maximum.
    """
    side = rgb.shape[0]
    if feats is None:                                  # (g,g,D); callers may pass them in to save a CNN pass
        feats = feature_fn([rgb])[0]
    g = feats.shape[0]
    queries = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    bank32 = bank.astype(np.float32)
    bank_sq = (bank32 ** 2).sum(axis=1)
    patch_scores = _nn_distance(queries, bank32, bank_sq).reshape(g, g)
    amap = cv2.resize(patch_scores, (side, side), interpolation=cv2.INTER_LINEAR)
    amap = cv2.GaussianBlur(amap, (0, 0), blur_sigma)
    return float(amap.max()), amap


# --------------------------------------------------------------------------
# Threshold + metrics helpers
# --------------------------------------------------------------------------
THRESHOLD_RULES = ("strict", "balanced", "sensitive")


def choose_threshold(val_good_scores: list, rule: str = "strict") -> float:
    """
    Threshold from HELD-OUT GOOD images only (no defect images needed).
      strict    : max(mean + 3*std, highest good score)  -> almost no false alarms
      balanced  : mean + 2*std                            -> some false alarms, catches more
      sensitive : mean + 1*std                            -> many false alarms, catches most
    """
    s = np.asarray(val_good_scores, dtype=np.float64)
    if rule == "balanced":
        return float(s.mean() + 2.0 * s.std())
    if rule == "sensitive":
        return float(s.mean() + 1.0 * s.std())
    return float(max(s.mean() + 3.0 * s.std(), s.max()))


def auroc(labels, scores) -> float:
    """Rank-based AUROC (labels: 1 = defect). Handles ties; no sklearn needed."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    all_scores = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(all_scores))
    i = 0
    while i < len(all_scores):
        j = i
        while j + 1 < len(all_scores) and all_scores[j + 1] == all_scores[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    rank_of = np.empty(len(all_scores))
    rank_of[order] = ranks
    pos_rank_sum = rank_of[:len(pos)].sum()
    return float((pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


# --------------------------------------------------------------------------
# Loading trained models for the API
# --------------------------------------------------------------------------
def bank_paths(patchcore_dir: Path, category: str):
    return patchcore_dir / f"{category}_bank.npy", patchcore_dir / f"{category}_meta.json"


def has_model(patchcore_dir: Path, category: str) -> bool:
    bank_path, meta_path = bank_paths(patchcore_dir, category)
    return bank_path.exists() and meta_path.exists() and (patchcore_dir / BACKBONE_FILE).exists()


def load_model(patchcore_dir: Path, category: str):
    """Load (bank, meta). Only one bank is kept in RAM at a time (small servers)."""
    if category in _bank_cache:
        return _bank_cache[category]
    bank_path, meta_path = bank_paths(patchcore_dir, category)
    bank = np.load(bank_path)
    meta = json.loads(meta_path.read_text())
    _bank_cache.clear()
    _bank_cache[category] = (bank, meta)
    return bank, meta


_feature_fn_cache = {}


def get_feature_fn(patchcore_dir: Path) -> FeatureFn:
    key = str(patchcore_dir)
    if key not in _feature_fn_cache:
        _feature_fn_cache[key] = torch_feature_fn(patchcore_dir)
    return _feature_fn_cache[key]