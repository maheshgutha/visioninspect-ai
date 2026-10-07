"""
Defect Detection Module.

Primary model: PatchCore (see app/services/patchcore.py). It is trained
offline with `python train_patchcore.py` from GOOD images only, and the
resulting memory bank + threshold are loaded here for inference.

    predict_defect(path, name)  -> runs inference for the product's category

For images that don't match any trained product and cannot be identified,
predict_defect raises a RuntimeError with a clear message listing supported
products (the upload router records this as PENDING).

Supported products: bottle, cable, metal nut, screw, toothbrush.
"""
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import settings
from app.services import identifier, patchcore
from app.services.image_processing import preprocess_image
from app.services.severity_assessment import calculate_severity

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

SUPPORTED_PRODUCTS = ["bottle", "cable", "metal nut", "screw", "toothbrush"]
_UNSUPPORTED_MSG = (
    "Unsupported product or photo setup. "
    "Only these products are supported: "
    + ", ".join(SUPPORTED_PRODUCTS)
    + "."
)


def get_category_key(product_name: str) -> str:
    """
    Derive a PatchCore category key from a product name.
    MVTec-loaded items are named "MVTec - <category>"; manual uploads
    fall back to a slugified first word (or "generic" if nothing matches
    a trained category).
    """
    match = re.match(r"^\s*MVTec\s*-\s*(.+)$", product_name or "", re.IGNORECASE)
    if match:
        return match.group(1).strip().lower().replace(" ", "_")

    slug = re.sub(r"[^a-z0-9]+", "_", (product_name or "").strip().lower()).strip("_")
    if slug and patchcore.has_model(Path(settings.PATCHCORE_DIR), slug):
        return slug
    return "generic"


def _extract_regions(mask: np.ndarray, max_boxes: int = 5):
    """
    Find connected anomaly regions in the mask. Returns unscaled region
    dicts (still in analysis-resolution pixel space) that carry both the
    axis-aligned bounding box AND the true contour area.
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:max_boxes]:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 9:
            continue
        regions.append({"x": x, "y": y, "w": w, "h": h, "contour_area": float(cv2.contourArea(c))})
    return regions


def _scale_regions(regions: list, scale_x: float, scale_y: float):
    """Convert unscaled regions into the public bounding_boxes format (x/y/w/h only)."""
    return [
        {
            "x": int(r["x"] * scale_x),
            "y": int(r["y"] * scale_y),
            "w": max(1, int(r["w"] * scale_x)),
            "h": max(1, int(r["h"] * scale_y)),
        }
        for r in regions
    ]


def _classify_defect_type(mask_area: int, regions: list) -> Optional[str]:
    """
    Rule-based defect typing from the anomaly regions found in the mask.
    Only called when status == "fail" (i.e. regions is non-empty).
    """
    if not regions:
        return None

    if len(regions) >= 4:
        return "pitting"

    largest = max(regions, key=lambda r: r["w"] * r["h"])
    w, h = largest["w"], largest["h"]
    bbox_area = max(1, w * h)
    long_side, short_side = max(w, h), max(1, min(w, h))
    aspect_ratio = long_side / short_side
    extent = largest["contour_area"] / bbox_area
    coverage_ratio = largest["contour_area"] / mask_area

    if extent <= 0.35:
        return "scratch"
    if coverage_ratio >= 0.03 and extent >= 0.5:
        return "contamination"
    if aspect_ratio >= 1.8:
        return "crack"
    return "deformation"


def _save_heatmap(original_path: str, mask: np.ndarray, amap: Optional[np.ndarray] = None,
                  threshold: Optional[float] = None) -> str:
    """Overlay anomaly heat on the original image, only where the mask is set."""
    original = cv2.imread(original_path, cv2.IMREAD_COLOR)
    h, w = original.shape[:2]
    mask_resized = cv2.resize((mask.astype(np.uint8) * 255), (w, h), interpolation=cv2.INTER_NEAREST)
    if amap is not None and threshold:
        norm = np.clip((amap - threshold) / max(threshold, 1e-6), 0, 1)
        intensity = cv2.resize((norm * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap(intensity, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(original, 0.5, heat, 0.5, 0)
        overlay = np.where(mask_resized[..., None] > 0, blended, original)
    else:
        heat = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.65, heat, 0.35, 0)

    unique_filename = f"heatmap_{uuid.uuid4().hex}.png"
    out_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    cv2.imwrite(out_path, overlay)
    return unique_filename


def _title(category: str) -> str:
    return category.replace("_", " ")


def _route_image(image_path: str, hint_category: str) -> dict:
    """
    Decide which PatchCore model should inspect this photo.

    With a product recognizer built (build_identifier.py): match the photo to a
    trained product; no match -> category None (unknown product / different setup).
    Without one: fall back to the product name typed by the user.
    """
    pc_dir = Path(settings.PATCHCORE_DIR)
    legacy = {"category": hint_category if hint_category != "generic" else None,
              "similarity": None, "pack": None, "legacy": True}
    if not identifier.has_identifier(pc_dir):
        return legacy
    try:
        bank, labels, meta = identifier.load(pc_dir)
        size = int(meta["input_size"])
        rgb = patchcore.load_rgb(image_path, size)
        feats = patchcore.get_feature_fn(pc_dir)([rgb])
        label, sim = identifier.best_match(identifier.embed(feats)[0], bank, labels)
    except Exception:
        logger.exception("Product recognition failed; falling back to the product name")
        return legacy

    cutoff = settings.IDENTIFIER_CUTOFF_OVERRIDE
    if cutoff is None:
        cutoff = float(meta["cutoff"])
    known = sim >= cutoff and patchcore.has_model(pc_dir, label)
    return {"category": label if known else None, "similarity": sim,
            "pack": (rgb, feats[0], size) if known else None, "legacy": False}


def _predict_patchcore(image_path: str, category: str, pack=None):
    """
    Returns (mask, anomaly_ratio, confidence, model_used, amap, threshold, score)
    or None when PatchCore can't be used. `pack` = (rgb, feats, size) already
    computed by _route_image, reused when the sizes match.
    """
    pc_dir = Path(settings.PATCHCORE_DIR)
    if not patchcore.has_model(pc_dir, category):
        return None
    try:
        bank, meta = patchcore.load_model(pc_dir, category)
        size = int(meta.get("input_size", patchcore.INPUT_SIZE))
        if pack is not None and pack[2] == size:
            rgb, feats = pack[0], pack[1]
        else:
            rgb, feats = patchcore.load_rgb(image_path, size), None
        score, amap = patchcore.score_image(rgb, bank, patchcore.get_feature_fn(pc_dir), feats=feats)
    except Exception:
        logger.exception("PatchCore inference failed for %s", category)
        return None

    threshold = float(meta["threshold"]) * settings.PATCHCORE_THRESHOLD_SCALE
    mask = amap > threshold
    anomaly_ratio = float(mask.mean())
    if score > threshold:   # confidence grows with how far over the threshold
        confidence = 0.5 + 0.5 * min(1.0, (score - threshold) / threshold)
    else:                   # ...or how far safely under it
        confidence = 0.5 + 0.5 * min(1.0, (threshold - score) / threshold)
    confidence = float(min(0.99, confidence))
    return mask, anomaly_ratio, confidence, f"patchcore:{category}", amap, threshold, score


def predict_defect(image_path: str, product_name: Optional[str] = None) -> dict:
    """
    Inspect one image. Routing:
      1. recognised trained product      -> PatchCore for that product
      2. unknown product / other setup   -> raises RuntimeError with a clear
                                           human-readable message (inspection
                                           stays PENDING with that message as note)
    `product_name` is optional; it is only a hint when no recognizer is built.
    """
    started = time.perf_counter()
    hint = get_category_key(product_name) if product_name else "generic"

    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not read image file at {image_path}")
    orig_h, orig_w = original.shape[:2]

    route = _route_image(image_path, hint)
    category = route["category"]
    pc_result = _predict_patchcore(image_path, category, route["pack"]) if category else None

    if pc_result is not None and pc_result[1] > settings.PATCHCORE_MAX_ANOMALY_AREA:
        # anomaly covering most of the photo means the photo is not like the
        # training images (different background/lighting/product), not a real defect
        pc_result = None
        raise RuntimeError(_UNSUPPORTED_MSG)

    if pc_result is None:
        raise RuntimeError(_UNSUPPORTED_MSG)

    mask, anomaly_ratio, confidence, model_used, amap, threshold, score = pc_result
    status = "fail" if score > threshold else "pass"   # decided by image score vs trained threshold

    mask_size = mask.shape[0]   # PatchCore models may use a larger input size
    scale_x, scale_y = orig_w / mask_size, orig_h / mask_size
    regions = _extract_regions(mask)
    boxes = _scale_regions(regions, scale_x, scale_y)
    heatmap_filename = _save_heatmap(image_path, mask, amap, threshold) if boxes else None
    defect_type = _classify_defect_type(mask.size, regions) if status == "fail" else None

    detected_product = _title(category)

    severity_assessment = calculate_severity(
        status=status,
        anomaly_ratio=anomaly_ratio,
        bounding_boxes=boxes,
        defect_type=defect_type,
        confidence_score=confidence,
        image_width=orig_w,
        image_height=orig_h,
    )

    return {
        "inference_ms": int((time.perf_counter() - started) * 1000),
        "detected_product": detected_product,
        "status": status,
        "confidence_score": round(confidence, 3),
        "anomaly_ratio": round(anomaly_ratio, 5),
        "bounding_boxes": boxes,
        "heatmap_filename": heatmap_filename,
        "model_used": model_used,
        "defect_type": defect_type,
        "severity_score": severity_assessment["severity_score"],
        "severity_level": severity_assessment["severity_level"],
        "quality_recommendation": severity_assessment["quality_recommendation"],
        "severity_details": severity_assessment["severity_details"],
    }