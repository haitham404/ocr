"""
evaluate.py

Evaluates the PaddleOCR text DETECTOR ONLY (e.g. PP-OCRv5_mobile_det) against
the ground-truth boxes listed in Dataset/val.txt.

It deliberately uses the standalone `TextDetection` module (not the full
`PaddleOCR` pipeline used in run.py), so recognition never runs and metrics
reflect detection quality alone.

For every image in the validation split:
  1. Loads GT polygons from val.txt (PPOCRLabel format, same format Label.txt
     / train.txt / val.txt all use: "img_path\t[{transcription, points}, ...]").
  2. Runs the detector and collects predicted polygons + scores.
  3. Greedily matches predictions to GT boxes using polygon IoU.
  4. Accumulates TP / FP / FN -> Precision / Recall / F1 / mean IoU.
  5. Draws GT boxes in GREEN and predicted boxes in RED, saves the image.

Usage:
    python evaluate.py
    python evaluate.py --iou-thresh 0.5 --score-thresh 0.3
    python evaluate.py --val-file Dataset/val.txt --limit 50 --no-vis
    python evaluate.py --model-name PP-OCRv5_mobile_det --device cpu

Outputs (default OUTPUT_DIR = eval_results/):
    eval_results/detector_metrics.json   -> overall summary
    eval_results/per_image_metrics.csv   -> per-image breakdown
    eval_results/visualizations/*.jpg    -> GT (green) vs predictions (red)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# =====================================================
# CONFIG (defaults, all overridable via CLI)
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VAL_FILE = BASE_DIR / "Dataset" / "val.txt"
DEFAULT_IMAGE_DIR = BASE_DIR / "Dataset"
DEFAULT_OUTPUT_DIR = BASE_DIR / "eval_results"
DEFAULT_MODEL_NAME = "PP-OCRv5_mobile_det"

GT_COLOR = (0, 255, 0)      # green, BGR
PRED_COLOR = (0, 0, 255)    # red, BGR
GT_THICKNESS = 2
PRED_THICKNESS = 2

# Ground-truth boxes whose transcription is one of these are treated as
# "ignore" regions (standard convention in text-detection datasets): they
# are neither counted as FN if missed, nor do overlapping predictions count
# as FP.
IGNORE_TRANSCRIPTIONS = {"###", ""}


# =====================================================
# GEOMETRY HELPERS (pure numpy, no extra dependencies)
# =====================================================
def polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula. poly: (N, 2) array."""
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _clip_by_edge(subject: np.ndarray, edge_p1: np.ndarray, edge_p2: np.ndarray) -> np.ndarray:
    """One step of Sutherland-Hodgman: clip `subject` polygon against the
    half-plane defined by the directed edge (edge_p1 -> edge_p2), keeping
    points to the left of the edge (assumes clip polygon is CCW)."""
    if len(subject) == 0:
        return subject

    def is_inside(p):
        return (edge_p2[0] - edge_p1[0]) * (p[1] - edge_p1[1]) - \
               (edge_p2[1] - edge_p1[1]) * (p[0] - edge_p1[0]) >= 0

    def intersect(p1, p2):
        d1 = edge_p2 - edge_p1
        d2 = p2 - p1
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            return p2
        t = ((p1[0] - edge_p1[0]) * d2[1] - (p1[1] - edge_p1[1]) * d2[0]) / denom
        return edge_p1 + t * d1

    output = []
    n = len(subject)
    for i in range(n):
        cur = subject[i]
        prev = subject[i - 1]
        cur_in = is_inside(cur)
        prev_in = is_inside(prev)
        if cur_in:
            if not prev_in:
                output.append(intersect(prev, cur))
            output.append(cur)
        elif prev_in:
            output.append(intersect(prev, cur))
    return np.array(output, dtype=np.float64)


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman needs the clip polygon oriented CCW."""
    x = poly[:, 0]
    y = poly[:, 1]
    signed_area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly[::-1] if signed_area < 0 else poly


def polygon_intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Intersection area of two convex polygons via Sutherland-Hodgman."""
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    clip = _ensure_ccw(poly_b.astype(np.float64))
    subject = poly_a.astype(np.float64)
    n = len(clip)
    for i in range(n):
        subject = _clip_by_edge(subject, clip[i - 1], clip[i])
        if len(subject) == 0:
            return 0.0
    return polygon_area(subject)


def poly_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """IoU of two (assumed convex) polygons, each an (N, 2) array."""
    poly_a = np.asarray(poly_a, dtype=np.float64).reshape(-1, 2)
    poly_b = np.asarray(poly_b, dtype=np.float64).reshape(-1, 2)
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter = polygon_intersection_area(poly_a, poly_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# =====================================================
# LABEL PARSING (Dataset/val.txt, PPOCRLabel format)
# =====================================================
def parse_label_file(label_path: Path):
    """Returns a list of (image_relative_path, gt_boxes) where gt_boxes is a
    list of {"points": np.ndarray (N,2), "text": str, "ignore": bool}."""
    if not label_path.exists():
        raise FileNotFoundError(f"Label/val file not found: {label_path}")

    entries = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            print(f"[warn] line {line_no}: no tab separator, skipping -> {line[:60]}")
            continue
        img_rel, ann_json = line.split("\t", 1)
        try:
            anns = json.loads(ann_json)
        except json.JSONDecodeError as e:
            print(f"[warn] line {line_no}: bad JSON annotations, skipping ({e})")
            continue

        gt_boxes = []
        for a in anns:
            pts = np.array(a.get("points", []), dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            text = str(a.get("transcription", ""))
            ignore = text.strip() in IGNORE_TRANSCRIPTIONS or bool(a.get("difficult", False))
            gt_boxes.append({"points": pts, "text": text, "ignore": ignore})
        entries.append((img_rel, gt_boxes))
    return entries


# =====================================================
# MATCHING / METRICS
# =====================================================
def match_predictions_to_gt(gt_boxes, pred_boxes, iou_thresh):
    """
    gt_boxes:   list of {"points", "ignore"}
    pred_boxes: list of {"points", "score"}  (any order; sorted internally)

    Returns a dict with tp, fp, fn, matched_ious (list), and per-prediction
    match info (list of bool, aligned with the ORIGINAL pred_boxes order)
    useful for visualization.
    """
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_boxes[i]["score"])
    gt_used = [False] * len(gt_boxes)
    pred_is_tp = [False] * len(pred_boxes)  # aligned to original order

    tp = 0
    fp = 0
    matched_ious = []

    for i in order:
        pred_poly = pred_boxes[i]["points"]
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            iou = poly_iou(pred_poly, g["points"])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j != -1 and best_iou >= iou_thresh:
            gt_used[best_j] = True
            if gt_boxes[best_j]["ignore"]:
                # matched an ignore region -> not counted either way
                continue
            tp += 1
            matched_ious.append(best_iou)
            pred_is_tp[i] = True
        else:
            # unmatched prediction: check if it mostly overlaps an ignore
            # region before counting it as a false positive
            overlaps_ignore = False
            for g in gt_boxes:
                if g["ignore"] and poly_iou(pred_poly, g["points"]) >= iou_thresh:
                    overlaps_ignore = True
                    break
            if not overlaps_ignore:
                fp += 1

    fn = sum(1 for j, g in enumerate(gt_boxes) if not g["ignore"] and not gt_used[j])

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_ious": matched_ious,
        "pred_is_tp": pred_is_tp,
    }


def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# =====================================================
# VISUALIZATION
# =====================================================
def draw_boxes(image, gt_boxes, pred_boxes):
    vis = image.copy()

    for g in gt_boxes:
        pts = g["points"].astype(int).reshape(-1, 1, 2)
        color = (0, 200, 0) if g["ignore"] else GT_COLOR
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=GT_THICKNESS)

    for p in pred_boxes:
        pts = p["points"].astype(int).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=PRED_COLOR, thickness=PRED_THICKNESS)
        x, y = pts[0][0]
        cv2.putText(
            vis,
            f"{p['score']:.2f}",
            (int(x), max(int(y) - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            PRED_COLOR,
            1,
            cv2.LINE_AA,
        )

    legend_y = 25
    cv2.putText(vis, "GT", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(vis, "Pred", (60, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PRED_COLOR, 2, cv2.LINE_AA)
    return vis


# =====================================================
# ARGPARSE
# =====================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the PaddleOCR text DETECTOR only, on Dataset/val.txt."
    )
    p.add_argument("--val-file", type=Path, default=DEFAULT_VAL_FILE,
                    help="Path to val.txt (PPOCRLabel format). Default: Dataset/val.txt")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                    help="Root dir the paths in val.txt are relative to. Default: Dataset/")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Where to write metrics + visualizations. Default: eval_results/")
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                    help="PaddleOCR detection model name. Default: PP-OCRv5_mobile_det")
    p.add_argument("--model-dir", type=str, default=None,
                    help="Local path to a custom-trained/exported det model (optional).")
    p.add_argument("--device", type=str, default="cpu", help="cpu / gpu:0 / etc.")

    p.add_argument("--iou-thresh", type=float, default=0.5,
                    help="IoU threshold to count a prediction as matching a GT box.")
    p.add_argument("--score-thresh", type=float, default=0.0,
                    help="Extra post-hoc filter: drop predictions below this score "
                         "before matching (on top of the model's own box_thresh).")

    p.add_argument("--limit-side-len", type=int, default=None)
    p.add_argument("--limit-type", type=str, default=None, choices=[None, "min", "max"])
    p.add_argument("--thresh", type=float, default=None, help="Detector pixel score threshold.")
    p.add_argument("--box-thresh", type=float, default=None, help="Detector box score threshold.")
    p.add_argument("--unclip-ratio", type=float, default=None)

    p.add_argument("--limit", type=int, default=None,
                    help="Only evaluate the first N images (useful for a quick sanity check).")
    p.add_argument("--no-vis", action="store_true", help="Skip saving visualization images.")
    p.add_argument("--vis-count", type=int, default=None,
                    help="Only save visualizations for the first N evaluated images.")
    return p.parse_args()


# =====================================================
# MAIN
# =====================================================
def main():
    args = parse_args()

    entries = parse_label_file(args.val_file)
    if args.limit is not None:
        entries = entries[: args.limit]
    print(f"Loaded {len(entries)} images from {args.val_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Load detector (standalone module -> detection only, no recognition)
    # -------------------------------------------------
    from paddleocr import TextDetection

    model_kwargs = {"model_name": args.model_name, "device": args.device}
    if args.model_dir:
        model_kwargs["model_dir"] = args.model_dir
    detector = TextDetection(**model_kwargs)

    predict_kwargs = {}
    for key in ("limit_side_len", "limit_type", "thresh", "box_thresh", "unclip_ratio"):
        val = getattr(args, key)
        if val is not None:
            predict_kwargs[key] = val

    # -------------------------------------------------
    # Evaluate
    # -------------------------------------------------
    total_tp = total_fp = total_fn = 0
    all_matched_ious = []
    per_image_rows = []
    skipped = 0

    for idx, (img_rel, gt_boxes) in enumerate(entries):
        image_path = args.image_dir / img_rel
        if not image_path.exists():
            print(f"[warn] missing image, skipping: {image_path}")
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[warn] could not read image, skipping: {image_path}")
            skipped += 1
            continue

        outputs = detector.predict(input=str(image_path), batch_size=1, **predict_kwargs)
        dt_polys, dt_scores = [], []
        for res in outputs:
            data = res.json.get("res", res.json)
            dt_polys = data.get("dt_polys", [])
            dt_scores = data.get("dt_scores", [])
            break  # single image -> single result

        pred_boxes = []
        for poly, score in zip(dt_polys, dt_scores):
            score = float(score)
            if score < args.score_thresh:
                continue
            pred_boxes.append({"points": np.asarray(poly, dtype=np.float32), "score": score})

        match = match_predictions_to_gt(gt_boxes, pred_boxes, args.iou_thresh)
        total_tp += match["tp"]
        total_fp += match["fp"]
        total_fn += match["fn"]
        all_matched_ious.extend(match["matched_ious"])

        precision, recall, f1 = prf1(match["tp"], match["fp"], match["fn"])
        mean_iou = float(np.mean(match["matched_ious"])) if match["matched_ious"] else 0.0

        per_image_rows.append({
            "image": img_rel,
            "num_gt": sum(1 for g in gt_boxes if not g["ignore"]),
            "num_pred": len(pred_boxes),
            "tp": match["tp"],
            "fp": match["fp"],
            "fn": match["fn"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_iou": round(mean_iou, 4),
        })

        print(
            f"[{idx + 1}/{len(entries)}] {img_rel:30s} "
            f"GT={len(gt_boxes):3d} Pred={len(pred_boxes):3d} "
            f"TP={match['tp']:3d} FP={match['fp']:3d} FN={match['fn']:3d} "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

        save_this = (not args.no_vis) and (args.vis_count is None or idx < args.vis_count)
        if save_this:
            vis = draw_boxes(image, gt_boxes, pred_boxes)
            out_name = img_rel.replace("/", "_").replace("\\", "_")
            cv2.imwrite(str(vis_dir / out_name), vis)

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    precision, recall, f1 = prf1(total_tp, total_fp, total_fn)
    mean_iou_all = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    summary = {
        "model_name": args.model_name,
        "val_file": str(args.val_file),
        "iou_thresh": args.iou_thresh,
        "score_thresh": args.score_thresh,
        "num_images_evaluated": len(per_image_rows),
        "num_images_skipped": skipped,
        "total_gt_boxes": sum(r["num_gt"] for r in per_image_rows),
        "total_pred_boxes": sum(r["num_pred"] for r in per_image_rows),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou_matched": round(mean_iou_all, 4),
    }

    with open(args.output_dir / "detector_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if per_image_rows:
        with open(args.output_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)

    print()
    print("=" * 70)
    print("DETECTOR EVALUATION FINISHED")
    print("=" * 70)
    print(f"Images evaluated : {summary['num_images_evaluated']} (skipped: {skipped})")
    print(f"GT boxes         : {summary['total_gt_boxes']}")
    print(f"Predicted boxes  : {summary['total_pred_boxes']}")
    print(f"TP / FP / FN     : {total_tp} / {total_fp} / {total_fn}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1-score         : {f1:.4f}")
    print(f"Mean IoU (TP)    : {mean_iou_all:.4f}")
    print(f"Metrics JSON     : {args.output_dir / 'detector_metrics.json'}")
    print(f"Per-image CSV    : {args.output_dir / 'per_image_metrics.csv'}")
    if not args.no_vis:
        print(f"Visualizations   : {vis_dir}")


if __name__ == "__main__":
    main()