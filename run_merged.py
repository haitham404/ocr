"""
run_merged.py

Same job as run.py (OCR over Dataset/card_*/{front,back}.jpg), but replaces
the single "PaddleOCR full pipeline" call with a two-stage flow:

    detect (TextDetection) -> merge same-line fragments -> recognize (TextRecognition)

Why: the detector sometimes splits one long line (e.g. the national ID
number) into 2+ separate boxes with a gap of un-boxed digits in between.
Raising unclip_ratio/lowering thresh only pushes each fragment's own edges
outward -- it can't rejoin two already-separate connected components. So
instead of fighting the probability map, we fix it geometrically after
detection: boxes that sit on the same text line and are close enough
horizontally get unioned into one box *before* recognition ever sees them.

Output layout matches run.py: results/<card>/<side>.jpg + results/ocr_results.csv
(with one extra column, n_fragments, showing how many raw detections were
merged into that row -- handy for spotting which fields fragment most).
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from paddleocr import TextDetection, TextRecognition

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REC_SCORE_THRESH = 0.70  # same cutoff run.py used before drawing/saving a box

# Detector post-processing overrides that fixed the truncation issue.
# thresh/use_dilation didn't help the fragmentation itself, so they're
# left at model defaults (commented out) here; flip them back on if you
# want to re-test them alongside merging.
DET_KWARGS = dict(
    limit_side_len=1600,
    limit_type="max",
    box_thresh=0.4,
    unclip_ratio=2.2,
    # thresh=0.2,
    # use_dilation=True,
)

# Merge heuristic tuning (see merge_same_line_boxes below).
GAP_RATIO = 2.0        # max horizontal gap between fragments, as a multiple of line height
Y_CENTER_RATIO = 0.6   # max vertical-center drift between fragments, as a multiple of line height
CROP_PAD = 4           # pixels of padding added around a merged box before cropping for recognition


# =====================================================
# GEOMETRY: same-line fragment merging
# =====================================================
def poly_to_bbox(poly):
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x1, y1 = poly[:, 0].min(), poly[:, 1].min()
    x2, y2 = poly[:, 0].max(), poly[:, 1].max()
    return float(x1), float(y1), float(x2), float(y2)


def merge_same_line_boxes(dt_polys, dt_scores, gap_ratio=GAP_RATIO, y_center_ratio=Y_CENTER_RATIO):
    """
    Groups detections that lie on the same text line and are close enough
    horizontally, then returns one union box per group.

    Returns a list of dicts: {"bbox": (x1,y1,x2,y2), "score": float, "n_fragments": int}
    sorted left-to-right, one per merged group (a group of size 1 just
    passes its original box through unchanged).
    """
    boxes = []
    for poly, score in zip(dt_polys, dt_scores):
        x1, y1, x2, y2 = poly_to_bbox(poly)
        boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "h": max(y2 - y1, 1e-6),
            "cy": (y1 + y2) / 2.0,
            "score": float(score),
        })
    boxes.sort(key=lambda b: b["x1"])

    lines = []  # list of open clusters, each a list of boxes in x-order
    for b in boxes:
        best_line = None
        best_gap = None
        for line in lines:
            last = line[-1]
            avg_h = (last["h"] + b["h"]) / 2.0
            same_line = abs(b["cy"] - last["cy"]) < y_center_ratio * avg_h
            gap = b["x1"] - last["x2"]
            close_enough = gap < gap_ratio * avg_h
            if same_line and close_enough:
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_line = line
        if best_line is not None:
            best_line.append(b)
        else:
            lines.append([b])

    merged = []
    for line in lines:
        x1 = min(bb["x1"] for bb in line)
        y1 = min(bb["y1"] for bb in line)
        x2 = max(bb["x2"] for bb in line)
        y2 = max(bb["y2"] for bb in line)
        score = min(bb["score"] for bb in line)  # conservative: weakest fragment wins
        merged.append({"bbox": (x1, y1, x2, y2), "score": score, "n_fragments": len(line)})

    merged.sort(key=lambda m: m["bbox"][0])
    return merged


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    # LOAD MODELS (two standalone modules instead of one full pipeline)
    detector = TextDetection(
        model_name="PP-OCRv5_server_det",
        device="cpu",
    )
    recognizer = TextRecognition(
        model_name="arabic_PP-OCRv5_mobile_rec",
        device="cpu",
    )

    rows = []
    latencies = []

    # FIND CARDS
    cards = sorted(path for path in IMAGE_DIR.glob("card_*") if path.is_dir())
    print(f"Found {len(cards)} cards")

    # PROCESS DATASET
    for card_dir in cards:
        card_name = card_dir.name
        print()
        print("#" * 70)
        print(f"CARD: {card_name}")
        print("#" * 70)

        card_output_dir = OUTPUT_DIR / card_name
        card_output_dir.mkdir(parents=True, exist_ok=True)

        for side in ["front", "back"]:
            image_path = card_dir / f"{side}.jpg"
            if not image_path.exists():
                print(f"Missing: {image_path}")
                continue

            print()
            print("=" * 70)
            print(f"Image: {image_path}")
            print("=" * 70)

            image = cv2.imread(str(image_path))
            if image is None:
                print("Cannot read image")
                continue
            img_h, img_w = image.shape[:2]

            # ---------------------------------------------
            # STAGE 1: DETECTION
            # ---------------------------------------------
            t0 = time.perf_counter()
            det_output = detector.predict(input=str(image_path), batch_size=1, **DET_KWARGS)
            dt_polys, dt_scores = [], []
            for res in det_output:
                data = res.json.get("res", res.json)
                dt_polys = data.get("dt_polys", [])
                dt_scores = data.get("dt_scores", [])
                break

            # ---------------------------------------------
            # MERGE same-line fragments
            # ---------------------------------------------
            merged_boxes = merge_same_line_boxes(dt_polys, dt_scores)

            vis = image.copy()

            # ---------------------------------------------
            # STAGE 2: RECOGNITION (per merged box)
            # ---------------------------------------------
            for m in merged_boxes:
                x1, y1, x2, y2 = m["bbox"]
                x1 = max(int(x1) - CROP_PAD, 0)
                y1 = max(int(y1) - CROP_PAD, 0)
                x2 = min(int(x2) + CROP_PAD, img_w)
                y2 = min(int(y2) + CROP_PAD, img_h)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]
                rec_output = recognizer.predict(input=crop, batch_size=1)
                text, rec_score = "", 0.0
                for r in rec_output:
                    rdata = r.json.get("res", r.json)
                    text = rdata.get("rec_text", "")
                    rec_score = float(rdata.get("rec_score", 0.0))
                    break

                if rec_score < REC_SCORE_THRESH:
                    continue

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print(f"{text} ({rec_score:.3f}) [{m['n_fragments']} fragment(s) merged]")

                rows.append({
                    "card": card_name,
                    "side": side,
                    "image": str(image_path),
                    "text": text,
                    "score": rec_score,
                    "n_fragments": m["n_fragments"],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })

            latency = time.perf_counter() - t0
            latencies.append(latency)
            print(f"Latency: {latency:.3f} sec")

            # ---------------------------------------------
            # SAVE VISUALIZATION
            # ---------------------------------------------
            output_image = card_output_dir / f"{side}.jpg"
            cv2.imwrite(str(output_image), vis)
            print(f"Saved: {output_image}")

    # SAVE CSV
    csv_path = OUTPUT_DIR / "ocr_results.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # SUMMARY
    print()
    print("=" * 70)
    print("OCR FINISHED (detect -> merge -> recognize)")
    print("=" * 70)
    print(f"Cards          : {len(cards)}")
    print(f"Detections     : {len(rows)}")
    print(f"Multi-fragment : {sum(1 for r in rows if r['n_fragments'] > 1)} rows needed merging")
    if latencies:
        print(f"Avg latency    : {np.mean(latencies):.3f} sec")
    print(f"CSV            : {csv_path}")
    print(f"Results        : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()