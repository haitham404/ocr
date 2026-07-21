"""
run_merged.py

OCR over Dataset/card_*/{front,back}.jpg using a two-stage pipeline:

    detect (TextDetection) -> recognize (TextRecognition)

Output: results/<card>/<side>.jpg + results/ocr_results.csv
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

REC_SCORE_THRESH = 0.70

DET_KWARGS = dict(
    limit_side_len=1600,
    limit_type="max",
    box_thresh=0.4,
    unclip_ratio=2.2,
)


def poly_to_bbox(poly):
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x1, y1 = poly[:, 0].min(), poly[:, 1].min()
    x2, y2 = poly[:, 0].max(), poly[:, 1].max()
    return float(x1), float(y1), float(x2), float(y2)


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

            vis = image.copy()

            # ---------------------------------------------
            # STAGE 2: RECOGNITION (per detection)
            # ---------------------------------------------
            for poly, det_score in zip(dt_polys, dt_scores):
                x1, y1, x2, y2 = poly_to_bbox(poly)
                x1 = max(int(x1) - 2, 0)
                y1 = max(int(y1) - 2, 0)
                x2 = min(int(x2) + 2, img_w)
                y2 = min(int(y2) + 2, img_h)
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
                print(f"{text} ({rec_score:.3f})")

                rows.append({
                    "card": card_name,
                    "side": side,
                    "image": str(image_path),
                    "text": text,
                    "score": rec_score,
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
    print("OCR FINISHED (detect -> recognize)")
    print("=" * 70)
    print(f"Cards          : {len(cards)}")
    print(f"Detections     : {len(rows)}")
    if latencies:
        print(f"Avg latency    : {np.mean(latencies):.3f} sec")
    print(f"CSV            : {csv_path}")
    print(f"Results        : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()