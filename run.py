import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from paddleocr import PaddleOCR


# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "Dataset"
OUTPUT_DIR = BASE_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD MODEL
# =====================================================

ocr = PaddleOCR(
    lang="ar",
    device="cpu",

    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",

    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,

    enable_mkldnn=False,
)


rows = []
latencies = []


# =====================================================
# FIND CARDS
# =====================================================

cards = sorted(
    path
    for path in IMAGE_DIR.glob("card_*")
    if path.is_dir()
)

print(f"Found {len(cards)} cards")


# =====================================================
# PROCESS DATASET
# =====================================================

for card_dir in cards:

    card_name = card_dir.name

    print()
    print("#" * 70)
    print(f"CARD: {card_name}")
    print("#" * 70)

    # Output folder per card
    card_output_dir = OUTPUT_DIR / card_name
    card_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

       
        # =====================================================
        # OCR
        # =====================================================

        t0 = time.perf_counter()

        result = ocr.predict(
        str(image_path),
        text_det_limit_side_len=1600,
        text_det_limit_type="max",
        text_det_box_thresh=0.4,      # lower = keep more low-confidence tail digits
        text_det_unclip_ratio=2.2,    # higher = expand box further past the shrunk core
        text_det_use_dilation=True,
        )

        latency = time.perf_counter() - t0

        latencies.append(latency)

        print(f"Latency: {latency:.3f} sec")


        if not result:
            print("No OCR result")
            continue


        # =====================================================
        # RESULTS
        # =====================================================

        res = result[0]

        boxes = res.get("rec_boxes", [])
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])

        vis = image.copy()


        # =====================================================
        # DRAW BOXES
        # =====================================================

        for box, text, score in zip(
            boxes,
            texts,
            scores,
        ):
           
            score = float(score)

            if score < 0.70:
                continue

            box = np.asarray(box).astype(int)

            if box.ndim == 2:

                x1 = int(box[:, 0].min())
                y1 = int(box[:, 1].min())

                x2 = int(box[:, 0].max())
                y2 = int(box[:, 1].max())

            elif box.ndim == 1 and len(box) == 4:

                x1, y1, x2, y2 = map(
                    int,
                    box,
                )

            else:

                print(
                    "Unknown box format:",
                    box,
                )

                continue

            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            rows.append(
                {
                    "card": card_name,
                    "side": side,
                    "image": str(image_path),
                    "text": text,
                    "score": score,
                    "latency": latency,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

            print(f"{text} ({score:.3f})")


        # =====================================================
        # SAVE VISUALIZATION
        # =====================================================

        output_image = (
            card_output_dir
            / f"{side}.jpg"
        )

        cv2.imwrite(
            str(output_image),
            vis,
        )

        print(
            f"Saved: {output_image}"
        )


# =====================================================
# SAVE CSV
# =====================================================

csv_path = OUTPUT_DIR / "ocr_results.csv"

df = pd.DataFrame(rows)

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig",
)


# =====================================================
# SUMMARY
# =====================================================

print()
print("=" * 70)
print("OCR FINISHED")
print("=" * 70)

print(f"Cards       : {len(cards)}")
print(f"Detections  : {len(rows)}")

if latencies:

    print(
        f"Avg latency : "
        f"{np.mean(latencies):.3f} sec"
    )

print(f"CSV          : {csv_path}")
print(f"Results      : {OUTPUT_DIR}")