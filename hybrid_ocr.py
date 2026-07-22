#!/usr/bin/env python3
"""
hybrid_ocr.py — Two-pass YOLO + PaddleOCR hybrid pipeline.

Pass 1 — YOLO field detection only:
    python hybrid_ocr.py --pass1 --input-dir Actual_cards

Pass 2 — PaddleOCR recognition (also handles back cards via PaddleOCR fallback):
    python hybrid_ocr.py --pass2 --input-dir Actual_cards

Output:
    results/hybrid_results.csv
    results/visualizations/<image>.jpg
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from collections import OrderedDict

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "Actual_cards"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results"
CROP_DIR = BASE_DIR / ".hybrid_crops"

DETECT_OBJECTS_PATH = BASE_DIR / "detect_odjects.pt"
DETECT_ID_PATH = BASE_DIR / "detect_id.pt"
REC_MODEL_NAME = "arabic_PP-OCRv5_mobile_rec"
REC_SCORE_THRESH = 0.50
FIELD_CLASSES = ["firstName", "lastName", "serial", "address", "nid"]


# ═══════════════════════════════════════════════════════════════════════════
# PASS 1 — YOLO Detection (no PaddleOCR loaded in this process)
# ═══════════════════════════════════════════════════════════════════════════

def pass1_yolo(args):
    import cv2
    import numpy as np
    from ultralytics import YOLO

    if not DETECT_OBJECTS_PATH.exists():
        print(f"[ERROR] YOLO model not found: {DETECT_OBJECTS_PATH}")
        sys.exit(1)

    if CROP_DIR.exists():
        shutil.rmtree(CROP_DIR)

    use_nid_yolo = not args.rec_nid and DETECT_ID_PATH.exists()

    print("Loading YOLO field detector...")
    field_model = YOLO(str(DETECT_OBJECTS_PATH))

    nid_model = None
    if use_nid_yolo:
        print("Loading YOLO NID digit detector...")
        nid_model = YOLO(str(DETECT_ID_PATH))

    images = discover_images(args.input_dir)
    if not images:
        print(f"[ERROR] No images found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images\n")

    all_meta = {}
    yolo_latencies = []

    for idx, img_path in enumerate(images):
        print(f"[{idx+1}/{len(images)}] {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print("  [SKIP] Cannot read")
            continue

        img_h, img_w = image.shape[:2]

        t0 = time.perf_counter()
        yolo_out = field_model(img_path, verbose=False)
        t1 = time.perf_counter()
        yolo_latency = t1 - t0
        yolo_latencies.append(yolo_latency)

        detections = []
        for result in yolo_out:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 > x1 and y2 > y1:
                    detections.append(OrderedDict(name=name, bbox=(x1, y1, x2, y2)))

        # Store entry even if YOLO found nothing (back cards, etc.)
        img_key = img_path.stem
        meta_fields = {}

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = image[y1:y2, x1:x2]

            field_path = CROP_DIR / img_key / f"{det['name']}.jpg"
            field_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(field_path), crop)

            meta_fields[det["name"]] = {
                "bbox": [x1, y1, x2, y2],
                "crop_path": str(field_path),
            }

            # For NID: expand BOTH width and height so digits at edges aren't cut off
            if det["name"] == "nid" and nid_model is not None:
                ex1, ey1, ex2, ey2 = expand_bbox_wh(
                    (x1, y1, x2, y2), 1.5, 1.2, img_h, img_w
                )
                nid_crop = image[ey1:ey2, ex1:ex2]
                nid_path = CROP_DIR / img_key / "nid_expanded.jpg"
                cv2.imwrite(str(nid_path), nid_crop)
                meta_fields["nid"]["nid_crop_path"] = str(nid_path)

                # Run digit detection now (YOLO on YOLO is safe)
                digits = recognize_nid_digits(nid_crop, nid_model)
                meta_fields["nid"]["nid_text"] = digits

        all_meta[img_key] = {
            "image_path": str(img_path),
            "fields": meta_fields,
        }

        if detections:
            names = ", ".join(meta_fields.keys())
            print(f"  Detected: {names}  [YOLO: {yolo_latency*1000:.0f}ms]")
        else:
            print(f"  No YOLO fields (may be back card)  [YOLO: {yolo_latency*1000:.0f}ms]")

    # Save metadata
    meta_path = CROP_DIR / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print(f"\nPass 1 done. Metadata saved to {meta_path}")
    print(f"Cropped fields saved to {CROP_DIR}/")
    if yolo_latencies:
        print(f"\nYOLO Latency per image:")
        print(f"  Avg: {np.mean(yolo_latencies)*1000:.0f}ms")
        print(f"  Min: {np.min(yolo_latencies)*1000:.0f}ms")
        print(f"  Max: {np.max(yolo_latencies)*1000:.0f}ms")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# PASS 2 — PaddleOCR Recognition (no YOLO loaded)
#   For back cards where YOLO found nothing, runs full PaddleOCR pipeline
# ═══════════════════════════════════════════════════════════════════════════

def pass2_ocr(args):
    import cv2
    import numpy as np
    from paddleocr import TextRecognition, TextDetection

    meta_path = CROP_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"[ERROR] No metadata found at {meta_path}")
        print("Run Pass 1 first: python hybrid_ocr.py --pass1 --input-dir Actual_cards")
        sys.exit(1)

    with open(meta_path) as f:
        all_meta = json.load(f)

    print("Loading PaddleOCR TextRecognition recognizer...")
    recognizer = TextRecognition(model_name=REC_MODEL_NAME, device=args.device)

    # Load detector only for back-card fallback
    print("Loading PaddleOCR TextDetection (for back card fallback)...")
    detector = TextDetection(model_name="arabic_PP-OCRv5_mobile_rec", device=args.device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    rows = []
    latencies = []
    DET_KWARGS = dict(limit_side_len=1600, limit_type="max",
                      box_thresh=0.4, unclip_ratio=2.2)

    for img_key, meta in sorted(all_meta.items()):
        print(f"\n[{img_key}]")

        image = cv2.imread(meta["image_path"])
        if image is None:
            print("  [SKIP] Cannot read source image")
            continue

        img_h, img_w = image.shape[:2]
        t0 = time.perf_counter()
        field_results = OrderedDict()

        if meta["fields"]:
            # ── Normal: YOLO found fields, recognise saved crops ──
            for field_name in FIELD_CLASSES:
                if field_name not in meta["fields"]:
                    continue
                field_info = meta["fields"][field_name]
                bbox = tuple(field_info["bbox"])

                if field_name == "nid" and "nid_text" in field_info:
                    text = field_info["nid_text"]
                    score = 1.0
                else:
                    crop = cv2.imread(field_info["crop_path"])
                    if crop is None:
                        continue
                    text, score = recognize_paddle(crop, recognizer)

                field_results[field_name] = dict(text=text, score=score, bbox=bbox)
        else:
            # ── Fallback: YOLO found nothing (back card) ──
            # Run PaddleOCR detection + recognition on the whole image
            det_output = detector.predict(input=meta["image_path"],
                                          batch_size=1, **DET_KWARGS)
            dt_polys, dt_scores = [], []
            for res in det_output:
                data = res.json.get("res", res.json)
                dt_polys = data.get("dt_polys", [])
                dt_scores = data.get("dt_scores", [])
                break

            for i, (poly, det_score) in enumerate(zip(dt_polys, dt_scores)):
                if float(det_score) < 0.3:
                    continue
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                x1 = int(max(pts[:, 0].min() - 2, 0))
                y1 = int(max(pts[:, 1].min() - 2, 0))
                x2 = int(min(pts[:, 0].max() + 2, img_w))
                y2 = int(min(pts[:, 1].max() + 2, img_h))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]
                text, score = recognize_paddle(crop, recognizer)
                if not text:
                    continue

                field_name = f"text_{i}"
                field_results[field_name] = dict(text=text, score=score,
                                                 bbox=(x1, y1, x2, y2))

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        print(f"  {elapsed:.2f}s")
        for name, data in field_results.items():
            label = data["text"] if data["text"] else "(empty)"
            print(f"    {name:12s} → {label}  ({data['score']:.3f})")

        # CSV row — add dynamic text_N columns for back card fallback
        row = OrderedDict(image=meta["image_path"], latency=f"{elapsed:.3f}")
        for name in FIELD_CLASSES:
            if name in field_results:
                row[f"{name}_text"] = field_results[name]["text"]
                row[f"{name}_score"] = f"{field_results[name]['score']:.3f}"
            else:
                row[f"{name}_text"] = ""
                row[f"{name}_score"] = ""
        # Add any extra fallback fields
        for name, data in field_results.items():
            if name.startswith("text_"):
                row[f"{name}_text"] = data["text"]
                row[f"{name}_score"] = f"{data['score']:.3f}"
        rows.append(row)

        # Visualisation
        vis = image.copy()
        for name, data in field_results.items():
            if not data["text"]:
                continue
            x1, y1, x2, y2 = data["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name}: {data['text']} ({data['score']:.2f})"
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        out_name = img_key + ".jpg"
        cv2.imwrite(str(vis_dir / out_name), vis)

    # CSV
    csv_path = output_dir / "hybrid_results.csv"
    if rows:
        import csv as csv_writer
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv_writer.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 60)
    print("PASS 2 — FINISHED")
    print("=" * 60)
    print(f"  Images          : {len(rows)}")
    total = sum(1 for r in rows for k, v in r.items()
                if k.endswith("_text") and v)
    print(f"  Fields detected : {total}")
    if latencies:
        print(f"  Avg latency     : {np.mean(latencies):.3f}s")
    print(f"  CSV             : {csv_path}")
    print(f"  Visualisations  : {vis_dir}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def expand_bbox_wh(bbox, scale_h: float, scale_w: float,
                   img_h: int, img_w: int):
    """Expand bounding box both height and width."""
    x1, y1, x2, y2 = bbox
    cy = (y1 + y2) // 2
    cx = (x1 + x2) // 2
    new_h = int((y2 - y1) * scale_h)
    new_w = int((x2 - x1) * scale_w)
    ny1 = max(cy - new_h // 2, 0)
    ny2 = min(cy + new_h // 2, img_h)
    nx1 = max(cx - new_w // 2, 0)
    nx2 = min(cx + new_w // 2, img_w)
    return nx1, ny1, nx2, ny2


def recognize_paddle(crop, recognizer) -> tuple:
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return "", 0.0
    output = recognizer.predict(input=crop, batch_size=1)
    for r in output:
        data = r.json.get("res", r.json)
        text = data.get("rec_text", "").strip()
        score = float(data.get("rec_score", 0.0))
        if text:
            return text, score
    return "", 0.0


def recognize_nid_digits(crop, nid_model) -> str:
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return ""
    results = nid_model(crop, verbose=False)
    digits = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, _, _, _ = map(int, box.xyxy[0])
            digits.append((x1, str(cls_id)))
    digits.sort(key=lambda x: x[0])
    return "".join(d[1] for d in digits)


def discover_images(input_dir: Path):
    images = []
    card_dirs = sorted(input_dir.glob("card_*"))
    if card_dirs:
        for cd in card_dirs:
            for side in ("front", "back"):
                p = cd / f"{side}.jpg"
                if p.exists():
                    images.append(p)
    else:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(input_dir.glob(ext))
        images.sort()
    return images


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Two-pass YOLO + PaddleOCR hybrid OCR pipeline"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rec-nid", action="store_true",
                        help="Use PaddleOCR for NID instead of YOLO detect_id.pt")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pass1", action="store_true",
                      help="Run Pass 1 only: YOLO detection")
    mode.add_argument("--pass2", action="store_true",
                      help="Run Pass 2 only: PaddleOCR recognition")

    args = parser.parse_args()

    if args.pass1:
        pass1_yolo(args)
    elif args.pass2:
        pass2_ocr(args)
    else:
        print("Running single-process (may segfault if PyTorch vs PaddlePaddle conflict).")
        print("If it crashes, use:")
        print("  python hybrid_ocr.py --pass1 --input-dir Actual_cards")
        print("  python hybrid_ocr.py --pass2 --input-dir Actual_cards")
        print()
        from paddleocr import TextRecognition
        from ultralytics import YOLO
        _run_single(args, TextRecognition, YOLO)


def _run_single(args, TextRecognition, YOLO):
    """Single-process mode (both libs loaded — may segfault)."""
    import cv2
    import numpy as np
    import csv as csv_writer
    from paddleocr import TextDetection

    if not DETECT_OBJECTS_PATH.exists():
        print(f"[ERROR] YOLO model not found: {DETECT_OBJECTS_PATH}")
        sys.exit(1)

    use_nid = not args.rec_nid and DETECT_ID_PATH.exists()

    print("Loading YOLO field detector...")
    field_model = YOLO(str(DETECT_OBJECTS_PATH))
    nid_model = YOLO(str(DETECT_ID_PATH)) if use_nid else None

    print("Loading PaddleOCR recognizer + detector...")
    recognizer = TextRecognition(model_name=REC_MODEL_NAME, device=args.device)
    detector = TextDetection(model_name="PP-OCRv5_mobile_det", device=args.device)

    images = discover_images(args.input_dir)
    if not images:
        print(f"[ERROR] No images found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images\n")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    rows = []
    latencies = []
    DET_KWARGS = dict(limit_side_len=1600, limit_type="max",
                      box_thresh=0.4, unclip_ratio=2.2)

    for idx, img_path in enumerate(images):
        print(f"[{idx+1}/{len(images)}] {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            print("  [SKIP]")
            continue
        img_h, img_w = image.shape[:2]
        t0 = time.perf_counter()

        yolo_out = field_model(img_path, verbose=False)
        detections = []
        for result in yolo_out:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = result.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 > x1 and y2 > y1:
                    detections.append(OrderedDict(name=name, bbox=(x1, y1, x2, y2)))

        field_results = OrderedDict()

        if detections:
            # YOLO found fields — recognise crops
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                crop = image[y1:y2, x1:x2]
                if det["name"] == "nid" and nid_model is not None:
                    ex1, ey1, ex2, ey2 = expand_bbox_wh(
                        (x1, y1, x2, y2), 1.5, 1.2, img_h, img_w
                    )
                    text = recognize_nid_digits(image[ey1:ey2, ex1:ex2], nid_model)
                    score = 1.0
                else:
                    text, score = recognize_paddle(crop, recognizer)
                field_results[det["name"]] = dict(text=text, score=score,
                                                  bbox=det["bbox"])
        else:
            # Fallback: full PaddleOCR for back cards
            det_output = detector.predict(input=str(img_path),
                                          batch_size=1, **DET_KWARGS)
            dt_polys, dt_scores = [], []
            for res in det_output:
                data = res.json.get("res", res.json)
                dt_polys = data.get("dt_polys", [])
                dt_scores = data.get("dt_scores", [])
                break
            for i, (poly, det_score) in enumerate(zip(dt_polys, dt_scores)):
                if float(det_score) < 0.3:
                    continue
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                x1 = int(max(pts[:, 0].min() - 2, 0))
                y1 = int(max(pts[:, 1].min() - 2, 0))
                x2 = int(min(pts[:, 0].max() + 2, img_w))
                y2 = int(min(pts[:, 1].max() + 2, img_h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                text, score = recognize_paddle(crop, recognizer)
                if not text:
                    continue
                field_results[f"text_{i}"] = dict(text=text, score=score,
                                                  bbox=(x1, y1, x2, y2))

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        print(f"  {elapsed:.2f}s")
        for name, data in field_results.items():
            print(f"    {name:12s} → {data['text']}  ({data['score']:.3f})")

        row = OrderedDict(image=str(img_path), latency=f"{elapsed:.3f}")
        for name in FIELD_CLASSES:
            if name in field_results:
                row[f"{name}_text"] = field_results[name]["text"]
                row[f"{name}_score"] = f"{field_results[name]['score']:.3f}"
            else:
                row[f"{name}_text"] = ""
                row[f"{name}_score"] = ""
        for name, data in field_results.items():
            if name.startswith("text_"):
                row[f"{name}_text"] = data["text"]
                row[f"{name}_score"] = f"{data['score']:.3f}"
        rows.append(row)

        vis = image.copy()
        for name, data in field_results.items():
            if not data["text"]:
                continue
            x1, y1, x2, y2 = data["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name}: {data['text']} ({data['score']:.2f})"
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(vis_dir / img_path.stem + ".jpg"), vis)

    csv_path = output_dir / "hybrid_results.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv_writer.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 60)
    print("HYBRID OCR — FINISHED")
    print("=" * 60)
    print(f"  Images          : {len(rows)}")
    total = sum(1 for r in rows for k, v in r.items()
                if k.endswith("_text") and v)
    print(f"  Fields detected : {total}")
    if latencies:
        print(f"  Avg latency     : {np.mean(latencies):.3f}s")
    print(f"  CSV             : {csv_path}")
    print(f"  Visualisations  : {vis_dir}")


if __name__ == "__main__":
    main()
