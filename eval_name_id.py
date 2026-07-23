import json
import time
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from paddleocr import TextDetection

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

# NOTE: Check these paths!
IMAGE_ROOT_DIR = BASE_DIR / "Actual_cards" 
VAL_TXT_PATH = IMAGE_ROOT_DIR / "val.txt"
OUTPUT_DIR = BASE_DIR / "eval_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. UPGRADED MODEL: Initializing PaddleOCR with the heavier Server model
# The server model natively handles complex layouts and challenging spacing better than mobile.
model_name = "PP-OCRv5_mobile_det"
ocr = TextDetection(
    model_name = model_name, 
    device="cpu",
    enable_mkldnn=True, 
    cpu_threads=12,              # ↑ from 6 for better throughput
    limit_side_len=960,
    limit_type="max",        # ← CRITICAL: actually downsize large images
)

# =====================================================
# METRIC HELPER
# =====================================================
def calculate_iou(box1, box2, img_h, img_w):
    """Calculates IoU by drawing polygons on a blank mask."""
    mask1 = np.zeros((img_h, img_w), dtype=np.uint8)
    mask2 = np.zeros((img_h, img_w), dtype=np.uint8)
    
    cv2.fillPoly(mask1, [np.array(box1, dtype=np.int32)], 1)
    cv2.fillPoly(mask2, [np.array(box2, dtype=np.int32)], 1)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0

def identify_region(transcription, index):
    """
    Identifies the region based on the val.txt structure.
    """
    clean_text = transcription.replace(" ", "").replace("-", "")
    if len(clean_text) == 14 and clean_text.isnumeric():
        return 'id'
    
    if index == 0:
        return 'name'
        
    return 'other'

# =====================================================
# EVALUATION LOOP
# =====================================================
def run_evaluation():
    if not VAL_TXT_PATH.exists():
        print(f"[DEBUG] Cannot find txt file at: {VAL_TXT_PATH}")
        return

    with open(VAL_TXT_PATH, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # --- FILTERING LOGIC ---
    lines = []
    for line in raw_lines:
        image_path = line.split('\t')[0].lower() 
        # if "back" in image_path:
        lines.append(line)
        print(line)

    metrics = {
        'name': {'TP': 0, 'FN': 0, 'GT_Total': 0},
        'id':   {'TP': 0, 'FN': 0, 'GT_Total': 0}
    }

    print(f"Evaluating {len(lines)} images independently using PP-OCRv5 Server...")

    # 2. PROGRESS TRACKING: Processing will be slower on CPU; tqdm provides visibility.
    for line in tqdm(lines, desc="Running Server Detection Validation"):
        if not line.strip():
            continue
            
        img_rel_path, labels_json = line.strip().split('\t')
        image_path = IMAGE_ROOT_DIR / img_rel_path

        if not image_path.exists():
            continue

        gt_target_boxes = {}
        try:
            gt_data = json.loads(labels_json)
            for idx, item in enumerate(gt_data):
                region_type = identify_region(item.get('transcription', ''), idx)
                if region_type in ['name', 'id']:
                    gt_target_boxes[region_type] = item['points']
                    # Only count ID GT when image contains "back"
                    if region_type == 'id' and "back" not in img_rel_path.lower():
                        continue
                    metrics[region_type]['GT_Total'] += 1
        except Exception:
            continue
        
        img = cv2.imread(str(image_path))
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        vis_img = img.copy()

        # Get Predictions using the Server Model (with per-image latency)
        t_start = time.time()
        results = list(ocr.predict(img))
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000
        tqdm.write(f"  [{img_rel_path}] Latency: {latency_ms:.1f}ms")
        
        pred_boxes = []
        if results and isinstance(results[0], dict):
            pred_dict = results[0]
            if 'dt_polys' in pred_dict: pred_boxes = pred_dict['dt_polys']
            elif 'text_polys' in pred_dict: pred_boxes = pred_dict['text_polys']
            elif 'polygons' in pred_dict: pred_boxes = pred_dict['polygons']
        elif results and isinstance(results[0], list):
            pred_boxes = results[0]

        for region_type, gt_box in gt_target_boxes.items():
            matched = False
            
            for pred_box in pred_boxes:
                iou = calculate_iou(gt_box, pred_box, img_h, img_w)
                
                # 3. STRICTER METRIC: Enforcing 0.55 IoU to demand geometric precision.
                if iou >= 0.55:
                    matched = True
                    break 
                    
            # Only update ID metrics for back images; name metrics always update
            if region_type != 'id' or "back" in img_rel_path.lower():
                if matched:
                    metrics[region_type]['TP'] += 1
                else:
                    metrics[region_type]['FN'] += 1

            # Draw Ground Truths (Green)
            pts = np.array(gt_box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(vis_img, region_type.upper(), (pts[0][0][0], pts[0][0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw Predictions (Red)
        for box in pred_boxes:
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

        # Save Visualization
        save_name = img_rel_path.replace("/", "_")
        cv2.imwrite(str(OUTPUT_DIR / save_name), vis_img)

    # FINAL INDEPENDENT METRICS
    print("\n" + "=" * 50)
    print("REGION-SPECIFIC EVALUATION RESULTS (STRICT IOU >= 0.55)")
    print("=" * 50)
    
    for region in ['name', 'id']:
        tp = metrics[region]['TP']
        fn = metrics[region]['FN']
        total_gt = metrics[region]['GT_Total']
        
        recall = tp / total_gt if total_gt > 0 else 0
        
        print(f"Region: {region.upper()}")
        print(f"  Target Boxes in GT: {total_gt}")
        print(f"  Successful Matches: {tp}")
        print(f"  Missed / Dropped:   {fn}")
        print(f"  --> RECALL:         {recall:.4f}")
        print("-" * 50)
        print(f"Model Name {model_name}")

if __name__ == "__main__":
    run_evaluation()