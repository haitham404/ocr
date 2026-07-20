# import json
# import cv2
# import numpy as np
# from pathlib import Path
# from paddleocr import PaddleOCR
# from paddleocr import TextDetection

# # =====================================================
# # CONFIGURATION
# # =====================================================
# BASE_DIR = Path(__file__).resolve().parent

# # NOTE: Check these paths!
# IMAGE_ROOT_DIR = BASE_DIR / "Dataset" # The folder containing 'card_042', etc.
# VAL_TXT_PATH = IMAGE_ROOT_DIR / "val.txt"
# OUTPUT_DIR = BASE_DIR / "eval_results"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Initialize PaddleOCR (Detector ONLY)
# ocr = TextDetection(
#     model_name="PP-OCRv5_mobile_det",
#     device="cpu",
# )

# # =====================================================
# # METRIC HELPER
# # =====================================================
# def calculate_iou(box1, box2, img_h, img_w):
#     """Calculates IoU by drawing polygons on a blank mask."""
#     mask1 = np.zeros((img_h, img_w), dtype=np.uint8)
#     mask2 = np.zeros((img_h, img_w), dtype=np.uint8)
    
#     cv2.fillPoly(mask1, [np.array(box1, dtype=np.int32)], 1)
#     cv2.fillPoly(mask2, [np.array(box2, dtype=np.int32)], 1)
    
#     intersection = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
    
#     return intersection / union if union > 0 else 0

# # =====================================================
# # EVALUATION LOOP
# # =====================================================
# def run_evaluation():
#     if not VAL_TXT_PATH.exists():
#         print(f"[DEBUG] Cannot find txt file at: {VAL_TXT_PATH}")
#         return

#     with open(VAL_TXT_PATH, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     # These MUST stay outside the loop
#     total_gt = 0
#     total_pred = 0
#     total_tp = 0

#     print(f"Evaluating {len(lines)} images...")

#     for line in lines:
#         if not line.strip():
#             continue
            
#         img_rel_path, labels_json = line.strip().split('\t')
#         image_path = IMAGE_ROOT_DIR / img_rel_path

#         if not image_path.exists():
#             print(f"[DEBUG] Missing image on disk: {image_path}")
#             continue

#         # Load Ground Truth
#         try:
#             gt_data = json.loads(labels_json)
#             gt_boxes = [item['points'] for item in gt_data]
#         except Exception as e:
#             print(f"[DEBUG] JSON parsing failed for {img_rel_path}: {e}")
#             gt_boxes = []
        
#         # Read Image
#         img = cv2.imread(str(image_path))
#         if img is None:
#             print(f"[DEBUG] cv2 failed to read image: {image_path}")
#             continue
            
#         img_h, img_w = img.shape[:2]
#         vis_img = img.copy()

#         # Get Predictions using V5 (Standard Pipeline)
#         # 1. Read the exact pixel array using OpenCV first
#         img_array = cv2.imread(str(image_path))

#         # 2. Pass the numpy array DIRECTLY to the predictor (not the file path)
#         results = list(ocr.predict(img_array))
        
#         pred_boxes = []
#         if results and isinstance(results[0], dict):
#             pred_dict = results[0]
            
#             # Check common PaddleOCR V5 keys
#             if 'dt_polys' in pred_dict:
#                 pred_boxes = pred_dict['dt_polys']
#             elif 'text_polys' in pred_dict:
#                 pred_boxes = pred_dict['text_polys']
#             elif 'polygons' in pred_dict:
#                 pred_boxes = pred_dict['polygons']
#             else:
#                 print(f"[DEBUG] No matching keys found! Model returned these keys: {list(pred_dict.keys())}")
#         elif results and isinstance(results[0], list):
#             pred_boxes = results[0]

#         # Accumulate Totals
#         total_gt += len(gt_boxes)
#         total_pred += len(pred_boxes)

#         # Calculate Matches (IoU > 0.5)
#         matched_preds = set()
#         for gt_box in gt_boxes:
#             best_iou = 0
#             best_pred_idx = -1
            
#             for i, pred_box in enumerate(pred_boxes):
#                 if i in matched_preds:
#                     continue
                
#                 iou = calculate_iou(gt_box, pred_box, img_h, img_w)
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_pred_idx = i
                    
#             if best_iou > 0.5:
#                 total_tp += 1
#                 matched_preds.add(best_pred_idx)

#         # Draw Boxes: Ground Truth (Green), Prediction (Red)
#         for box in gt_boxes:
#             pts = np.array(box, np.int32).reshape((-1, 1, 2))
#             cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
#         for box in pred_boxes:
#             pts = np.array(box, np.int32).reshape((-1, 1, 2))
#             cv2.polylines(vis_img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

#         # Save Visualization
#         save_name = img_rel_path.replace("/", "_")
#         cv2.imwrite(str(OUTPUT_DIR / save_name), vis_img)

#     # FINAL METRICS
#     precision = total_tp / total_pred if total_pred > 0 else 0
#     recall = total_tp / total_gt if total_gt > 0 else 0
#     f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

#     print("\n" + "=" * 50)
#     print("DETECTOR EVALUATION RESULTS")
#     print("=" * 50)
#     print(f"Total Ground Truth Boxes : {total_gt}")
#     print(f"Total Predicted Boxes    : {total_pred}")
#     print(f"True Positives (Matches) : {total_tp}")
#     print("-" * 50)
#     print(f"Precision : {precision:.4f}")
#     print(f"Recall    : {recall:.4f}")
#     print(f"F-Score   : {f_score:.4f}")
#     print("=" * 50)
#     print(f"Visualizations saved to: {OUTPUT_DIR}")

# ###############################################################################
import json
import cv2
import numpy as np
from pathlib import Path
from paddleocr import TextDetection

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

# NOTE: Check these paths!
IMAGE_ROOT_DIR = BASE_DIR / "Dataset" 
VAL_TXT_PATH = IMAGE_ROOT_DIR / "val.txt"
OUTPUT_DIR = BASE_DIR / "eval_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize PaddleOCR (Detector ONLY)
ocr = TextDetection(
    model_name="PP-OCRv5_mobile_det",
    device="cpu",
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
    Identifies the region based on your specific val.txt structure.
    """
    # 1. Identify National ID (14 digits)
    # Python's .isnumeric() natively understands Arabic numerals (e.g., ٢٤١٠١١٣٠١٢١٦٩٥)
    clean_text = transcription.replace(" ", "").replace("-", "")
    if len(clean_text) == 14 and clean_text.isnumeric():
        return 'id'
    
    # 2. Identify First Name
    # In your val.txt, the first name is consistently the first dictionary (index 0)
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
        lines = f.readlines()

    # Independent Metrics Tracking
    metrics = {
        'name': {'TP': 0, 'FN': 0, 'GT_Total': 0},
        'id':   {'TP': 0, 'FN': 0, 'GT_Total': 0}
    }

    print(f"Evaluating {len(lines)} images independently for First Name and ID...")

    for line in lines:
        if not line.strip():
            continue
            
        img_rel_path, labels_json = line.strip().split('\t')
        image_path = IMAGE_ROOT_DIR / img_rel_path

        if not image_path.exists():
            print(f"[DEBUG] Missing image on disk: {image_path}")
            continue

        # 1. Parse Ground Truth and filter for Name and ID using Index and Text
        gt_target_boxes = {}
        try:
            gt_data = json.loads(labels_json)
            for idx, item in enumerate(gt_data):
                region_type = identify_region(item.get('transcription', ''), idx)
                if region_type in ['name', 'id']:
                    gt_target_boxes[region_type] = item['points']
                    metrics[region_type]['GT_Total'] += 1
        except Exception as e:
            print(f"[DEBUG] JSON parsing failed for {img_rel_path}: {e}")
            continue
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[DEBUG] cv2 failed to read image: {image_path}")
            continue
            
        img_h, img_w = img.shape[:2]
        vis_img = img.copy()

        # 2. Get Predictions using V5
        results = list(ocr.predict(img))
        
        pred_boxes = []
        if results and isinstance(results[0], dict):
            pred_dict = results[0]
            if 'dt_polys' in pred_dict: pred_boxes = pred_dict['dt_polys']
            elif 'text_polys' in pred_dict: pred_boxes = pred_dict['text_polys']
            elif 'polygons' in pred_dict: pred_boxes = pred_dict['polygons']
        elif results and isinstance(results[0], list):
            pred_boxes = results[0]

        # 3. Calculate Matches per Region
        for region_type, gt_box in gt_target_boxes.items():
            matched = False
            
            for pred_box in pred_boxes:
                iou = calculate_iou(gt_box, pred_box, img_h, img_w)
                # If the prediction box overlaps the ground truth by more than 50%
                if iou > 0.5:
                    matched = True
                    break 
                    
            if matched:
                metrics[region_type]['TP'] += 1
            else:
                metrics[region_type]['FN'] += 1

            # Draw Ground Truths (Green for Target Regions)
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

    # 4. FINAL INDEPENDENT METRICS
    print("\n" + "=" * 50)
    print("REGION-SPECIFIC EVALUATION RESULTS")
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
if __name__ == "__main__":
    run_evaluation()