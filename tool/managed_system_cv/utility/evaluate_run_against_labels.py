import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---
LABEL_DIR = "data/bdd100k/labels/train"
PRED_DIR = "runs_artifact/knowledge_07_13:20:51_harmone/inferences"
CONFIDENCE_THRESHOLD_FOR_P_R_F1 = 0.5 # Confidence threshold for calculating overall P, R, F1

# --- HELPER FUNCTIONS ---

def yolo_to_corners(box):
    """Convert YOLO box format [center_x, center_y, w, h] to corners [x1, y1, x2, y2]."""
    x_center, y_center, w, h = box
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return np.array([x1, y1, x2, y2])

def calculate_iou(box1_yolo, box2_yolo):
    """Calculate Intersection over Union (IoU) for two boxes in YOLO format."""
    box1 = yolo_to_corners(box1_yolo)
    box2 = yolo_to_corners(box2_yolo)

    # Calculate intersection coordinates
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def read_boxes(filepath, with_conf=False):
    """Reads a YOLO label file and returns a list of boxes."""
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            class_id = int(parts[0])
            box_coords = [float(p) for p in parts[1:5]]
            if with_conf:
                conf = float(parts[5])
                boxes.append([class_id, conf, box_coords])
            else:
                boxes.append([class_id, box_coords])
    return boxes

def calculate_ap(recall, precision):
    """Calculate Average Precision (AP) from recall and precision arrays."""
    # Append sentinel values for correct integration
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # Sum the area of the rectangles
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# --- MAIN SCRIPT ---

def main():
    print("Starting evaluation...")
    print(f"Labels directory: {LABEL_DIR}")
    print(f"Predictions directory: {PRED_DIR}")

    # 1. Load all ground truths and predictions into memory
    ground_truths = defaultdict(list)
    predictions = []
    
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    if not label_files:
        print(f"Error: No label files found in {LABEL_DIR}. Aborting.")
        return

    print("\nStep 1/3: Loading ground truths and predictions...")
    for filename in tqdm(label_files):
        image_id = filename.split('.')[0]
        
        # Load ground truths
        gt_boxes = read_boxes(os.path.join(LABEL_DIR, filename), with_conf=False)
        for class_id, box in gt_boxes:
            ground_truths[image_id].append({'class_id': class_id, 'box': box, 'used': False})
            
        # Load predictions
        pred_boxes = read_boxes(os.path.join(PRED_DIR, filename), with_conf=True)
        for class_id, conf, box in pred_boxes:
            predictions.append({'image_id': image_id, 'class_id': class_id, 'confidence': conf, 'box': box})

    # Get all unique class IDs
    all_gt_classes = {gt['class_id'] for gts in ground_truths.values() for gt in gts}
    all_pred_classes = {p['class_id'] for p in predictions}
    class_ids = sorted(list(all_gt_classes.union(all_pred_classes)))
    num_classes = len(class_ids)
    
    if num_classes == 0:
        print("Error: No classes found in labels or predictions. Aborting.")
        return

    print(f"Found {len(ground_truths)} images with ground truths.")
    print(f"Found {len(predictions)} total predictions across {num_classes} classes.")

    # 2. Calculate metrics for different IoU thresholds
    print("\nStep 2/3: Calculating AP for each class and IoU threshold...")
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    ap_results = defaultdict(dict)
    
    # Global stats for overall P/R/F1 at a fixed IoU
    total_tp_50 = 0
    total_fp_50 = 0
    total_fn_50 = 0

    for iou_thresh in tqdm(iou_thresholds, desc="IoU Thresholds"):
        # Reset 'used' status for all ground truths for each IoU run
        for gts in ground_truths.values():
            for gt in gts:
                gt['used'] = False
        
        # Sort predictions by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Dictionaries to store TP/FP for each class
        tps = defaultdict(list)
        fps = defaultdict(list)
        
        if not predictions: continue

        for pred in predictions:
            image_gts = ground_truths.get(pred['image_id'], [])
            class_gts = [gt for gt in image_gts if gt['class_id'] == pred['class_id']]
            
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(class_gts):
                iou = calculate_iou(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_thresh and best_gt_idx != -1 and not class_gts[best_gt_idx]['used']:
                tps[pred['class_id']].append(1)
                fps[pred['class_id']].append(0)
                class_gts[best_gt_idx]['used'] = True
            else:
                tps[pred['class_id']].append(0)
                fps[pred['class_id']].append(1)

        # Calculate AP for each class at this IoU threshold
        for cid in class_ids:
            num_gt_for_class = sum(1 for gts in ground_truths.values() for gt in gts if gt['class_id'] == cid)
            
            if num_gt_for_class == 0:
                ap_results[cid][iou_thresh] = 0.0
                continue

            class_preds = [p for p in predictions if p['class_id'] == cid]
            if not class_preds:
                ap_results[cid][iou_thresh] = 0.0
                continue

            # Sort TP/FP lists according to original confidence sort
            class_tps = np.array(tps[cid])
            class_fps = np.array(fps[cid])
            
            cum_tps = np.cumsum(class_tps)
            cum_fps = np.cumsum(class_fps)
            
            recall = cum_tps / num_gt_for_class
            precision = cum_tps / (cum_tps + cum_fps)
            
            ap = calculate_ap(recall, precision)
            ap_results[cid][iou_thresh] = ap

    # 3. Calculate final metrics and display results
    print("\nStep 3/3: Aggregating and displaying results...")
    
    # Calculate overall P, R, F1 at IoU=0.5 and a fixed confidence
    num_total_gts = sum(len(gts) for gts in ground_truths.values())
    preds_above_conf = [p for p in predictions if p['confidence'] >= CONFIDENCE_THRESHOLD_FOR_P_R_F1]
    
    # Reset GT usage
    for gts in ground_truths.values():
        for gt in gts: gt['used'] = False

    for pred in preds_above_conf:
        image_gts = ground_truths.get(pred['image_id'], [])
        class_gts = [gt for gt in image_gts if gt['class_id'] == pred['class_id']]
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(class_gts):
            iou = calculate_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= 0.5 and best_gt_idx != -1 and not class_gts[best_gt_idx]['used']:
            total_tp_50 += 1
            class_gts[best_gt_idx]['used'] = True
        else:
            total_fp_50 += 1
    
    total_fn_50 = num_total_gts - total_tp_50
    
    avg_precision = total_tp_50 / (total_tp_50 + total_fp_50) if (total_tp_50 + total_fp_50) > 0 else 0.0
    avg_recall = total_tp_50 / (total_tp_50 + total_fn_50) if (total_tp_50 + total_fn_50) > 0 else 0.0
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

    # Calculate mAPs
    map_50 = np.mean([ap_results[cid][0.5] for cid in class_ids])
    map_90 = np.mean([ap_results[cid][iou_thresholds[8]] for cid in class_ids]) # iou_thresholds[8] is 0.9
    map_50_95 = np.mean([ap_results[cid][iou] for cid in class_ids for iou in iou_thresholds])

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("           EVALUATION RESULTS")
    print("="*40)
    print(f"\nOverall Metrics (at IoU=0.5, Conf={CONFIDENCE_THRESHOLD_FOR_P_R_F1}):")
    print(f"  - Average Precision: {avg_precision:.4f}")
    print(f"  - Average Recall:    {avg_recall:.4f}")
    print(f"  - Average F1-Score:  {avg_f1:.4f}")
    print("\nMean Average Precision (mAP):")
    print(f"  - mAP @ IoU=0.50:         {map_50:.4f}")
    print(f"  - mAP @ IoU=0.90:         {map_90:.4f}")
    print(f"  - mAP @ IoU=0.50:0.95:    {map_50_95:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()