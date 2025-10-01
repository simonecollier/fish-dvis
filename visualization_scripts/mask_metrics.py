import json
import os
import numpy as np
from pycocotools import mask as maskUtils
from collections import defaultdict
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import copy
from contextlib import contextmanager
import time
import sys
from io import StringIO

# Import DVIS-DAQ evaluation functions
try:
    from dvis_daq_eval import compute_dvis_daq_metrics
    DVIS_DAQ_AVAILABLE = True
except ImportError:
    print("Warning: DVIS-DAQ evaluation not available. Using custom implementation.")
    DVIS_DAQ_AVAILABLE = False

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

@contextmanager
def dvis_daq_lock():
    """File-based lock to serialize DVIS-DAQ evaluation across processes."""
    lock_path = '/tmp/dvis_daq_eval.lock'
    # Ensure file exists and acquire exclusive lock
    lock_file = open(lock_path, 'w')
    try:
        import fcntl
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        finally:
            lock_file.close()

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def decode_rle(rle):
    # rle: dict with 'size' and 'counts'
    return maskUtils.decode(rle)

def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 1.0

def compute_mask_area(mask):
    """Compute the area of a mask in pixels."""
    return np.sum(mask)

def get_area_range(area):
    """Determine if an object is small, medium, or large based on area."""
    if area < 32**2:
        return 'small'
    elif area < 96**2:
        return 'medium'
    else:
        return 'large'

def compute_dvis_daq_metrics_wrapper(preds, gts, val_json_path):
    """
    Wrapper function to call the DVIS-DAQ evaluation methodology.
    """
    try:
        # Call the DVIS-DAQ evaluation function directly
        from dvis_daq_eval import compute_dvis_daq_metrics
        result = compute_dvis_daq_metrics(preds, val_json_path)
        return result
    except Exception as e:
        print(f"Error in DVIS-DAQ evaluation: {e}")
        return None

def compute_frame_level_coco_metrics(preds, gts, iou_thresholds=None):
    """
    Compute standard COCO-style metrics at frame level.
    
    Args:
        preds: List of predictions in COCO format
        gts: Ground truth annotations in COCO format
        iou_thresholds: List of IoU thresholds (default: [0.5, 0.55, ..., 0.95])
    
    Returns:
        Dictionary with standard COCO metrics
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    # Convert video predictions to frame-level predictions
    frame_preds = []
    frame_gts = []
    
    # Group by video and frame
    video_frame_preds = defaultdict(lambda: defaultdict(list))
    video_frame_gts = defaultdict(lambda: defaultdict(list))
    
    # Process predictions
    for pred in preds:
        video_id = pred['video_id']
        segmentations = pred['segmentations']
        for frame_idx, seg in enumerate(segmentations):
            if seg is not None:
                frame_preds.append({
                    'image_id': f"{video_id}_{frame_idx}",
                    'category_id': pred['category_id'],
                    'segmentation': seg,
                    'score': pred['score'],
                    'area': compute_mask_area(decode_rle(seg))
                })
                video_frame_preds[video_id][frame_idx].append(frame_preds[-1])
    
    # Process ground truth
    for ann in gts['annotations']:
        video_id = ann['video_id']
        segmentations = ann['segmentations']
        for frame_idx, seg in enumerate(segmentations):
            if seg is not None:
                frame_gts.append({
                    'image_id': f"{video_id}_{frame_idx}",
                    'category_id': ann['category_id'],
                    'segmentation': seg,
                    'area': compute_mask_area(decode_rle(seg))
                })
                video_frame_gts[video_id][frame_idx].append(frame_gts[-1])
    
    # Compute metrics for each IoU threshold
    all_metrics = {}
    
    for iou_thresh in iou_thresholds:
        metrics = compute_ap_at_iou_threshold(frame_preds, frame_gts, iou_thresh)
        all_metrics[f'IoU_{iou_thresh:.2f}'] = metrics
    
    # Compute area-based metrics
    area_metrics = compute_area_based_metrics(frame_preds, frame_gts)
    all_metrics.update(area_metrics)
    
    # Compute recall-based metrics
    recall_metrics = compute_recall_metrics(frame_preds, frame_gts)
    all_metrics.update(recall_metrics)
    
    return all_metrics

def compute_ap_at_iou_threshold(preds, gts, iou_threshold):
    """
    Compute AP at a specific IoU threshold using frame-level evaluation.
    This implementation follows the COCO evaluation protocol more closely.
    """
    # Group by image_id AND category_id (like OVISeval)
    pred_by_image_cat = defaultdict(list)
    gt_by_image_cat = defaultdict(list)
    
    for pred in preds:
        key = (pred['image_id'], pred['category_id'])
        pred_by_image_cat[key].append(pred)
    
    for gt in gts:
        key = (gt['image_id'], gt['category_id'])
        gt_by_image_cat[key].append(gt)
    
    all_scores = []
    all_matches = []
    num_gt_total = 0
    
    # Process each image-category combination separately
    for (image_id, cat_id) in set(list(pred_by_image_cat.keys()) + list(gt_by_image_cat.keys())):
        image_preds = pred_by_image_cat[(image_id, cat_id)]
        image_gts = gt_by_image_cat[(image_id, cat_id)]
        
        num_gt_total += len(image_gts)
        
        if not image_preds:
            continue
        
        # Sort predictions by score
        image_preds.sort(key=lambda p: p['score'], reverse=True)
        
        if not image_gts:
            all_scores.extend([p['score'] for p in image_preds])
            all_matches.extend([0] * len(image_preds))
            continue
        
        # Compute IoU matrix (only within same category)
        iou_matrix = np.zeros((len(image_gts), len(image_preds)))
        for i, gt in enumerate(image_gts):
            gt_mask = decode_rle(gt['segmentation'])
            for j, pred in enumerate(image_preds):
                pred_mask = decode_rle(pred['segmentation'])
                iou = compute_iou(pred_mask, gt_mask)
                iou_matrix[i, j] = iou
        
        # Match predictions to ground truth (greedy matching within category)
        gt_matched = {i: False for i in range(len(image_gts))}
        
        for j in range(len(image_preds)):
            all_scores.append(image_preds[j]['score'])
            
            best_iou = -1
            best_gt_idx = -1
            for i in range(len(image_gts)):
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = i
            
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                gt_matched[best_gt_idx] = True
                all_matches.append(1)  # TP
            else:
                all_matches.append(0)  # FP (low IoU or duplicate detection)
    
    if num_gt_total == 0:
        return 1.0 if not all_scores else 0.0
    
    if not all_scores:
        return 0.0
    
    # Compute precision-recall curve
    sorted_indices = np.argsort(-np.array(all_scores))
    all_matches = np.array(all_matches)[sorted_indices]
    
    tp = np.cumsum(all_matches)
    fp = np.cumsum(all_matches == 0)
    
    recalls = tp / num_gt_total
    precisions = tp / (tp + fp)
    
    # Compute AP using 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
        ap += p / 101
    
    return ap

def compute_area_based_metrics(preds, gts):
    """
    Compute AP for small, medium, and large objects.
    """
    area_ranges = {
        'small': (0, 32**2),
        'medium': (32**2, 96**2),
        'large': (96**2, float('inf'))
    }
    
    metrics = {}
    
    for area_name, (min_area, max_area) in area_ranges.items():
        # Filter predictions and ground truth by area
        area_preds = [p for p in preds if min_area <= p['area'] < max_area]
        area_gts = [g for g in gts if min_area <= g['area'] < max_area]
        
        # Compute AP at IoU=0.5 for this area range
        ap = compute_ap_at_iou_threshold(area_preds, area_gts, 0.5)
        metrics[f'AP_{area_name[0]}'] = ap  # APs, APm, APl
    
    return metrics

def compute_recall_metrics(preds, gts, max_dets_list=[1, 10]):
    """
    Compute Average Recall (AR) for different max detections.
    """
    metrics = {}
    
    for max_dets in max_dets_list:
        # Limit predictions to max_dets per image-category combination
        pred_by_image_cat = defaultdict(list)
        for pred in preds:
            key = (pred['image_id'], pred['category_id'])
            pred_by_image_cat[key].append(pred)
        
        limited_preds = []
        for (image_id, cat_id), image_preds in pred_by_image_cat.items():
            # Sort by score and take top max_dets per category
            image_preds.sort(key=lambda p: p['score'], reverse=True)
            limited_preds.extend(image_preds[:max_dets])
        
        # Compute recall at IoU=0.5
        recall = compute_recall_at_iou_threshold(limited_preds, gts, 0.5)
        metrics[f'AR{max_dets}'] = recall
    
    return metrics

def compute_recall_at_iou_threshold(preds, gts, iou_threshold):
    """
    Compute recall at a specific IoU threshold.
    This implementation follows the COCO evaluation protocol more closely.
    """
    # Group by image_id AND category_id (like OVISeval)
    pred_by_image_cat = defaultdict(list)
    gt_by_image_cat = defaultdict(list)
    
    for pred in preds:
        key = (pred['image_id'], pred['category_id'])
        pred_by_image_cat[key].append(pred)
    
    for gt in gts:
        key = (gt['image_id'], gt['category_id'])
        gt_by_image_cat[key].append(gt)
    
    total_gt = 0
    total_matched = 0
    
    # Process each image-category combination separately
    for (image_id, cat_id) in set(list(pred_by_image_cat.keys()) + list(gt_by_image_cat.keys())):
        image_preds = pred_by_image_cat[(image_id, cat_id)]
        image_gts = gt_by_image_cat[(image_id, cat_id)]
        
        total_gt += len(image_gts)
        
        if not image_preds or not image_gts:
            continue
        
        # Compute IoU matrix (only within same category)
        iou_matrix = np.zeros((len(image_gts), len(image_preds)))
        for i, gt in enumerate(image_gts):
            gt_mask = decode_rle(gt['segmentation'])
            for j, pred in enumerate(image_preds):
                pred_mask = decode_rle(pred['segmentation'])
                iou = compute_iou(pred_mask, gt_mask)
                iou_matrix[i, j] = iou
        
        # Count matches (greedy matching within category)
        gt_matched = {i: False for i in range(len(image_gts))}
        
        for j in range(len(image_preds)):
            best_iou = -1
            best_gt_idx = -1
            for i in range(len(image_gts)):
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = i
            
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                gt_matched[best_gt_idx] = True
                total_matched += 1
    
    return total_matched / total_gt if total_gt > 0 else 0.0







# Note: Confusion matrix functions have been moved to confusion_mat_plot.py
# Import them using: from confusion_mat_plot import tracking_confusion_matrix, simple_confusion_matrix, etc.

def main(results_json, val_json, csv_path=None, cm_plot_path=None, confidence_threshold=0.0, verbose=True):
    # Helper function for conditional printing
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    # Default CSV path to the same directory as results.json
    if csv_path is None:
        results_dir = os.path.dirname(results_json)
        csv_path = os.path.join(results_dir, "mask_metrics.csv")
    
    # Default confusion matrix path to the same directory as results.json
    if cm_plot_path is None:
        results_dir = os.path.dirname(results_json)
        cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    preds = load_json(results_json)
    gts = load_json(val_json)

    # Filter out low confidence predictions (for reference only)
    if confidence_threshold > 0.0:
        filtered_preds = [pred for pred in preds if pred.get('score', 0.0) >= confidence_threshold]
        vprint(f"Original predictions: {len(preds)}")
        vprint(f"Filtered predictions (confidence >= {confidence_threshold}): {len(filtered_preds)}")
        vprint(f"Note: Using ALL predictions for all metrics to match DVIS-DAQ evaluation methodology")
    else:
        filtered_preds = preds
        vprint(f"Using all {len(preds)} predictions (no confidence threshold applied)")
    
    if len(filtered_preds) == 0:
        vprint("WARNING: No predictions meet the confidence threshold!")
        vprint("This will result in all metrics being 0.")
        # Create empty CSV with headers
        fieldnames = ["video_id", "category_id", "file_name", "frame_IoU", 
                     "video_IoU", "dataset_IoU", 
                     "ap_instance_Aweighted", "ap50_instance_Aweighted", "ap75_instance_Aweighted", 
                     "aps_instance_Aweighted", "apm_instance_Aweighted", "apl_instance_Aweighted", 
                     "ar1_instance", "ar10_instance",
                     "ap_instance_per_cat", "ap50_instance_per_cat", "ap75_instance_per_cat", 
                     "aps_instance_per_cat", "apm_instance_per_cat", "apl_instance_per_cat", 
                     "ar1_instance_per_cat", "ar10_instance_per_cat",
                     "predicted_category_id", "predicted_category_name", "predicted_score", 
                     "gt_category_id", "gt_category_name"]
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        return

    
    gt_category_ids = {cat['id'] for cat in gts['categories']}
    gt_category_names = {cat['id']: cat['name'] for cat in gts['categories']}
        
    # Try to use DVIS-DAQ evaluation first
    if DVIS_DAQ_AVAILABLE:
        # Use a file lock to serialize DVIS-DAQ evaluation (it renames/restores pycocotools directory)
        with dvis_daq_lock():
            # Suppress stdout if not verbose
            if verbose:
                dvis_daq_metrics = compute_dvis_daq_metrics_wrapper(preds, gts, val_json)
            else:
                with suppress_stdout():
                    dvis_daq_metrics = compute_dvis_daq_metrics_wrapper(preds, gts, val_json)
        
        if dvis_daq_metrics is not None:
            coco_metrics = dvis_daq_metrics
        else:
            coco_metrics = compute_frame_level_coco_metrics(filtered_preds, gts)
    else:
        coco_metrics = compute_frame_level_coco_metrics(filtered_preds, gts)
    
    # Extract standard metrics - handle both DVIS-DAQ and custom evaluation formats
    if 'AP' in coco_metrics:  # DVIS-DAQ format
        ap = coco_metrics['AP']
        ap50 = coco_metrics['AP50']
        ap75 = coco_metrics['AP75']
        aps = coco_metrics['APs']
        apm = coco_metrics['APm']
        apl = coco_metrics['APl']
        ar1 = coco_metrics['AR1']
        ar10 = coco_metrics['AR10']
    else:  # Custom evaluation format
        ap = np.mean([coco_metrics[f'IoU_{iou:.2f}'] for iou in np.linspace(0.5, 0.95, 10)])
        ap50 = coco_metrics['IoU_0.50']
        ap75 = coco_metrics['IoU_0.75']
        aps = coco_metrics['AP_s']
        apm = coco_metrics['AP_m']
        apl = coco_metrics['AP_l']
        ar1 = coco_metrics['AR1']
        ar10 = coco_metrics['AR10']
        
    # Print summary metrics only
    vprint(f"COCO Metrics - AP: {ap:.3f}, AP50: {ap50:.3f}, AP75: {ap75:.3f}")
    
    # Build GT dict: {video_id: [annotations]}
    gt_dict = defaultdict(list)
    gt_meta = {}
    for ann in gts['annotations']:
        gt_dict[ann['video_id']].append(ann)
        gt_meta[(ann['video_id'], ann['id'])] = (ann['category_id'], ann.get('file_name', ''))

    # Group predictions by video_id - use ALL predictions for consistency with DVIS-DAQ
    pred_by_video = defaultdict(list)
    for pred in preds:  # Use ALL predictions (not filtered) for consistency
        if pred.get('category_id') in gt_category_ids:
            vid = pred['video_id']
            pred_by_video[vid].append(pred)

    all_frame_ious = []
    frame_metrics_rows = []
    video_metrics_rows = []
    category_metrics_rows = []
    dataset_metrics_rows = []
    video_ious = {}

    video_ids = set(list(gt_dict.keys()) + list(pred_by_video.keys()))
    # Prepare per-video prediction info
    video_pred_info = {}
    for video_id in video_ids:
        preds_for_video = pred_by_video.get(video_id, [])
        if preds_for_video:
            best_pred = max(preds_for_video, key=lambda p: p.get('score', 0.0))
            pred_cat_id = best_pred.get('category_id', None)
            pred_cat_name = gt_category_names.get(pred_cat_id, f"ID {pred_cat_id}")
            pred_score = best_pred.get('score', 0.0)
        else:
            pred_cat_id = None
            pred_cat_name = None
            pred_score = None
        gt_cat_ids = sorted(set(ann['category_id'] for ann in gt_dict.get(video_id, [])))
        gt_cat_names = [gt_category_names.get(cid, f"ID {cid}") for cid in gt_cat_ids]
        video_pred_info[video_id] = {
            'predicted_category_id': pred_cat_id,
            'predicted_category_name': pred_cat_name,
            'predicted_score': pred_score,
            'gt_category_ids': gt_cat_ids,
            'gt_category_names': gt_cat_names
        }

    # Note: Per-video AP metrics are no longer computed
    # We now compute simplified track-level metrics directly in the video processing loop

    for video_id in video_ids:
        # Get GT and predictions for this video
        gt_tracks = gt_dict.get(video_id, [])
        pred_tracks = pred_by_video.get(video_id, [])
        
        # Initialize video metrics
        mean_track_iou = 0.0
        std_track_iou = 0.0
        completeness = 0.0
        mean_track_length = 0.0
        meets_iou_0_5 = False
        meets_iou_0_75 = False
        pred_score = 0.0
        top_pred_category_id = None
        top_pred_category_name = None
        gt_category_id = None
        gt_category_name = None
        
        if gt_tracks and pred_tracks:
            # For single fish per video, match the best prediction to the GT
            best_pred = max(pred_tracks, key=lambda p: p.get('score', 0.0))
            pred_score = best_pred.get('score', 0.0)
            top_pred_category_id = best_pred.get('category_id')
            top_pred_category_name = gt_category_names.get(top_pred_category_id, f"ID {top_pred_category_id}")
            
            # Get GT category (assuming single category per video)
            gt_category_id = gt_tracks[0]['category_id']
            gt_category_name = gt_category_names.get(gt_category_id, f"ID {gt_category_id}")
            
            # Compute track-level metrics
            gt_segs = gt_tracks[0]['segmentations']
            pred_segs = best_pred['segmentations']
            length = min(len(pred_segs), len(gt_segs))
            
            if length > 0:
                frame_ious = []
                frames_with_both = 0
                
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    
                    # Always use fixed shape for all-zeros mask
                    mask_shape = (960, 1280)
                    if pred_rle is None and gt_rle is not None:
                        pred_mask = np.zeros(mask_shape, dtype=np.uint8)
                        gt_mask = decode_rle(gt_rle)
                    elif gt_rle is None and pred_rle is not None:
                        gt_mask = np.zeros(mask_shape, dtype=np.uint8)
                        pred_mask = decode_rle(pred_rle)
                    elif pred_rle is None and gt_rle is None:
                        pred_mask = np.zeros(mask_shape, dtype=np.uint8)
                        gt_mask = np.zeros(mask_shape, dtype=np.uint8)
                    else:
                        pred_mask = decode_rle(pred_rle)
                        gt_mask = decode_rle(gt_rle)
                    
                    iou = compute_iou(pred_mask, gt_mask)
                    frame_ious.append(iou)
                    
                    # Count frames with both GT and prediction
                    if pred_rle is not None and gt_rle is not None:
                        frames_with_both += 1
                    
                    # Add to frame-level metrics
                    all_frame_ious.append(iou)
                    
                    frame_metrics_rows.append({
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "prediction_id": best_pred.get('id', f"pred_{video_id}"),
                        "gt_id": gt_tracks[0].get('id', f"gt_{video_id}"),
                        "category_id": gt_category_id,
                        "file_name": gt_meta.get((video_id, gt_tracks[0]['id']), ('', ''))[1],
                        "frame_IoU": iou,
                        "predicted_category_id": top_pred_category_id,
                        "predicted_score": pred_score,
                        "gt_category_id": gt_category_id,
                        "gt_category_name": gt_category_name
                    })
                
                # Compute video-level metrics
                mean_track_iou = np.mean(frame_ious) if frame_ious else 0.0
                std_track_iou = np.std(frame_ious) if len(frame_ious) > 1 else 0.0
                completeness = frames_with_both / length if length > 0 else 0.0
                mean_track_length = length
                meets_iou_0_5 = mean_track_iou >= 0.5
                meets_iou_0_75 = mean_track_iou >= 0.75
        
        # Store video metrics
        video_ious[video_id] = mean_track_iou
        
        video_metrics_rows.append({
            "video_id": video_id,
            "gt_category_id": gt_category_id,
            "gt_category_name": gt_category_name,
            "mean_track_IoU": mean_track_iou,
            "std_track_IoU": std_track_iou,
            "completeness": completeness,
            "mean_track_length": mean_track_length,
            "meets_iou_0_5": meets_iou_0_5,
            "meets_iou_0_75": meets_iou_0_75,
            "pred_score": pred_score,
            "top_pred_category_id": top_pred_category_id,
            "top_pred_category_name": top_pred_category_name
        })

    dataset_iou = np.mean(list(video_ious.values())) if video_ious else 0.0

    # Compute standard COCO-style metrics (frame-level)
    # Try to use DVIS-DAQ evaluation first
    if DVIS_DAQ_AVAILABLE:
        # Use a file lock to serialize DVIS-DAQ evaluation (it renames/restores pycocotools directory)
        with dvis_daq_lock():
            # Suppress stdout if not verbose
            if verbose:
                dvis_daq_metrics = compute_dvis_daq_metrics_wrapper(preds, gts, val_json)
            else:
                with suppress_stdout():
                    dvis_daq_metrics = compute_dvis_daq_metrics_wrapper(preds, gts, val_json)
            
        if dvis_daq_metrics is not None:
            coco_metrics = dvis_daq_metrics
        else:
            coco_metrics = compute_frame_level_coco_metrics(filtered_preds, gts)
    else:
        coco_metrics = compute_frame_level_coco_metrics(filtered_preds, gts)
    
    # Per-video metrics already computed above
    
    # Extract standard metrics - handle both DVIS-DAQ and custom evaluation formats
    if 'AP' in coco_metrics:  # DVIS-DAQ format
        ap = coco_metrics['AP']
        ap50 = coco_metrics['AP50']
        ap75 = coco_metrics['AP75']
        aps = coco_metrics['APs']
        apm = coco_metrics['APm']
        apl = coco_metrics['APl']
        ar1 = coco_metrics['AR1']
        ar10 = coco_metrics['AR10']
    else:  # Custom evaluation format
        ap = np.mean([coco_metrics[f'IoU_{iou:.2f}'] for iou in np.linspace(0.5, 0.95, 10)])
        ap50 = coco_metrics['IoU_0.50']
        ap75 = coco_metrics['IoU_0.75']
        aps = coco_metrics['AP_s']
        apm = coco_metrics['AP_m']
        apl = coco_metrics['AP_l']
        ar1 = coco_metrics['AR1']
        ar10 = coco_metrics['AR10']
    
    # Print summary metrics only
    vprint(f"Standard COCO Metrics - AP: {ap:.3f}, AP50: {ap50:.3f}, AP75: {ap75:.3f}")
    
    # Extract per-category instance-level metrics
    if DVIS_DAQ_AVAILABLE and dvis_daq_metrics is not None:
        if 'per_category_metrics' in dvis_daq_metrics:
            # Use the new per-category metrics from DVIS-DAQ evaluation
            per_cat_instance_metrics = {}
            for cat_name, cat_metrics in dvis_daq_metrics['per_category_metrics'].items():
                # Find the category ID for this category name
                cat_id = None
                for cid, cname in gt_category_names.items():
                    if cname == cat_name:
                        cat_id = cid
                        break
                
                if cat_id is not None:
                    per_cat_instance_metrics[cat_id] = cat_metrics
        elif 'evaluator' in dvis_daq_metrics:
            # Fallback to old method
            per_cat_instance_metrics = extract_per_category_instance_metrics(
                dvis_daq_metrics['evaluator'], 
                gt_category_ids, 
                gt_category_names
            )
        else:
            per_cat_instance_metrics = {}
    else:
        per_cat_instance_metrics = {}
    
    # Process categories for per-category metrics
    for cat_id in gt_category_ids:
        # Add category-level metrics
        cat_name = gt_category_names.get(cat_id, f"ID {cat_id}")
        cat_metrics_row = {
            "category_id": cat_id,
            "category_name": cat_name,
            "num_videos": len([v for v in video_ids if any(ann['category_id'] == cat_id for ann in gt_dict.get(v, []))]),
            "num_frames": len([row for row in frame_metrics_rows if row['gt_category_id'] == cat_id]),
            "num_gt_objects": len([ann for ann in gts['annotations'] if ann['category_id'] == cat_id])
        }
        
        # Add instance-level per-category metrics if available
        if cat_id in per_cat_instance_metrics:
            cat_metrics = per_cat_instance_metrics[cat_id]
            # DVIS-DAQ evaluator already returns decimal values (0-1 range)
            cat_metrics_row.update({
                "ap_instance_per_cat": cat_metrics.get('AP', None) if cat_metrics.get('AP') is not None else None,
                "ap50_instance_per_cat": cat_metrics.get('AP50', None) if cat_metrics.get('AP50') is not None else None,
                "ap75_instance_per_cat": cat_metrics.get('AP75', None) if cat_metrics.get('AP75') is not None else None,
                "aps_instance_per_cat": cat_metrics.get('APs', None) if cat_metrics.get('APs') is not None else None,
                "apm_instance_per_cat": cat_metrics.get('APm', None) if cat_metrics.get('APm') is not None else None,
                "apl_instance_per_cat": cat_metrics.get('APl', None) if cat_metrics.get('APl') is not None else None,
                "ar1_instance_per_cat": cat_metrics.get('AR1', None) if cat_metrics.get('AR1') is not None else None,
                "ar10_instance_per_cat": cat_metrics.get('AR10', None) if cat_metrics.get('AR10') is not None else None
            })
        else:
            cat_metrics_row.update({
                "ap_instance_per_cat": None,
                "ap50_instance_per_cat": None,
                "ap75_instance_per_cat": None,
                "aps_instance_per_cat": None,
                "apm_instance_per_cat": None,
                "apl_instance_per_cat": None,
                "ar1_instance_per_cat": None,
                "ar10_instance_per_cat": None
            })
        
        category_metrics_rows.append(cat_metrics_row)


    # Add dataset-level metrics
    dataset_metrics_rows = [
        {"metric_name": "dataset_IoU", "value": dataset_iou, "description": "Average IoU across all videos"},
        {"metric_name": "ap_instance_Aweighted", "value": ap, "description": "Instance-level AP (DVIS-DAQ, area-weighted)"},
        {"metric_name": "ap50_instance_Aweighted", "value": ap50, "description": "Instance-level AP50 (DVIS-DAQ, area-weighted)"},
        {"metric_name": "ap75_instance_Aweighted", "value": ap75, "description": "Instance-level AP75 (DVIS-DAQ, area-weighted)"},
        {"metric_name": "aps_instance_Aweighted", "value": aps, "description": "Instance-level APs (DVIS-DAQ, small objects)"},
        {"metric_name": "apm_instance_Aweighted", "value": apm, "description": "Instance-level APm (DVIS-DAQ, medium objects)"},
        {"metric_name": "apl_instance_Aweighted", "value": apl, "description": "Instance-level APl (DVIS-DAQ, large objects)"},
        {"metric_name": "ar1_instance", "value": ar1, "description": "Instance-level AR1 (DVIS-DAQ)"},
        {"metric_name": "ar10_instance", "value": ar10, "description": "Instance-level AR10 (DVIS-DAQ)"}
    ]

    # Print summary only
    vprint(f"Dataset IoU: {dataset_iou:.3f}")

    # Save separate CSV files
    csv_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Save frame.csv
    frame_csv_path = os.path.join(csv_dir, f"{base_name}_frame.csv")
    with open(frame_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["video_id", "frame_idx", "prediction_id", "gt_id", "category_id", "file_name", "frame_IoU", "predicted_category_id", "predicted_score", "gt_category_id", "gt_category_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_metrics_rows:
            writer.writerow(row)
    
    # Save video.csv
    video_csv_path = os.path.join(csv_dir, f"{base_name}_video.csv")
    with open(video_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["video_id", "gt_category_id", "gt_category_name", "mean_track_IoU", "std_track_IoU", "completeness", "mean_track_length", "meets_iou_0_5", "meets_iou_0_75", "pred_score", "top_pred_category_id", "top_pred_category_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in video_metrics_rows:
            writer.writerow(row)
    
    # Save category.csv
    category_csv_path = os.path.join(csv_dir, f"{base_name}_category.csv")
    with open(category_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["category_id", "category_name", "ap_instance_per_cat", "ap50_instance_per_cat", "ap75_instance_per_cat", "aps_instance_per_cat", "apm_instance_per_cat", "apl_instance_per_cat", "ar1_instance_per_cat", "ar10_instance_per_cat", "num_videos", "num_frames", "num_gt_objects"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in category_metrics_rows:
            writer.writerow(row)
    
    # Category metrics processed silently
    
    # Save dataset.csv
    dataset_csv_path = os.path.join(csv_dir, f"{base_name}_dataset.csv")
    with open(dataset_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["metric_name", "value", "description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset_metrics_rows:
            writer.writerow(row)
    
    # CSV files saved silently
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["video_id", "frame_idx", "prediction_id", "gt_id", "category_id", "file_name", "frame_IoU", "video_IoU", "dataset_IoU", "ap_instance_Aweighted", "ap50_instance_Aweighted", "ap75_instance_Aweighted", "aps_instance_Aweighted", "apm_instance_Aweighted", "apl_instance_Aweighted", "ar1_instance", "ar10_instance", "ap_instance_per_cat", "ap50_instance_per_cat", "ap75_instance_per_cat", "aps_instance_per_cat", "apm_instance_per_cat", "apl_instance_per_cat", "ar1_instance_per_cat", "ar10_instance_per_cat", "predicted_category_id", "predicted_category_name", "predicted_score", "gt_category_id", "gt_category_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Add frame-level rows
        for row in frame_metrics_rows:
            # Add video and dataset level metrics to each frame row
            video_id = row["video_id"]
            row["video_IoU"] = video_ious[video_id]
            row["dataset_IoU"] = dataset_iou
            row["ap_instance_Aweighted"] = ap
            row["ap50_instance_Aweighted"] = ap50
            row["ap75_instance_Aweighted"] = ap75
            row["aps_instance_Aweighted"] = aps
            row["apm_instance_Aweighted"] = apm
            row["apl_instance_Aweighted"] = apl
            row["ar1_instance"] = ar1
            row["ar10_instance"] = ar10
            
            # Add per-category metrics
            cat_id = row["category_id"]
            if cat_id in per_cat_instance_metrics:
                cat_metrics = per_cat_instance_metrics[cat_id]
                row["ap_instance_per_cat"] = cat_metrics.get('AP', None)
                row["ap50_instance_per_cat"] = cat_metrics.get('AP50', None)
                row["ap75_instance_per_cat"] = cat_metrics.get('AP75', None)
                row["aps_instance_per_cat"] = cat_metrics.get('APs', None)
                row["apm_instance_per_cat"] = cat_metrics.get('APm', None)
                row["apl_instance_per_cat"] = cat_metrics.get('APl', None)
                row["ar1_instance_per_cat"] = cat_metrics.get('AR1', None)
                row["ar10_instance_per_cat"] = cat_metrics.get('AR10', None)
            else:
                row["ap_instance_per_cat"] = None
                row["ap50_instance_per_cat"] = None
                row["ap75_instance_per_cat"] = None
                row["aps_instance_per_cat"] = None
                row["apm_instance_per_cat"] = None
                row["apl_instance_per_cat"] = None
                row["ar1_instance_per_cat"] = None
                row["ar10_instance_per_cat"] = None
            
            writer.writerow(row)

    # --- Standard VIS Confusion Matrix Calculation ---
    # Import confusion matrix functions from the new script
    try:
        from confusion_mat_plot import tracking_confusion_matrix, simple_confusion_matrix, plot_confusion_matrix, print_class_metrics
        
        # Use simple confusion matrix as default (highest scoring prediction per video) with no confidence threshold
        
        # Compute confusion matrix using simple method (default)
        cm, class_names, class_metrics = simple_confusion_matrix(
            preds, gts, 
            confidence_threshold=0.0
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, cm_plot_path, "Simple VIS Confusion Matrix", 
                            method="simple", conf_threshold=0.0)
        
        # Print per-class metrics
        overall_metrics = print_class_metrics(class_metrics, "simple")
        
    except ImportError:
        pass  # Skip confusion matrix computation silently

# Note: plot_confusion_matrix_from_csv function has been moved to confusion_mat_plot.py






if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run mask metrics analysis')
    parser.add_argument('--results-json', type=str, required=True,
                       help='Path to results.json file')
    parser.add_argument('--val-json', type=str, default='/data/fishway_ytvis/val.json',
                       help='Path to validation JSON file')
    parser.add_argument('--csv-path', type=str, default=None,
                       help='Path to save mask metrics CSV (defaults to same directory as results.json)')
    parser.add_argument('--cm-plot-path', type=str, default=None,
                       help='Path to save confusion matrix plot (defaults to same directory as results.json)')

    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                       help='Confidence threshold for filtering predictions (default: 0.0 = no threshold)')
    
    args = parser.parse_args()
    
    # Run main analysis
    main(results_json=args.results_json, 
         val_json=args.val_json, 
         csv_path=args.csv_path,
         cm_plot_path=args.cm_plot_path,
         confidence_threshold=args.confidence_threshold)
        
