import json
import os
import numpy as np
from pycocotools import mask as maskUtils
from collections import defaultdict
import csv
from skimage import measure
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import copy

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

def compute_temporal_consistency_metrics(pred_by_video, gt_dict):
    """
    Compute temporal consistency metrics for video instance segmentation.
    
    Returns:
        Dictionary with temporal consistency metrics
    """
    temporal_metrics = {
        'track_completeness': 0.0,
        'temporal_iou_stability': 0.0,
        'track_fragmentation': 0.0,
        'mean_track_length': 0.0
    }
    
    total_tracks = 0
    total_completeness = 0.0
    total_stability = 0.0
    total_fragmentation = 0.0
    total_length = 0.0
    
    for video_id in set(list(gt_dict.keys()) + list(pred_by_video.keys())):
        gt_tracks = gt_dict.get(video_id, [])
        pred_tracks = pred_by_video.get(video_id, [])
        
        if not gt_tracks or not pred_tracks:
            continue
        
        # Match tracks using Hungarian algorithm
        num_gt = len(gt_tracks)
        num_pred = len(pred_tracks)
        cost_matrix = np.ones((num_gt, num_pred))
        
        for i, gt_ann in enumerate(gt_tracks):
            gt_segs = gt_ann['segmentations']
            for j, pred_ann in enumerate(pred_tracks):
                pred_segs = pred_ann['segmentations']
                length = min(len(pred_segs), len(gt_segs))
                ious = []
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    if pred_rle is None and gt_rle is None:
                        ious.append(1.0)
                        continue
                    if pred_rle is None or gt_rle is None:
                        ious.append(0.0)
                        continue
                    
                    pred_mask = decode_rle(pred_rle)
                    gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    ious.append(iou)
                mean_iou = np.mean(ious) if ious else 0.0
                cost_matrix[i, j] = 1 - mean_iou
        
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            gt_ann = gt_tracks[gt_idx]
            pred_ann = pred_tracks[pred_idx]
            gt_segs = gt_ann['segmentations']
            pred_segs = pred_ann['segmentations']
            
            # Track completeness: fraction of frames where both GT and pred exist
            length = min(len(pred_segs), len(gt_segs))
            valid_frames = sum(1 for i in range(length) 
                             if pred_segs[i] is not None and gt_segs[i] is not None)
            completeness = valid_frames / length if length > 0 else 0.0
            total_completeness += completeness
            
            # Temporal IoU stability: std of IoU across frames
            ious = []
            for frame_idx in range(length):
                pred_rle = pred_segs[frame_idx]
                gt_rle = gt_segs[frame_idx]
                if pred_rle is None and gt_rle is None:
                    ious.append(1.0)
                    continue
                if pred_rle is None or gt_rle is None:
                    ious.append(0.0)
                    continue
                
                pred_mask = decode_rle(pred_rle)
                gt_mask = decode_rle(gt_rle)
                iou = compute_iou(pred_mask, gt_mask)
                ious.append(iou)
            
            if len(ious) > 1:
                stability = 1.0 - np.std(ious)  # Higher is better
                total_stability += stability
            
            # Track fragmentation: number of gaps in prediction
            pred_gaps = 0
            for i in range(1, len(pred_segs)):
                if pred_segs[i] is None and pred_segs[i-1] is not None:
                    pred_gaps += 1
            fragmentation = pred_gaps / max(1, len(pred_segs) - 1)
            total_fragmentation += fragmentation
            
            # Track length
            total_length += length
            total_tracks += 1
    
    if total_tracks > 0:
        temporal_metrics['track_completeness'] = total_completeness / total_tracks
        temporal_metrics['temporal_iou_stability'] = total_stability / total_tracks
        temporal_metrics['track_fragmentation'] = total_fragmentation / total_tracks
        temporal_metrics['mean_track_length'] = total_length / total_tracks
    
    return temporal_metrics

def compute_mota_metrics(pred_by_video, gt_dict):
    """
    Compute MOTA (Multiple Object Tracking Accuracy) and related metrics.
    
    MOTA = 1 - (FN + FP + IDSW) / GT
    where:
    - FN = False Negatives (missed detections)
    - FP = False Positives (false detections)
    - IDSW = Identity Switches
    - GT = Total ground truth objects
    
    Returns:
        Dictionary with MOTA metrics
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    total_matches = 0
    total_iou = 0.0
    
    for video_id in set(list(gt_dict.keys()) + list(pred_by_video.keys())):
        gt_tracks = gt_dict.get(video_id, [])
        pred_tracks = pred_by_video.get(video_id, [])
        
        if not gt_tracks or not pred_tracks:
            # Count unmatched GT as false negatives
            total_gt += sum(len(gt['segmentations']) for gt in gt_tracks)
            total_fn += sum(len(gt['segmentations']) for gt in gt_tracks)
            # Count all predictions as false positives
            total_fp += sum(len(pred['segmentations']) for pred in pred_tracks)
            continue
        
        # Match tracks using Hungarian algorithm
        num_gt = len(gt_tracks)
        num_pred = len(pred_tracks)
        cost_matrix = np.ones((num_gt, num_pred))
        
        for i, gt_ann in enumerate(gt_tracks):
            gt_segs = gt_ann['segmentations']
            for j, pred_ann in enumerate(pred_tracks):
                pred_segs = pred_ann['segmentations']
                length = min(len(pred_segs), len(gt_segs))
                ious = []
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    if pred_rle is None and gt_rle is None:
                        ious.append(1.0)
                        continue
                    if pred_rle is None or gt_rle is None:
                        ious.append(0.0)
                        continue
                    
                    pred_mask = decode_rle(pred_rle)
                    gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    ious.append(iou)
                mean_iou = np.mean(ious) if ious else 0.0
                cost_matrix[i, j] = 1 - mean_iou
        
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        
        # Count matched pairs
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            gt_ann = gt_tracks[gt_idx]
            pred_ann = pred_tracks[pred_idx]
            gt_segs = gt_ann['segmentations']
            pred_segs = pred_ann['segmentations']
            
            # Count valid frames
            length = min(len(pred_segs), len(gt_segs))
            valid_frames = sum(1 for i in range(length) 
                             if pred_segs[i] is not None and gt_segs[i] is not None)
            
            total_gt += valid_frames
            total_matches += valid_frames
            
            # Calculate IoU for MOTP
            for frame_idx in range(length):
                pred_rle = pred_segs[frame_idx]
                gt_rle = gt_segs[frame_idx]
                if pred_rle is not None and gt_rle is not None:
                    pred_mask = decode_rle(pred_rle)
                    gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    total_iou += iou
        
        # Count unmatched GT as false negatives
        matched_gt = len(gt_indices)
        unmatched_gt = num_gt - matched_gt
        total_fn += sum(len(gt_tracks[i]['segmentations']) for i in range(num_gt) if i not in gt_indices)
        
        # Count unmatched predictions as false positives
        matched_pred = len(pred_indices)
        unmatched_pred = num_pred - matched_pred
        total_fp += sum(len(pred_tracks[i]['segmentations']) for i in range(num_pred) if i not in pred_indices)
        
        # Count identity switches (simplified - count track switches)
        # This is a simplified version - full IDSW requires frame-by-frame analysis
        if matched_gt > 1:
            total_idsw += max(0, matched_gt - 1)  # Simplified approximation
    
    # Calculate MOTA
    mota = 1.0 - (total_fn + total_fp + total_idsw) / max(total_gt, 1)
    
    # Calculate MOTP (Multiple Object Tracking Precision)
    motp = total_iou / max(total_matches, 1)
    
    mota_metrics = {
        'MOTA': mota,
        'MOTP': motp,
        'total_gt': total_gt,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_idsw': total_idsw,
        'total_matches': total_matches
    }
    
    return mota_metrics

def compute_idf1_metrics(pred_by_video, gt_dict):
    """
    Compute IDF1 (ID F1-Score) metric.
    
    IDF1 measures the identity consistency between predictions and ground truth.
    It's the F1-score of identity assignments.
    
    Returns:
        Dictionary with IDF1 metrics
    """
    total_gt_objects = 0
    total_pred_objects = 0
    total_correct_assignments = 0
    
    for video_id in set(list(gt_dict.keys()) + list(pred_by_video.keys())):
        gt_tracks = gt_dict.get(video_id, [])
        pred_tracks = pred_by_video.get(video_id, [])
        
        if not gt_tracks or not pred_tracks:
            continue
        
        # Match tracks using Hungarian algorithm
        num_gt = len(gt_tracks)
        num_pred = len(pred_tracks)
        cost_matrix = np.ones((num_gt, num_pred))
        
        for i, gt_ann in enumerate(gt_tracks):
            gt_segs = gt_ann['segmentations']
            for j, pred_ann in enumerate(pred_tracks):
                pred_segs = pred_ann['segmentations']
                length = min(len(pred_segs), len(gt_segs))
                ious = []
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    if pred_rle is None and gt_rle is None:
                        ious.append(1.0)
                        continue
                    if pred_rle is None or gt_rle is None:
                        ious.append(0.0)
                        continue
                    
                    pred_mask = decode_rle(pred_rle)
                    gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    ious.append(iou)
                mean_iou = np.mean(ious) if ious else 0.0
                cost_matrix[i, j] = 1 - mean_iou
        
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        
        # Count correct identity assignments
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            gt_ann = gt_tracks[gt_idx]
            pred_ann = pred_tracks[pred_idx]
            gt_segs = gt_ann['segmentations']
            pred_segs = pred_ann['segmentations']
            
            # Count valid frames for this track pair
            length = min(len(pred_segs), len(gt_segs))
            valid_frames = sum(1 for i in range(length) 
                             if pred_segs[i] is not None and gt_segs[i] is not None)
            
            total_correct_assignments += valid_frames
        
        total_gt_objects += sum(len(gt['segmentations']) for gt in gt_tracks)
        total_pred_objects += sum(len(pred['segmentations']) for pred in pred_tracks)
    
    # Calculate IDF1
    precision = total_correct_assignments / max(total_pred_objects, 1)
    recall = total_correct_assignments / max(total_gt_objects, 1)
    idf1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    idf1_metrics = {
        'IDF1': idf1,
        'IDP': precision,  # ID Precision
        'IDR': recall,     # ID Recall
        'total_gt_objects': total_gt_objects,
        'total_pred_objects': total_pred_objects,
        'total_correct_assignments': total_correct_assignments
    }
    
    return idf1_metrics

def compute_hota_metrics(pred_by_video, gt_dict, alpha=0.05):
    """
    Compute HOTA (Higher Order Tracking Accuracy) metric.
    
    HOTA combines detection accuracy (DetA) and association accuracy (AssA).
    It's a comprehensive metric that considers both spatial and temporal aspects.
    
    Args:
        pred_by_video: Predictions grouped by video
        gt_dict: Ground truth grouped by video
        alpha: Alpha parameter for HOTA calculation (default: 0.05)
    
    Returns:
        Dictionary with HOTA metrics
    """
    total_hota = 0.0
    total_deta = 0.0
    total_assa = 0.0
    num_videos = 0
    
    for video_id in set(list(gt_dict.keys()) + list(pred_by_video.keys())):
        gt_tracks = gt_dict.get(video_id, [])
        pred_tracks = pred_by_video.get(video_id, [])
        
        if not gt_tracks or not pred_tracks:
            continue
        
        num_videos += 1
        
        # Match tracks using Hungarian algorithm
        num_gt = len(gt_tracks)
        num_pred = len(pred_tracks)
        cost_matrix = np.ones((num_gt, num_pred))
        
        for i, gt_ann in enumerate(gt_tracks):
            gt_segs = gt_ann['segmentations']
            for j, pred_ann in enumerate(pred_tracks):
                pred_segs = pred_ann['segmentations']
                length = min(len(pred_segs), len(gt_segs))
                ious = []
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    if pred_rle is None and gt_rle is None:
                        ious.append(1.0)
                        continue
                    if pred_rle is None or gt_rle is None:
                        ious.append(0.0)
                        continue
                    
                    pred_mask = decode_rle(pred_rle)
                    gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    ious.append(iou)
                mean_iou = np.mean(ious) if ious else 0.0
                cost_matrix[i, j] = 1 - mean_iou
        
        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        
        # Calculate HOTA components for this video
        video_deta = 0.0
        video_assa = 0.0
        video_hota = 0.0
        num_matches = 0
        
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            gt_ann = gt_tracks[gt_idx]
            pred_ann = pred_tracks[pred_idx]
            gt_segs = gt_ann['segmentations']
            pred_segs = pred_ann['segmentations']
            
            # Calculate IoU for this track pair
            length = min(len(pred_segs), len(gt_segs))
            ious = []
            for frame_idx in range(length):
                pred_rle = pred_segs[frame_idx]
                gt_rle = gt_segs[frame_idx]
                if pred_rle is not None and gt_rle is not None:
                    pred_mask = decode_rle(pred_rle)
                    gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    ious.append(iou)
            
            if ious:
                mean_iou = np.mean(ious)
                # Simplified HOTA calculation
                deta = mean_iou
                assa = mean_iou  # Simplified - in full HOTA this is more complex
                hota = np.sqrt(deta * assa)
                
                video_deta += deta
                video_assa += assa
                video_hota += hota
                num_matches += 1
        
        if num_matches > 0:
            video_deta /= num_matches
            video_assa /= num_matches
            video_hota /= num_matches
            
            total_deta += video_deta
            total_assa += video_assa
            total_hota += video_hota
    
    if num_videos > 0:
        avg_hota = total_hota / num_videos
        avg_deta = total_deta / num_videos
        avg_assa = total_assa / num_videos
    else:
        avg_hota = 0.0
        avg_deta = 0.0
        avg_assa = 0.0
    
    hota_metrics = {
        'HOTA': avg_hota,
        'DetA': avg_deta,
        'AssA': avg_assa,
        'num_videos': num_videos
    }
    
    return hota_metrics

def boundary_f_measure(pred_mask, gt_mask, bound_th=2):
    # Compute boundary F-measure for binary masks
    pred_contour = measure.find_contours(pred_mask, 0.5)
    gt_contour = measure.find_contours(gt_mask, 0.5)
    if not pred_contour or not gt_contour:
        return 0.0
    pred_points = np.concatenate(pred_contour)
    gt_points = np.concatenate(gt_contour)
    # Compute distances
    from scipy.spatial.distance import cdist
    dists_pred_to_gt = cdist(pred_points, gt_points)
    dists_gt_to_pred = cdist(gt_points, pred_points)
    # Precision: fraction of pred points within bound_th of any gt point
    precision = np.mean(np.min(dists_pred_to_gt, axis=1) <= bound_th)
    # Recall: fraction of gt points within bound_th of any pred point
    recall = np.mean(np.min(dists_gt_to_pred, axis=1) <= bound_th)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_ap_at_iou(pred_by_video, gt_dict, iou_threshold, category_id):
    """
    Compute Average Precision (AP) for a single category at a given IoU threshold.
    """
    all_scores = []
    all_matches = []
    num_gt_total = 0

    video_ids = set(list(gt_dict.keys()) + list(pred_by_video.keys()))

    for video_id in video_ids:
        gt_tracks = [ann for ann in gt_dict.get(video_id, []) if ann['category_id'] == category_id]
        pred_tracks = [pred for pred in pred_by_video.get(video_id, []) if pred['category_id'] == category_id]

        num_gt_total += len(gt_tracks)

        if not pred_tracks:
            continue

        gt_matched = {i: False for i in range(len(gt_tracks))}
        
        # Sort predictions by score
        pred_tracks.sort(key=lambda p: p.get('score', 0.0), reverse=True)
        
        if not gt_tracks:
            all_scores.extend([p.get('score', 0.0) for p in pred_tracks])
            all_matches.extend([0] * len(pred_tracks)) # All are FPs
            continue

        iou_matrix = np.zeros((len(gt_tracks), len(pred_tracks)))
        for i, gt_ann in enumerate(gt_tracks):
            gt_segs = gt_ann['segmentations']
            for j, pred_ann in enumerate(pred_tracks):
                pred_segs = pred_ann['segmentations']
                length = min(len(pred_segs), len(gt_segs))
                ious = []
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    if pred_rle is None and gt_rle is None:
                        ious.append(1.0)
                        continue
                    if pred_rle is None or gt_rle is None:
                        ious.append(0.0)
                        continue
                    
                    pred_mask = decode_rle(pred_rle)
                    gt_mask = decode_rle(gt_rle)
                    iou = compute_iou(pred_mask, gt_mask)
                    ious.append(iou)
                mean_iou = np.mean(ious) if ious else 0.0
                iou_matrix[i, j] = mean_iou
        
        for j in range(len(pred_tracks)):
            all_scores.append(pred_tracks[j].get('score', 0.0))
            
            best_iou = -1
            best_gt_idx = -1
            for i in range(len(gt_tracks)):
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = i
            
            if best_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    gt_matched[best_gt_idx] = True
                    all_matches.append(1)  # TP
                else:
                    all_matches.append(0)  # FP (duplicate detection)
            else:
                all_matches.append(0)  # FP
    
    if num_gt_total == 0:
        return 1.0 if not all_scores else 0.0

    if not all_scores:
        return 0.0

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

# Note: Confusion matrix functions have been moved to confusion_mat_plot.py
# Import them using: from confusion_mat_plot import tracking_confusion_matrix, simple_confusion_matrix, etc.

def main(results_json, val_json, csv_path=None, cm_plot_path=None, confidence_threshold=0.0, fast_mode=False):
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
        print(f"Original predictions: {len(preds)}")
        print(f"Filtered predictions (confidence >= {confidence_threshold}): {len(filtered_preds)}")
        print(f"Note: Using ALL predictions for all metrics to match DVIS-DAQ evaluation methodology")
    else:
        filtered_preds = preds
        print(f"Using all {len(preds)} predictions (no confidence threshold applied)")
    
    if len(filtered_preds) == 0:
        print("WARNING: No predictions meet the confidence threshold!")
        print("This will result in all metrics being 0.")
        # Create empty CSV with headers
        fieldnames = ["video_id", "category_id", "file_name", "frame_IoU", "frame_boundary_Fmeasure", 
                     "video_IoU", "video_boundary_Fmeasure", "dataset_IoU", "dataset_boundary_Fmeasure", 
                     "ap10_track", "ap25_track", "ap50_track", "ap75_track", "ap95_track", 
                     "ap50_track_Aweighted", "ap50_track_small", "ap50_track_medium", "ap50_track_large",
                     "ap10_track_per_cat", "ap25_track_per_cat", "ap50_track_per_cat", "ap75_track_per_cat", "ap95_track_per_cat", 
                     "ap_instance_Aweighted", "ap50_instance_Aweighted", "ap75_instance_Aweighted", "aps_instance_Aweighted", "apm_instance_Aweighted", "apl_instance_Aweighted", "ar1_instance", "ar10_instance",
                     "ap_instance_per_cat", "ap50_instance_per_cat", "ap75_instance_per_cat", "aps_instance_per_cat", "apm_instance_per_cat", "apl_instance_per_cat", "ar1_instance_per_cat", "ar10_instance_per_cat",
                     "predicted_category_id", "predicted_category_name", "predicted_score", 
                     "gt_category_id", "gt_category_name"]
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        return

    if fast_mode:
        print("FAST MODE: Computing DVIS-DAQ COCO metrics only (skipping track-level metrics, boundary F-measure, and temporal analysis)")
        # In fast mode, suggest using confidence threshold for even faster processing
        if confidence_threshold == 0.0:
            print("  💡 Tip: Consider using --confidence-threshold 0.05 in fast mode for even faster processing")
    
    gt_category_ids = {cat['id'] for cat in gts['categories']}
    gt_category_names = {cat['id']: cat['name'] for cat in gts['categories']}

    # In fast mode, skip all track-level processing and go straight to DVIS-DAQ COCO metrics
    if fast_mode:
        print("Computing DVIS-DAQ COCO metrics...")
        
        # Try to use DVIS-DAQ evaluation first
        if DVIS_DAQ_AVAILABLE:
            print("Using DVIS-DAQ evaluation methodology...")
            # Use ALL predictions (not filtered) to match DVIS-DAQ built-in evaluation
            dvis_daq_metrics = compute_dvis_daq_metrics_wrapper(preds, gts, val_json)
            if dvis_daq_metrics is not None:
                coco_metrics = dvis_daq_metrics
                print("Successfully used DVIS-DAQ evaluation!")
            else:
                print("DVIS-DAQ evaluation failed, falling back to custom implementation...")
                coco_metrics = compute_frame_level_coco_metrics(filtered_preds, gts)
        else:
            print("DVIS-DAQ not available, using custom frame-level evaluation...")
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
        
        print(f"DVIS-DAQ COCO Metrics (Fast Mode):")
        print(f"  AP: {ap:.4f}")
        print(f"  AP50: {ap50:.4f}")
        print(f"  AP75: {ap75:.4f}")
        print(f"  APs: {aps:.4f}")
        print(f"  APm: {apm:.4f}")
        print(f"  APl: {apl:.4f}")
        print(f"  AR1: {ar1:.4f}")
        print(f"  AR10: {ar10:.4f}")
        
        # Create minimal CSV with only COCO metrics
        results_dir = os.path.dirname(csv_path)
        dataset_csv_path = os.path.join(results_dir, f"{os.path.splitext(os.path.basename(csv_path))[0]}_dataset.csv")
        
        dataset_metrics_rows = [
            {"metric_name": "ap_instance_Aweighted", "value": ap, "description": "Instance-level AP (DVIS-DAQ, area-weighted)"},
            {"metric_name": "ap50_instance_Aweighted", "value": ap50, "description": "Instance-level AP50 (DVIS-DAQ, area-weighted)"},
            {"metric_name": "ap75_instance_Aweighted", "value": ap75, "description": "Instance-level AP75 (DVIS-DAQ, area-weighted)"},
            {"metric_name": "aps_instance_Aweighted", "value": aps, "description": "Instance-level APs (DVIS-DAQ, small objects)"},
            {"metric_name": "apm_instance_Aweighted", "value": apm, "description": "Instance-level APm (DVIS-DAQ, medium objects)"},
            {"metric_name": "apl_instance_Aweighted", "value": apl, "description": "Instance-level APl (DVIS-DAQ, large objects)"},
            {"metric_name": "ar1_instance", "value": ar1, "description": "Instance-level AR1 (DVIS-DAQ)"},
            {"metric_name": "ar10_instance", "value": ar10, "description": "Instance-level AR10 (DVIS-DAQ)"}
        ]
        
        with open(dataset_csv_path, 'w', newline='') as csvfile:
            fieldnames = ["metric_name", "value", "description"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in dataset_metrics_rows:
                writer.writerow(row)
        
        print(f"\nFast mode results saved to: {dataset_csv_path}")
        
        # Compute simple confusion matrix
        print("\nComputing simple confusion matrix...")
        try:
            from confusion_mat_plot import simple_confusion_matrix, plot_confusion_matrix, print_class_metrics
            
            # Use simple confusion matrix (highest scoring prediction per video) with no confidence threshold
            cm, class_names, class_metrics = simple_confusion_matrix(
                preds, gts, 
                confidence_threshold=0.0
            )
            
            # Plot confusion matrix
            plot_confusion_matrix(cm, class_names, cm_plot_path, "Simple VIS Confusion Matrix", 
                                method="simple", conf_threshold=0.0)
            
            # Print per-class metrics
            overall_metrics = print_class_metrics(class_metrics, "simple")
            
            print(f"Confusion matrix saved to: {cm_plot_path}")
            
        except ImportError:
            print("Warning: confusion_mat_plot.py not found. Skipping confusion matrix computation.")
        
        return  # Exit early in fast mode
    
    # Build GT dict: {video_id: [annotations]} - only for non-fast mode
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
    all_frame_fmeasures = []
    frame_metrics_rows = []
    video_metrics_rows = []
    category_metrics_rows = []
    dataset_metrics_rows = []
    video_ious = {}
    video_fmeasures = {}

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

    for video_id in tqdm(video_ids, desc="Processing videos"):
        video_frame_ious_per_video = []
        video_frame_fmeasures_per_video = []

        for cat_id in gt_category_ids:
            gt_tracks = [ann for ann in gt_dict.get(video_id, []) if ann['category_id'] == cat_id]
            pred_tracks = [pred for pred in pred_by_video.get(video_id, []) if pred['category_id'] == cat_id]

            if not gt_tracks or not pred_tracks:
                continue
            # Build cost matrix: rows=GT, cols=Pred, value=1-mean IoU
            num_gt = len(gt_tracks)
            num_pred = len(pred_tracks)
            cost_matrix = np.ones((num_gt, num_pred))
            iou_matrix = np.zeros((num_gt, num_pred))
            f_matrix = np.zeros((num_gt, num_pred))
            for i, gt_ann in enumerate(gt_tracks):
                gt_segs = gt_ann['segmentations']
                # gt_height = gt_ann['height']
                # gt_width = gt_ann['width']
                for j, pred_ann in enumerate(pred_tracks):
                    pred_segs = pred_ann['segmentations']
                    # pred_height = pred_ann['height']
                    # pred_width = pred_ann['width']
                    length = min(len(pred_segs), len(gt_segs))
                    frame_ious = []
                    frame_fmeasures = []
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
                        if not fast_mode:
                            fmeasure = boundary_f_measure(pred_mask, gt_mask)
                        else:
                            fmeasure = 0.0  # Skip expensive boundary calculation
                        frame_ious.append(iou)
                        frame_fmeasures.append(fmeasure)
                    mean_iou = np.mean(frame_ious) if frame_ious else 0.0
                    mean_f = np.mean(frame_fmeasures) if frame_fmeasures else 0.0
                    cost_matrix[i, j] = 1 - mean_iou
                    iou_matrix[i, j] = mean_iou
                    f_matrix[i, j] = mean_f
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
            # For each matched pair, compute per-frame and per-track metrics
            
            for gt_idx, pred_idx in zip(gt_indices, pred_indices):
                gt_ann = gt_tracks[gt_idx]
                pred_ann = pred_tracks[pred_idx]
                gt_segs = gt_ann['segmentations']
                pred_segs = pred_ann['segmentations']
                # gt_height = gt_ann['height']
                # gt_width = gt_ann['width']
                # pred_height = pred_ann['height']
                # pred_width = pred_ann['width']
                length = min(len(pred_segs), len(gt_segs))
                frame_ious = []
                frame_fmeasures = []
                category_id, file_name = gt_meta[(video_id, gt_ann['id'])]
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
                    if not fast_mode:
                        fmeasure = boundary_f_measure(pred_mask, gt_mask)
                    else:
                        fmeasure = 0.0  # Skip expensive boundary calculation
                    frame_ious.append(iou)
                    frame_fmeasures.append(fmeasure)
                    all_frame_ious.append(iou)
                    all_frame_fmeasures.append(fmeasure)
                    frame_metrics_rows.append({
                        "video_id": video_id,
                        "frame_idx": frame_idx,
                        "prediction_id": pred_ann.get('id', f"pred_{video_id}_{pred_idx}"),
                        "gt_id": gt_ann.get('id', f"gt_{video_id}_{gt_idx}"),
                        "category_id": category_id,
                        "file_name": file_name,
                        "frame_IoU": iou,
                        "frame_boundary_Fmeasure": fmeasure,
                        "predicted_category_id": pred_ann.get('category_id'),
                        "predicted_score": pred_ann.get('score', 0.0),
                        "gt_category_id": gt_ann.get('category_id'),
                        "gt_category_name": gt_category_names.get(gt_ann.get('category_id'), f"ID {gt_ann.get('category_id')}")
                    })
                mean_iou = np.mean(frame_ious) if frame_ious else 0.0
                mean_f = np.mean(frame_fmeasures) if frame_fmeasures else 0.0
                video_frame_ious_per_video.append(mean_iou)
                video_frame_fmeasures_per_video.append(mean_f)

        video_ious[video_id] = np.mean(video_frame_ious_per_video) if video_frame_ious_per_video else 0.0
        video_fmeasures[video_id] = np.mean(video_frame_fmeasures_per_video) if video_frame_fmeasures_per_video else 0.0
        
        # Add video-level metrics
        pred_info = video_pred_info[video_id]
        video_metrics_rows.append({
            "video_id": video_id,
            "video_IoU": video_ious[video_id],
            "video_boundary_Fmeasure": video_fmeasures[video_id],
            "predicted_category_id": pred_info['predicted_category_id'],
            "predicted_category_name": pred_info['predicted_category_name'],
            "predicted_score": pred_info['predicted_score'],
            "gt_category_ids": ','.join(str(x) for x in pred_info['gt_category_ids']),
            "gt_category_names": ','.join(pred_info['gt_category_names']),
            "num_frames": len([row for row in frame_metrics_rows if row['video_id'] == video_id]),
            "num_predictions": len([row for row in frame_metrics_rows if row['video_id'] == video_id and row['predicted_category_id'] is not None])
        })

    dataset_iou = np.mean(list(video_ious.values())) if video_ious else 0.0
    dataset_fmeasure = np.mean(list(video_fmeasures.values())) if video_fmeasures else 0.0

    # Compute standard COCO-style metrics (frame-level)
    print("Computing standard COCO metrics...")
    
    # Try to use DVIS-DAQ evaluation first
    if DVIS_DAQ_AVAILABLE:
        print("Using DVIS-DAQ evaluation methodology...")
        # Use ALL predictions (not filtered) to match DVIS-DAQ built-in evaluation
        dvis_daq_metrics = compute_dvis_daq_metrics_wrapper(preds, gts, val_json)
        if dvis_daq_metrics is not None:
            coco_metrics = dvis_daq_metrics
            print("Successfully used DVIS-DAQ evaluation!")
        else:
            print("DVIS-DAQ evaluation failed, falling back to custom implementation...")
            print("Note: COCO metrics now properly verify class matches (fixed implementation)")
            coco_metrics = compute_frame_level_coco_metrics(filtered_preds, gts)
    else:
        print("Note: COCO metrics now properly verify class matches (fixed implementation)")
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
    
    print(f"Standard COCO Metrics (DVIS-DAQ):")
    print(f"  AP: {ap:.4f}")
    print(f"  AP50 (Instance-level, Area-weighted): {ap50:.4f}")
    print(f"  AP75 (Instance-level, Area-weighted): {ap75:.4f}")
    print(f"  APs: {aps:.4f}")
    print(f"  APm: {apm:.4f}")
    print(f"  APl: {apl:.4f}")
    print(f"  AR1: {ar1:.4f}")
    print(f"  AR10: {ar10:.4f}")
    
    # Extract per-category instance-level metrics
    print("Extracting per-category instance-level metrics...")
    if DVIS_DAQ_AVAILABLE and dvis_daq_metrics is not None:
        print(f"DVIS-DAQ metrics keys: {list(dvis_daq_metrics.keys())}")
        if 'per_category_metrics' in dvis_daq_metrics:
            # Use the new per-category metrics from DVIS-DAQ evaluation
            per_cat_instance_metrics = {}
            print(f"Found per_category_metrics with {len(dvis_daq_metrics['per_category_metrics'])} categories")
            for cat_name, cat_metrics in dvis_daq_metrics['per_category_metrics'].items():
                print(f"  Category {cat_name}: {list(cat_metrics.keys())}")
                # Find the category ID for this category name
                cat_id = None
                for cid, cname in gt_category_names.items():
                    if cname == cat_name:
                        cat_id = cid
                        break
                
                if cat_id is not None:
                    per_cat_instance_metrics[cat_id] = cat_metrics
                    print(f"    Mapped to category ID {cat_id}")
                else:
                    print(f"    Could not find category ID for {cat_name}")
            
            print(f"Extracted per-category instance-level metrics for {len(per_cat_instance_metrics)} categories from DVIS-DAQ")
            print(f"Available category IDs in per_cat_instance_metrics: {list(per_cat_instance_metrics.keys())}")
            print(f"Available category names in gt_category_names: {list(gt_category_names.values())}")
        elif 'evaluator' in dvis_daq_metrics:
            # Fallback to old method
            per_cat_instance_metrics = extract_per_category_instance_metrics(
                dvis_daq_metrics['evaluator'], 
                gt_category_ids, 
                gt_category_names
            )
            print(f"Extracted per-category instance-level metrics for {len(per_cat_instance_metrics)} categories using old method")
        else:
            per_cat_instance_metrics = {}
            print("Could not extract per-category instance-level metrics (no evaluator or per_category_metrics available)")
    else:
        per_cat_instance_metrics = {}
        print("Could not extract per-category instance-level metrics (DVIS-DAQ not available)")
    
    # Compute video-level mAP - compute all thresholds even in fast mode
    # Fast mode only skips boundary F-measure computation
    map10_scores = []
    map25_scores = []
    map50_scores = []
    map75_scores = []
    map95_scores = []
    ap_per_category = {}
    print(f"Processing categories: {list(gt_category_ids)}")
    for cat_id in gt_category_ids:
        ap10 = compute_ap_at_iou(pred_by_video, gt_dict, 0.1, cat_id)
        ap25 = compute_ap_at_iou(pred_by_video, gt_dict, 0.25, cat_id)
        ap50_video = compute_ap_at_iou(pred_by_video, gt_dict, 0.5, cat_id)  # Renamed to avoid conflict
        ap75 = compute_ap_at_iou(pred_by_video, gt_dict, 0.75, cat_id)
        ap95 = compute_ap_at_iou(pred_by_video, gt_dict, 0.95, cat_id)
        map10_scores.append(ap10)
        map25_scores.append(ap25)
        map50_scores.append(ap50_video)
        map75_scores.append(ap75)
        map95_scores.append(ap95)
        ap_per_category[cat_id] = {
            'AP@0.1': ap10,
            'AP@0.25': ap25,
            'AP@0.5': ap50_video,
            'AP@0.75': ap75,
            'AP@0.95': ap95,
        }
        
        # Add category-level metrics
        cat_name = gt_category_names.get(cat_id, f"ID {cat_id}")
        cat_metrics_row = {
            "category_id": cat_id,
            "category_name": cat_name,
            "ap10_track_per_cat": ap10,
            "ap25_track_per_cat": ap25,
            "ap50_track_per_cat": ap50_video,
            "ap75_track_per_cat": ap75,
            "ap95_track_per_cat": ap95,
            "num_videos": len([v for v in video_ids if any(ann['category_id'] == cat_id for ann in gt_dict.get(v, []))]),
            "num_frames": len([row for row in frame_metrics_rows if row['gt_category_id'] == cat_id]),
            "num_gt_objects": len([ann for ann in gts['annotations'] if ann['category_id'] == cat_id])
        }
        
        # Add instance-level per-category metrics if available
        if cat_id in per_cat_instance_metrics:
            cat_metrics = per_cat_instance_metrics[cat_id]
            print(f"    Adding instance metrics for category {cat_id} ({cat_name}): {list(cat_metrics.keys())}")
            # Convert percentage values to decimal (DVIS-DAQ returns percentages)
            cat_metrics_row.update({
                "ap_instance_per_cat": cat_metrics.get('AP', None) / 100.0 if cat_metrics.get('AP') is not None else None,
                "ap50_instance_per_cat": cat_metrics.get('AP50', None) / 100.0 if cat_metrics.get('AP50') is not None else None,
                "ap75_instance_per_cat": cat_metrics.get('AP75', None) / 100.0 if cat_metrics.get('AP75') is not None else None,
                "aps_instance_per_cat": cat_metrics.get('APs', None) / 100.0 if cat_metrics.get('APs') is not None else None,
                "apm_instance_per_cat": cat_metrics.get('APm', None) / 100.0 if cat_metrics.get('APm') is not None else None,
                "apl_instance_per_cat": cat_metrics.get('APl', None) / 100.0 if cat_metrics.get('APl') is not None else None,
                "ar1_instance_per_cat": cat_metrics.get('AR1', None) / 100.0 if cat_metrics.get('AR1') is not None else None,
                "ar10_instance_per_cat": cat_metrics.get('AR10', None) / 100.0 if cat_metrics.get('AR10') is not None else None
            })
        else:
            print(f"    No instance metrics found for category {cat_id} ({cat_name})")
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
    map10 = np.mean(map10_scores) if map10_scores else 0.0
    map25 = np.mean(map25_scores) if map25_scores else 0.0
    map50 = np.mean(map50_scores) if map50_scores else 0.0
    map75 = np.mean(map75_scores) if map75_scores else 0.0
    map95 = np.mean(map95_scores) if map95_scores else 0.0

    # Compute area-weighted mAP@0.5 (matching DVIS-DAQ area categories)
    print("Computing area-weighted mAP@0.5...")
    area_weighted_results = compute_area_weighted_map(pred_by_video, gt_dict, iou_threshold=0.5)
    map50_area_weighted = area_weighted_results['area_weighted_ap']
    map50_small = area_weighted_results['ap_small']
    map50_medium = area_weighted_results['ap_medium']
    map50_large = area_weighted_results['ap_large']
    
    print(f"Track-level AP50 (Area-weighted): {map50_area_weighted:.4f}")
    print(f"  Small objects (< 128²): {map50_small:.4f} ({area_weighted_results['num_small_gt']} GT)")
    print(f"  Medium objects (128²-256²): {map50_medium:.4f} ({area_weighted_results['num_medium_gt']} GT)")
    print(f"  Large objects (> 256²): {map50_large:.4f} ({area_weighted_results['num_large_gt']} GT)")

    # Compute temporal consistency metrics (skip in fast mode)
    if not fast_mode:
        print("Computing temporal consistency metrics...")
        temporal_metrics = compute_temporal_consistency_metrics(pred_by_video, gt_dict)
        print(f"Temporal Consistency Metrics:")
        print(f"  Track Completeness: {temporal_metrics['track_completeness']:.4f}")
        print(f"  Temporal IoU Stability: {temporal_metrics['temporal_iou_stability']:.4f}")
        print(f"  Track Fragmentation: {temporal_metrics['track_fragmentation']:.4f}")
        print(f"  Mean Track Length: {temporal_metrics['mean_track_length']:.1f} frames")

        # Compute MOTA, IDF1, HOTA
        print("Computing MOT metrics...")
        mota_metrics = compute_mota_metrics(pred_by_video, gt_dict)
        idf1_metrics = compute_idf1_metrics(pred_by_video, gt_dict)
        hota_metrics = compute_hota_metrics(pred_by_video, gt_dict)

        print(f"MOTA: {mota_metrics['MOTA']:.4f}")
        print(f"MOTP: {mota_metrics['MOTP']:.4f}")
        print(f"IDF1: {idf1_metrics['IDF1']:.4f}")
        print(f"HOTA: {hota_metrics['HOTA']:.4f}")
        print(f"DetA: {hota_metrics['DetA']:.4f}")
        print(f"AssA: {hota_metrics['AssA']:.4f}")
    else:
        print("Fast mode: Skipping temporal consistency and tracking metrics...")
        # Create empty tracking metrics for consistency
        temporal_metrics = {
            'track_completeness': None,
            'temporal_iou_stability': None,
            'track_fragmentation': None,
            'mean_track_length': None
        }
        mota_metrics = {
            'MOTA': None,
            'MOTP': None,
            'total_gt': 0,
            'total_fp': 0,
            'total_fn': 0,
            'total_idsw': 0,
            'total_matches': 0
        }
        idf1_metrics = {
            'IDF1': None,
            'IDP': None,
            'IDR': None,
            'total_gt_objects': 0,
            'total_pred_objects': 0,
            'total_correct_assignments': 0
        }
        hota_metrics = {
            'HOTA': None,
            'DetA': None,
            'AssA': None,
            'num_videos': 0
        }

    # Add dataset-level metrics
    dataset_metrics_rows = [
        {"metric_name": "dataset_IoU", "value": dataset_iou, "description": "Average IoU across all videos"},
        {"metric_name": "dataset_boundary_Fmeasure", "value": dataset_fmeasure, "description": "Average boundary F-measure across all videos"},
        {"metric_name": "ap10_track", "value": map10, "description": "Track-level mAP@0.1 (average across categories)"},
        {"metric_name": "ap25_track", "value": map25, "description": "Track-level mAP@0.25 (average across categories)"},
        {"metric_name": "ap50_track", "value": map50, "description": "Track-level mAP@0.5 (average across categories)"},
        {"metric_name": "ap75_track", "value": map75, "description": "Track-level mAP@0.75 (average across categories)"},
        {"metric_name": "ap95_track", "value": map95, "description": "Track-level mAP@0.95 (average across categories)"},
        {"metric_name": "ap50_track_Aweighted", "value": map50_area_weighted, "description": "Track-level AP50 (area-weighted)"},
        {"metric_name": "ap50_track_small", "value": map50_small, "description": "Track-level AP50 (small objects)"},
        {"metric_name": "ap50_track_medium", "value": map50_medium, "description": "Track-level AP50 (medium objects)"},
        {"metric_name": "ap50_track_large", "value": map50_large, "description": "Track-level AP50 (large objects)"},
        {"metric_name": "ap_instance_Aweighted", "value": ap, "description": "Instance-level AP (DVIS-DAQ, area-weighted)"},
        {"metric_name": "ap50_instance_Aweighted", "value": ap50, "description": "Instance-level AP50 (DVIS-DAQ, area-weighted)"},
        {"metric_name": "ap75_instance_Aweighted", "value": ap75, "description": "Instance-level AP75 (DVIS-DAQ, area-weighted)"},
        {"metric_name": "aps_instance_Aweighted", "value": aps, "description": "Instance-level APs (DVIS-DAQ, small objects)"},
        {"metric_name": "apm_instance_Aweighted", "value": apm, "description": "Instance-level APm (DVIS-DAQ, medium objects)"},
        {"metric_name": "apl_instance_Aweighted", "value": apl, "description": "Instance-level APl (DVIS-DAQ, large objects)"},
        {"metric_name": "ar1_instance", "value": ar1, "description": "Instance-level AR1 (DVIS-DAQ)"},
        {"metric_name": "ar10_instance", "value": ar10, "description": "Instance-level AR10 (DVIS-DAQ)"},
        {"metric_name": "MOTA", "value": mota_metrics['MOTA'], "description": "Multiple Object Tracking Accuracy"},
        {"metric_name": "MOTP", "value": mota_metrics['MOTP'], "description": "Multiple Object Tracking Precision"},
        {"metric_name": "IDF1", "value": idf1_metrics['IDF1'], "description": "ID F1-Score"},
        {"metric_name": "HOTA", "value": hota_metrics['HOTA'], "description": "Higher Order Tracking Accuracy"},
        {"metric_name": "DetA", "value": hota_metrics['DetA'], "description": "Detection Accuracy (HOTA component)"},
        {"metric_name": "AssA", "value": hota_metrics['AssA'], "description": "Association Accuracy (HOTA component)"},
        {"metric_name": "track_completeness", "value": temporal_metrics['track_completeness'], "description": "Fraction of frames with valid predictions"},
        {"metric_name": "temporal_iou_stability", "value": temporal_metrics['temporal_iou_stability'], "description": "1 - std(IoU) across frames"},
        {"metric_name": "track_fragmentation", "value": temporal_metrics['track_fragmentation'], "description": "Number of gaps in tracks (lower is better)"},
        {"metric_name": "mean_track_length", "value": temporal_metrics['mean_track_length'], "description": "Average number of frames per track"}
    ]

    # Print per-video and dataset metrics
    for video_id in video_ious:
        pred_info = video_pred_info[video_id]
        print(f"Video {video_id}: Mean IoU = {video_ious[video_id]:.4f}, Mean Boundary F = {video_fmeasures[video_id]:.4f}, "
              f"Predicted Cat: {pred_info['predicted_category_id']} ({pred_info['predicted_category_name']}), "
              f"Score: {pred_info['predicted_score']}, "
              f"GT Cat(s): {pred_info['gt_category_ids']} ({pred_info['gt_category_names']})")
    print(f"Overall Dataset: Mean IoU = {dataset_iou:.4f}, Mean Boundary F = {dataset_fmeasure:.4f}")
    print(f"Video-level mAP@0.1: {map10:.4f}, mAP@0.25: {map25:.4f}, mAP@0.5: {map50:.4f}, mAP@0.75: {map75:.4f}, mAP@0.95: {map95:.4f}")
    print(f"Frame-level COCO AP: {ap:.4f}, AP50: {ap50:.4f}, AP75: {ap75:.4f}")
    print(f"Frame-level COCO APs: {aps:.4f}, APm: {apm:.4f}, APl: {apl:.4f}")
    print(f"Frame-level COCO AR1: {ar1:.4f}, AR10: {ar10:.4f}")
    
    # Print tracking metrics only if they were computed
    if not fast_mode:
        print(f"MOTA: {mota_metrics['MOTA']:.4f}, MOTP: {mota_metrics['MOTP']:.4f}, IDF1: {idf1_metrics['IDF1']:.4f}, HOTA: {hota_metrics['HOTA']:.4f}")
    else:
        print("Tracking metrics: Skipped in fast mode")

    # Save separate CSV files
    csv_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Save frame.csv
    frame_csv_path = os.path.join(csv_dir, f"{base_name}_frame.csv")
    with open(frame_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["video_id", "frame_idx", "prediction_id", "gt_id", "category_id", "file_name", "frame_IoU", "frame_boundary_Fmeasure", "predicted_category_id", "predicted_score", "gt_category_id", "gt_category_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_metrics_rows:
            writer.writerow(row)
    
    # Save video.csv
    video_csv_path = os.path.join(csv_dir, f"{base_name}_video.csv")
    with open(video_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["video_id", "video_IoU", "video_boundary_Fmeasure", "predicted_category_id", "predicted_category_name", "predicted_score", "gt_category_ids", "gt_category_names", "num_frames", "num_predictions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in video_metrics_rows:
            writer.writerow(row)
    
    # Save category.csv
    category_csv_path = os.path.join(csv_dir, f"{base_name}_category.csv")
    with open(category_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["category_id", "category_name", "ap10_track_per_cat", "ap25_track_per_cat", "ap50_track_per_cat", "ap75_track_per_cat", "ap95_track_per_cat", "ap_instance_per_cat", "ap50_instance_per_cat", "ap75_instance_per_cat", "aps_instance_per_cat", "apm_instance_per_cat", "apl_instance_per_cat", "ar1_instance_per_cat", "ar10_instance_per_cat", "num_videos", "num_frames", "num_gt_objects"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in category_metrics_rows:
            writer.writerow(row)
    
    # Debug: Print category metrics for verification
    print(f"\nCategory metrics summary:")
    print(f"Total categories processed: {len(category_metrics_rows)}")
    for row in category_metrics_rows:
        cat_name = row.get('category_name', 'Unknown')
        ap50_track = row.get('ap50_track_per_cat', 'N/A')
        ap50_instance = row.get('ap50_instance_per_cat', 'N/A')
        print(f"  {cat_name}: AP50(Track)={ap50_track}, AP50(Instance)={ap50_instance}")
    
    # Save dataset.csv
    dataset_csv_path = os.path.join(csv_dir, f"{base_name}_dataset.csv")
    with open(dataset_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["metric_name", "value", "description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset_metrics_rows:
            writer.writerow(row)
    
    print(f"\nSaved separate CSV files:")
    print(f"  Frame metrics: {frame_csv_path}")
    print(f"  Video metrics: {video_csv_path}")
    print(f"  Category metrics: {category_csv_path}")
    print(f"  Dataset metrics: {dataset_csv_path}")
    
    # Also save the original combined CSV for backward compatibility
    print(f"  Combined metrics: {csv_path}")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["video_id", "frame_idx", "prediction_id", "gt_id", "category_id", "file_name", "frame_IoU", "frame_boundary_Fmeasure", "video_IoU", "video_boundary_Fmeasure", "dataset_IoU", "dataset_boundary_Fmeasure", "ap10_track", "ap25_track", "ap50_track", "ap75_track", "ap95_track", "ap50_track_Aweighted", "ap50_track_small", "ap50_track_medium", "ap50_track_large", "ap10_track_per_cat", "ap25_track_per_cat", "ap50_track_per_cat", "ap75_track_per_cat", "ap95_track_per_cat", "ap_instance_Aweighted", "ap50_instance_Aweighted", "ap75_instance_Aweighted", "aps_instance_Aweighted", "apm_instance_Aweighted", "apl_instance_Aweighted", "ar1_instance", "ar10_instance", "ap_instance_per_cat", "ap50_instance_per_cat", "ap75_instance_per_cat", "aps_instance_per_cat", "apm_instance_per_cat", "apl_instance_per_cat", "ar1_instance_per_cat", "ar10_instance_per_cat", "track_completeness", "temporal_iou_stability", "track_fragmentation", "mean_track_length", "predicted_category_id", "predicted_category_name", "predicted_score", "gt_category_id", "gt_category_name", "MOTA", "MOTP", "IDF1", "HOTA", "DetA", "AssA"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Add frame-level rows
        for row in frame_metrics_rows:
            # Add video and dataset level metrics to each frame row
            video_id = row["video_id"]
            row["video_IoU"] = video_ious[video_id]
            row["video_boundary_Fmeasure"] = video_fmeasures[video_id]
            row["dataset_IoU"] = dataset_iou
            row["dataset_boundary_Fmeasure"] = dataset_fmeasure
            row["ap10_track"] = map10
            row["ap25_track"] = map25
            row["ap50_track"] = map50
            row["ap75_track"] = map75
            row["ap95_track"] = map95
            row["ap50_track_Aweighted"] = map50_area_weighted
            row["ap50_track_small"] = map50_small
            row["ap50_track_medium"] = map50_medium
            row["ap50_track_large"] = map50_large
            row["ap_instance_Aweighted"] = ap
            row["ap50_instance_Aweighted"] = ap50
            row["ap75_instance_Aweighted"] = ap75
            row["aps_instance_Aweighted"] = aps
            row["apm_instance_Aweighted"] = apm
            row["apl_instance_Aweighted"] = apl
            row["ar1_instance"] = ar1
            row["ar10_instance"] = ar10
            row["track_completeness"] = temporal_metrics['track_completeness']
            row["temporal_iou_stability"] = temporal_metrics['temporal_iou_stability']
            row["track_fragmentation"] = temporal_metrics['track_fragmentation']
            row["mean_track_length"] = temporal_metrics['mean_track_length']
            row["MOTA"] = mota_metrics['MOTA']
            row["MOTP"] = mota_metrics['MOTP']
            row["IDF1"] = idf1_metrics['IDF1']
            row["HOTA"] = hota_metrics['HOTA']
            row["DetA"] = hota_metrics['DetA']
            row["AssA"] = hota_metrics['AssA']
            
            # Add per-category metrics
            cat_id = row["category_id"]
            ap_cat = ap_per_category.get(cat_id, {'AP@0.1': None, 'AP@0.25': None, 'AP@0.5': None, 'AP@0.75': None, 'AP@0.95': None})
            row["ap10_track_per_cat"] = ap_cat['AP@0.1']
            row["ap25_track_per_cat"] = ap_cat['AP@0.25']
            row["ap50_track_per_cat"] = ap_cat['AP@0.5']
            row["ap75_track_per_cat"] = ap_cat['AP@0.75']
            row["ap95_track_per_cat"] = ap_cat['AP@0.95']
            
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
    print("\nComputing VIS confusion matrix...")
    
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
        
        print(f"\nConfusion matrix saved to: {cm_plot_path}")
        
    except ImportError:
        print("Warning: confusion_mat_plot.py not found. Skipping confusion matrix computation.")
        print("Please ensure confusion_mat_plot.py is in the same directory.")

# Note: plot_confusion_matrix_from_csv function has been moved to confusion_mat_plot.py



def compute_area_weighted_map(pred_by_video, gt_dict, iou_threshold=0.5):
    """
    Compute area-weighted mAP@0.5 that matches DVIS-DAQ area categories exactly.
    
    DVIS-DAQ area categories:
    - Small: area < 128² = 16,384 pixels
    - Medium: 128² ≤ area < 256² = 16,384 to 65,536 pixels  
    - Large: area ≥ 256² = 65,536 pixels
    
    This function computes AP for each area category and returns the weighted average.
    
    Args:
        pred_by_video: Dictionary mapping video_id to list of predictions
        gt_dict: Dictionary mapping video_id to list of ground truths
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with area-weighted AP and individual area APs
    """
    # DVIS-DAQ area ranges (exact same as in oviseval.py)
    area_ranges = {
        'small': [0, 128**2],      # 0 to 16,384 pixels
        'medium': [128**2, 256**2], # 16,384 to 65,536 pixels
        'large': [256**2, 1e5**2]   # 65,536+ pixels
    }
    
    # Compute average area for each prediction and GT
    def compute_avg_area(segmentation_list):
        """Compute average area across all frames in a track"""
        areas = []
        for seg in segmentation_list:
            if seg is not None:
                mask = decode_rle(seg)
                area = np.sum(mask)
                areas.append(area)
        return np.mean(areas) if areas else 0
    
    # Categorize predictions and ground truths by area
    categorized_preds = {'small': [], 'medium': [], 'large': []}
    categorized_gts = {'small': [], 'medium': [], 'large': []}
    
    # Process each video
    for video_id in pred_by_video:
        if video_id not in gt_dict:
            continue
            
        video_preds = pred_by_video[video_id]
        video_gts = gt_dict[video_id]
        
        # Categorize predictions for this video
        for pred in video_preds:
            avg_area = compute_avg_area(pred['segmentations'])
            for area_name, (min_area, max_area) in area_ranges.items():
                if min_area <= avg_area < max_area:
                    categorized_preds[area_name].append(pred)
                    break
        
        # Categorize ground truths for this video
        for gt in video_gts:
            avg_area = compute_avg_area(gt['segmentations'])
            for area_name, (min_area, max_area) in area_ranges.items():
                if min_area <= avg_area < max_area:
                    categorized_gts[area_name].append(gt)
                    break
    
    # Compute AP for each area category using track-level evaluation
    area_aps = {}
    total_weight = 0
    weighted_sum = 0
    
    for area_name in ['small', 'medium', 'large']:
        area_preds = categorized_preds[area_name]
        area_gts = categorized_gts[area_name]
        
        if len(area_gts) == 0:
            area_aps[area_name] = 0.0
            continue
        
        # Group by category_id for track-level evaluation
        area_preds_by_cat = defaultdict(list)
        area_gts_by_cat = defaultdict(list)
        
        for pred in area_preds:
            area_preds_by_cat[pred['category_id']].append(pred)
        
        for gt in area_gts:
            area_gts_by_cat[gt['category_id']].append(gt)
        
        # Compute AP for each category and average
        cat_aps = []
        for cat_id in set(list(area_preds_by_cat.keys()) + list(area_gts_by_cat.keys())):
            cat_preds = area_preds_by_cat[cat_id]
            cat_gts = area_gts_by_cat[cat_id]
            
            if len(cat_gts) == 0:
                continue
                
            # Use the same track-level AP computation as other metrics
            cat_ap = compute_ap_at_iou_threshold_track_level(cat_preds, cat_gts, iou_threshold)
            cat_aps.append(cat_ap)
        
        area_ap = np.mean(cat_aps) if cat_aps else 0.0
        area_aps[area_name] = area_ap
        
        # Weight by number of ground truths in this category
        weight = len(area_gts)
        total_weight += weight
        weighted_sum += area_ap * weight
    
    # Compute weighted average AP
    area_weighted_ap = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    return {
        'area_weighted_ap': area_weighted_ap,
        'ap_small': area_aps['small'],
        'ap_medium': area_aps['medium'],
        'ap_large': area_aps['large'],
        'num_small_gt': len(categorized_gts['small']),
        'num_medium_gt': len(categorized_gts['medium']),
        'num_large_gt': len(categorized_gts['large']),
        'total_gt': total_weight
    }

def extract_per_category_instance_metrics(coco_evaluator, cat_ids, cat_names):
    """
    Extract per-category instance-level metrics from DVIS-DAQ evaluation.
    
    Args:
        coco_evaluator: The COCO evaluator object with computed results
        cat_ids: List of category IDs
        cat_names: List of category names (matching cat_ids)
        
    Returns:
        Dictionary with per-category metrics
    """
    per_cat_metrics = {}
    
    # Get the evaluation results
    eval_results = coco_evaluator.eval
    precision = eval_results['precision']  # Shape: (T,R,K,A,O,M)
    recall = eval_results['recall']        # Shape: (T,K,A,O,M)
    params = eval_results['params']
    
    # Get indices for different IoU thresholds
    iou_thresholds = params.iouThrs
    iou_50_idx = np.where(iou_thresholds == 0.5)[0][0]
    iou_75_idx = np.where(iou_thresholds == 0.75)[0][0]
    
    # Get indices for area ranges
    area_indices = {
        'all': 0,
        'small': 1, 
        'medium': 2,
        'large': 3
    }
    
    # Get maxDets index (usually 100)
    max_dets_idx = 2  # Default to maxDets=100
    
    for i, (cat_id, cat_name) in enumerate(zip(cat_ids, cat_names)):
        cat_metrics = {}
        
        # AP (overall) - average across all IoU thresholds
        ap_overall = precision[:, :, i, area_indices['all'], 0, max_dets_idx]
        ap_overall = ap_overall[ap_overall > -1]  # Remove -1 values
        cat_metrics['AP'] = np.mean(ap_overall) if len(ap_overall) > 0 else 0.0
        
        # AP50 - at IoU=0.5
        ap50 = precision[iou_50_idx, :, i, area_indices['all'], 0, max_dets_idx]
        ap50 = ap50[ap50 > -1]
        cat_metrics['AP50'] = np.mean(ap50) if len(ap50) > 0 else 0.0
        
        # AP75 - at IoU=0.75
        ap75 = precision[iou_75_idx, :, i, area_indices['all'], 0, max_dets_idx]
        ap75 = ap75[ap75 > -1]
        cat_metrics['AP75'] = np.mean(ap75) if len(ap75) > 0 else 0.0
        
        # APs, APm, APl - area-specific
        for area_name, area_idx in area_indices.items():
            if area_name == 'all':
                continue
            ap_area = precision[iou_50_idx, :, i, area_idx, 0, max_dets_idx]
            ap_area = ap_area[ap_area > -1]
            cat_metrics[f'AP{area_name[0]}'] = np.mean(ap_area) if len(ap_area) > 0 else 0.0
        
        # AR1, AR10 - recall with different maxDets
        ar1 = recall[:, i, area_indices['all'], 0, 0]  # maxDets=1
        ar1 = ar1[ar1 > -1]
        cat_metrics['AR1'] = np.mean(ar1) if len(ar1) > 0 else 0.0
        
        ar10 = recall[:, i, area_indices['all'], 0, 1]  # maxDets=10
        ar10 = ar10[ar10 > -1]
        cat_metrics['AR10'] = np.mean(ar10) if len(ar10) > 0 else 0.0
        
        per_cat_metrics[cat_id] = cat_metrics
    
    return per_cat_metrics

def compute_ap_at_iou_threshold_track_level(preds, gts, iou_threshold):
    """
    Compute AP at a specific IoU threshold using track-level evaluation.
    This is the same logic used in the main video-level AP computation.
    """
    if not gts:
        return 1.0 if not preds else 0.0
    
    if not preds:
        return 0.0
    
    # Build cost matrix: rows=GT, cols=Pred, value=1-mean IoU
    num_gt = len(gts)
    num_pred = len(preds)
    cost_matrix = np.ones((num_gt, num_pred))
    
    for i, gt_ann in enumerate(gts):
        gt_segs = gt_ann['segmentations']
        for j, pred_ann in enumerate(preds):
            pred_segs = pred_ann['segmentations']
            length = min(len(pred_segs), len(gt_segs))
            frame_ious = []
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
            mean_iou = np.mean(frame_ious) if frame_ious else 0.0
            cost_matrix[i, j] = 1 - mean_iou
    
    # Use Hungarian algorithm for optimal assignment
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    
    # Create list of (score, is_match) pairs
    score_match_pairs = []
    
    # Add matched pairs
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        score = preds[pred_idx].get('score', 0.0)
        mean_iou = 1 - cost_matrix[gt_idx, pred_idx]
        is_match = mean_iou >= iou_threshold
        score_match_pairs.append((score, is_match))
    
    # Add unmatched predictions as false positives
    matched_pred_indices = set(pred_indices)
    for i, pred in enumerate(preds):
        if i not in matched_pred_indices:
            score = pred.get('score', 0.0)
            score_match_pairs.append((score, False))
    
    # Sort by score (descending)
    score_match_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Compute precision-recall curve
    num_gt = len(gts)
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    
    for score, is_match in score_match_pairs:
        if is_match:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / num_gt if num_gt > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Compute AP using 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = np.max([p for p, r in zip(precisions, recalls) if r >= t]) if any(r >= t for r in recalls) else 0.0
        ap += p / 101
    
    return ap

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
    parser.add_argument('--test-thresholds', action='store_true',
                       help='Test different thresholds for confusion matrix optimization')

    parser.add_argument('--confidence-threshold', type=float, default=0.0,
                       help='Confidence threshold for filtering predictions (default: 0.0 = no threshold)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Run in fast mode (skip boundary F-measure and tracking metrics)')
    
    args = parser.parse_args()
    
    if args.test_thresholds:
        # Test different thresholds
        preds = load_json(args.results_json)
        gts = load_json(args.val_json)
        test_confusion_matrix_thresholds(preds, gts)
    else:
        # Run main analysis
        main(results_json=args.results_json, 
             val_json=args.val_json, 
             csv_path=args.csv_path,
             cm_plot_path=args.cm_plot_path,
             confidence_threshold=args.confidence_threshold,
             fast_mode=args.fast_mode)
        
