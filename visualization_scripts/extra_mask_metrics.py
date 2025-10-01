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
