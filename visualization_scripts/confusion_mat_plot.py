"""
Confusion Matrix Plotting for Video Instance Segmentation

This script provides functions to compute and plot confusion matrices for VIS evaluation.
Supports both simple (highest scoring prediction) and full tracking approaches.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict
from pycocotools import mask as maskUtils
import os

def decode_rle(rle):
    """Decode RLE mask."""
    return maskUtils.decode(rle)

def compute_iou(pred_mask, gt_mask):
    """Compute IoU between two masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 1.0

def simple_confusion_matrix(preds, gts, confidence_threshold=0.5):
    """
    Compute confusion matrix using only the highest-scoring prediction per video.
    
    Args:
        preds: List of predictions from model
        gts: Ground truth annotations
        confidence_threshold: Minimum confidence score
    
    Returns:
        confusion_matrix: numpy array of shape (num_classes, num_classes)
        class_names: List of class names
        metrics: Dictionary with precision, recall, f1 per class
    """
    # Get category information
    gt_category_ids = {cat['id'] for cat in gts['categories']}
    gt_category_names = {cat['id']: cat['name'] for cat in gts['categories']}
    class_names = [gt_category_names[cat_id] for cat_id in sorted(gt_category_ids)]
    
    # Filter predictions by confidence
    filtered_preds = [pred for pred in preds if pred.get('score', 0.0) >= confidence_threshold]
    print(f"Simple Confusion Matrix: {len(preds)} total predictions, {len(filtered_preds)} after confidence threshold {confidence_threshold}")
    
    # Group by video
    gt_by_video = defaultdict(list)
    pred_by_video = defaultdict(list)
    
    for ann in gts['annotations']:
        gt_by_video[ann['video_id']].append(ann)
    
    for pred in filtered_preds:
        pred_by_video[pred['video_id']].append(pred)
    
    # Initialize confusion matrix
    num_classes = len(gt_category_ids)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Track metrics per class
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    # Process each video
    for video_id in set(list(gt_by_video.keys()) + list(pred_by_video.keys())):
        gt_tracks = gt_by_video[video_id]
        pred_tracks = pred_by_video[video_id]
        
        # Get highest scoring prediction
        if pred_tracks:
            best_pred = max(pred_tracks, key=lambda p: p.get('score', 0.0))
            pred_cat = best_pred['category_id']
        else:
            pred_cat = None
        
        # Get ground truth category (assuming single object per video)
        if gt_tracks:
            gt_cat = gt_tracks[0]['category_id']  # Take first GT
        else:
            gt_cat = None
        
        # Update confusion matrix
        if gt_cat is not None and pred_cat is not None:
            gt_idx = sorted(gt_category_ids).index(gt_cat)
            pred_idx = sorted(gt_category_ids).index(pred_cat)
            confusion_matrix[gt_idx, pred_idx] += 1
            
            if gt_cat == pred_cat:
                class_metrics[gt_cat]['tp'] += 1
            else:
                class_metrics[gt_cat]['fn'] += 1
                class_metrics[pred_cat]['fp'] += 1
        elif gt_cat is not None:
            # False negative (no prediction)
            gt_idx = sorted(gt_category_ids).index(gt_cat)
            confusion_matrix[gt_idx, gt_idx] += 0  # No match
            class_metrics[gt_cat]['fn'] += 1
        elif pred_cat is not None:
            # False positive (no ground truth)
            class_metrics[pred_cat]['fp'] += 1
    
    # Compute per-class metrics
    metrics = {}
    for cat_id in sorted(gt_category_ids):
        cat_name = gt_category_names[cat_id]
        tp = class_metrics[cat_id]['tp']
        fp = class_metrics[cat_id]['fp']
        fn = class_metrics[cat_id]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[cat_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return confusion_matrix, class_names, metrics

def tracking_confusion_matrix(preds, gts, confidence_threshold=0.5, min_track_length=5, iou_threshold=0.5):
    """
    Compute confusion matrix using full tracking implementation.
    
    Args:
        preds: List of predictions from model
        gts: Ground truth annotations
        confidence_threshold: Minimum confidence score
        min_track_length: Minimum frames for a valid track
        iou_threshold: Minimum IoU for considering a match
    
    Returns:
        confusion_matrix: numpy array of shape (num_classes, num_classes)
        class_names: List of class names
        metrics: Dictionary with precision, recall, f1 per class
    """
    # Get category information
    gt_category_ids = {cat['id'] for cat in gts['categories']}
    gt_category_names = {cat['id']: cat['name'] for cat in gts['categories']}
    class_names = [gt_category_names[cat_id] for cat_id in sorted(gt_category_ids)]
    
    # Step 1: Filter predictions by confidence threshold
    filtered_preds = [pred for pred in preds if pred.get('score', 0.0) >= confidence_threshold]
    print(f"Tracking Confusion Matrix: {len(preds)} total predictions, {len(filtered_preds)} after confidence threshold {confidence_threshold}")
    
    # Step 2: Filter predictions by track length
    valid_preds = []
    for pred in filtered_preds:
        # Count non-None segmentations (track length)
        track_length = sum(1 for seg in pred['segmentations'] if seg is not None)
        if track_length >= min_track_length:
            valid_preds.append(pred)
    print(f"Tracking Confusion Matrix: {len(valid_preds)} predictions after track length filter (min {min_track_length} frames)")
    
    # Step 3: Group by video for per-video analysis
    gt_by_video = defaultdict(list)
    pred_by_video = defaultdict(list)
    
    for ann in gts['annotations']:
        gt_by_video[ann['video_id']].append(ann)
    
    for pred in valid_preds:
        pred_by_video[pred['video_id']].append(pred)
    
    # Step 4: Compute confusion matrix
    num_classes = len(gt_category_ids)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Track metrics per class
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for video_id in set(list(gt_by_video.keys()) + list(pred_by_video.keys())):
        gt_tracks = gt_by_video[video_id]
        pred_tracks = pred_by_video[video_id]
        
        # For each GT track, find best matching prediction
        for gt_track in gt_tracks:
            gt_cat = gt_track['category_id']
            gt_segs = gt_track['segmentations']
            
            best_iou = -1
            best_pred_cat = None
            best_pred_idx = -1
            
            # Find prediction with highest IoU
            for i, pred_track in enumerate(pred_tracks):
                pred_cat = pred_track['category_id']
                pred_segs = pred_track['segmentations']
                
                # Compute temporal IoU
                length = min(len(pred_segs), len(gt_segs))
                ious = []
                
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    
                    if pred_rle is None and gt_rle is None:
                        ious.append(1.0)  # Both missing = perfect match
                    elif pred_rle is None or gt_rle is None:
                        ious.append(0.0)  # One missing = no match
                    else:
                        pred_mask = decode_rle(pred_rle)
                        gt_mask = decode_rle(gt_rle)
                        iou = compute_iou(pred_mask, gt_mask)
                        ious.append(iou)
                
                mean_iou = np.mean(ious) if ious else 0.0
                
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    best_pred_cat = pred_cat
                    best_pred_idx = i
            
            # Update confusion matrix
            gt_idx = sorted(gt_category_ids).index(gt_cat)
            
            if best_iou >= iou_threshold:
                # True positive or correct classification
                pred_idx = sorted(gt_category_ids).index(best_pred_cat)
                confusion_matrix[gt_idx, pred_idx] += 1
                
                if gt_cat == best_pred_cat:
                    class_metrics[gt_cat]['tp'] += 1
                else:
                    # Wrong classification
                    class_metrics[gt_cat]['fn'] += 1
                    class_metrics[best_pred_cat]['fp'] += 1
            else:
                # False negative (no good match found)
                confusion_matrix[gt_idx, gt_idx] += 0  # No match
                class_metrics[gt_cat]['fn'] += 1
        
        # Count false positives (predictions not matched to any GT)
        matched_pred_indices = set()
        for gt_track in gt_tracks:
            gt_segs = gt_track['segmentations']
            
            for i, pred_track in enumerate(pred_tracks):
                if i in matched_pred_indices:
                    continue
                    
                pred_segs = pred_track['segmentations']
                length = min(len(pred_segs), len(gt_segs))
                ious = []
                
                for frame_idx in range(length):
                    pred_rle = pred_segs[frame_idx]
                    gt_rle = gt_segs[frame_idx]
                    
                    if pred_rle is None and gt_rle is None:
                        ious.append(1.0)
                    elif pred_rle is None or gt_rle is None:
                        ious.append(0.0)
                    else:
                        pred_mask = decode_rle(pred_rle)
                        gt_mask = decode_rle(gt_rle)
                        iou = compute_iou(pred_mask, gt_mask)
                        ious.append(iou)
                
                mean_iou = np.mean(ious) if ious else 0.0
                
                if mean_iou >= iou_threshold:
                    matched_pred_indices.add(i)
        
        # Count unmatched predictions as false positives
        for i, pred_track in enumerate(pred_tracks):
            if i not in matched_pred_indices:
                pred_cat = pred_track['category_id']
                class_metrics[pred_cat]['fp'] += 1
    
    # Compute per-class metrics
    metrics = {}
    for cat_id in sorted(gt_category_ids):
        cat_name = gt_category_names[cat_id]
        tp = class_metrics[cat_id]['tp']
        fp = class_metrics[cat_id]['fp']
        fn = class_metrics[cat_id]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[cat_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return confusion_matrix, class_names, metrics

def plot_confusion_matrix_from_csv(csv_path, output_path="confusion_matrix_from_csv.png"):
    """
    Plots a confusion matrix using only the first frame of each video from the mask_metrics.csv file.
    Compares predicted_category_name to gt_category_name.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Get the first frame for each video (assuming sorted by video_id and frame order)
    first_frames = df.groupby('video_id').first().reset_index()
    print(first_frames.head())
    # Extract predicted and ground truth category names
    y_true = first_frames['gt_category_name']
    y_pred = first_frames['predicted_category_name']

    # Get sorted list of all unique categories
    all_categories = sorted(set(y_true) | set(y_pred))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_categories)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_categories)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix (First Frame per Video)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def test_confusion_matrix_thresholds(preds, gts, confidence_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], 
                                   min_track_lengths=[3, 5, 10], iou_thresholds=[0.3, 0.5, 0.7]):
    """
    Test different thresholds for confusion matrix optimization.
    
    Args:
        preds: List of predictions from model
        gts: Ground truth annotations
        confidence_thresholds: List of confidence thresholds to test
        min_track_lengths: List of minimum track lengths to test
        iou_thresholds: List of IoU thresholds to test
    
    Returns:
        results: Dictionary with results for each combination
    """
    results = {}
    
    print("Testing confusion matrix thresholds...")
    print("=" * 80)
    
    for conf_thresh in confidence_thresholds:
        for track_len in min_track_lengths:
            for iou_thresh in iou_thresholds:
                print(f"Testing: Conf≥{conf_thresh}, Track≥{track_len}fr, IoU≥{iou_thresh}")
                
                try:
                    cm, class_names, class_metrics = tracking_confusion_matrix(
                        preds, gts,
                        confidence_threshold=conf_thresh,
                        min_track_length=track_len,
                        iou_threshold=iou_thresh
                    )
                    
                    # Calculate overall metrics
                    total_tp = total_fp = total_fn = 0
                    for class_name in class_names:
                        metrics = class_metrics[class_name]
                        total_tp += metrics['tp']
                        total_fp += metrics['fp']
                        total_fn += metrics['fn']
                    
                    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
                    
                    key = f"conf{conf_thresh}_track{track_len}_iou{iou_thresh}"
                    results[key] = {
                        'conf_threshold': conf_thresh,
                        'min_track_length': track_len,
                        'iou_threshold': iou_thresh,
                        'overall_precision': overall_precision,
                        'overall_recall': overall_recall,
                        'overall_f1': overall_f1,
                        'total_tp': total_tp,
                        'total_fp': total_fp,
                        'total_fn': total_fn,
                        'class_metrics': class_metrics
                    }
                    
                    print(f"  Overall: P={overall_precision:.3f}, R={overall_recall:.3f}, F1={overall_f1:.3f}, TP={total_tp}, FP={total_fp}, FN={total_fn}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
    
    # Find best parameters
    best_f1 = 0
    best_params = None
    
    for key, result in results.items():
        if result['overall_f1'] > best_f1:
            best_f1 = result['overall_f1']
            best_params = key
    
    print("\n" + "=" * 80)
    print("BEST PARAMETERS:")
    if best_params:
        best_result = results[best_params]
        print(f"Confidence threshold: {best_result['conf_threshold']}")
        print(f"Minimum track length: {best_result['min_track_length']}")
        print(f"IoU threshold: {best_result['iou_threshold']}")
        print(f"Overall F1: {best_result['overall_f1']:.3f}")
        print(f"Overall Precision: {best_result['overall_precision']:.3f}")
        print(f"Overall Recall: {best_result['overall_recall']:.3f}")
    else:
        print("No valid results found")
    
    return results

def plot_confusion_matrix(confusion_matrix, class_names, output_path, title="Confusion Matrix", 
                         method="tracking", conf_threshold=0.5, track_length=5, iou_threshold=0.5):
    """
    Plot confusion matrix with proper formatting.
    
    Args:
        confusion_matrix: numpy array of confusion matrix
        class_names: List of class names
        output_path: Path to save the plot
        title: Plot title
        method: Method used ("simple" or "tracking")
        conf_threshold: Confidence threshold used
        track_length: Minimum track length used
        iou_threshold: IoU threshold used
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    
    # Create title with parameters
    if method == "simple":
        title_text = f'{title}\n(Method: Highest Scoring Prediction, Conf≥{conf_threshold})'
    else:
        title_text = f'{title}\n(Method: Full Tracking, Conf≥{conf_threshold}, Track≥{track_length}fr, IoU≥{iou_threshold})'
    
    plt.title(title_text)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def print_class_metrics(metrics, method="tracking"):
    """
    Print per-class metrics in a formatted table.
    
    Args:
        metrics: Dictionary with per-class metrics
        method: Method used for display
    """
    print(f"\nPer-Class Classification Metrics ({method.title()} Method):")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<5} {'FP':<5} {'FN':<5}")
    print("-" * 80)
    
    total_tp = total_fp = total_fn = 0
    for class_name, class_metrics in metrics.items():
        print(f"{class_name:<15} {class_metrics['precision']:<10.3f} {class_metrics['recall']:<10.3f} "
              f"{class_metrics['f1']:<10.3f} {class_metrics['tp']:<5} {class_metrics['fp']:<5} {class_metrics['fn']:<5}")
        total_tp += class_metrics['tp']
        total_fp += class_metrics['fp']
        total_fn += class_metrics['fn']
    
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    print("-" * 80)
    print(f"{'OVERALL':<15} {overall_precision:<10.3f} {overall_recall:<10.3f} {overall_f1:<10.3f} {total_tp:<5} {total_fp:<5} {total_fn:<5}")
    print("-" * 80)
    
    return {
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }

def main(results_json, val_json, output_dir=None, method="tracking", 
         confidence_threshold=0.5, min_track_length=5, iou_threshold=0.5, test_thresholds=False):
    """
    Main function to compute and plot confusion matrices.
    
    Args:
        results_json: Path to results.json file
        val_json: Path to validation JSON file
        output_dir: Directory to save confusion matrix plots (default: same as results.json)
        method: "simple" or "tracking"
        confidence_threshold: Confidence threshold for filtering
        min_track_length: Minimum track length (for tracking method)
        iou_threshold: IoU threshold for matching (for tracking method)
        test_thresholds: Whether to test different threshold combinations
    """
    # Load data
    with open(results_json, 'r') as f:
        preds = json.load(f)
    with open(val_json, 'r') as f:
        gts = json.load(f)
    
    # Set default output directory to same directory as results.json
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(results_json))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if test_thresholds:
        print("Testing different threshold combinations...")
        results = test_confusion_matrix_thresholds(preds, gts)
        return results
    
    if method == "simple":
        print("Computing simple confusion matrix (highest scoring prediction)...")
        cm, class_names, metrics = simple_confusion_matrix(preds, gts, confidence_threshold)
        output_path = os.path.join(output_dir, "confusion_matrix_simple.png")
        plot_confusion_matrix(cm, class_names, output_path, "Simple Confusion Matrix", 
                            method="simple", conf_threshold=confidence_threshold)
    else:
        print("Computing tracking confusion matrix...")
        cm, class_names, metrics = tracking_confusion_matrix(preds, gts, confidence_threshold, 
                                                           min_track_length, iou_threshold)
        output_path = os.path.join(output_dir, "confusion_matrix_tracking.png")
        plot_confusion_matrix(cm, class_names, output_path, "Tracking Confusion Matrix", 
                            method="tracking", conf_threshold=confidence_threshold, 
                            track_length=min_track_length, iou_threshold=iou_threshold)
    
    # Print metrics
    overall_metrics = print_class_metrics(metrics, method)
    
    print(f"\nConfusion matrix saved to: {output_path}")
    
    return {
        'confusion_matrix': cm,
        'class_names': class_names,
        'metrics': metrics,
        'overall_metrics': overall_metrics,
        'output_path': output_path
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute and plot confusion matrices for VIS evaluation')
    parser.add_argument('--results-json', type=str, required=True,
                       help='Path to results.json file')
    parser.add_argument('--val-json', type=str, required=True,
                       help='Path to validation JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save confusion matrix plots (default: same directory as results.json)')
    parser.add_argument('--method', type=str, choices=['simple', 'tracking'], default='tracking',
                       help='Method to use: simple (highest scoring) or tracking (full implementation)')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold for filtering predictions')
    parser.add_argument('--min-track-length', type=int, default=5,
                       help='Minimum track length (for tracking method)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching (for tracking method)')
    parser.add_argument('--test-thresholds', action='store_true',
                       help='Test different threshold combinations')
    
    args = parser.parse_args()
    
    main(results_json=args.results_json,
         val_json=args.val_json,
         output_dir=args.output_dir,
         method=args.method,
         confidence_threshold=args.confidence_threshold,
         min_track_length=args.min_track_length,
         iou_threshold=args.iou_threshold,
         test_thresholds=args.test_thresholds)
