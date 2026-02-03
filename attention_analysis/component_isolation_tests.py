#!/usr/bin/env python3
"""
Component Isolation Tests for AP Performance Drop Analysis

This script performs three diagnostic tests to identify which component
(classification, segmentation, or score calibration) is causing a performance drop:

Test A: Classification-only impact (fix segmentation, test classification)
Test B: Segmentation-only impact (fix classification, test segmentation)
Test C: Score calibration impact (use scores from reference predictions, matched by rank order)
"""

import json
import numpy as np
import sys
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from pycocotools import mask as maskUtils
import numpy as np

# Configuration
# NORMAL_DIR = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/fold6/checkpoint_0004443"
# SCRAMBLED_DIR = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6/eval_4443_edit91_seed75"
NORMAL_DIR = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/fold6/checkpoint_0004443"
SCRAMBLED_DIR = "/home/simone/store/simone/dvis-model-outputs/top_fold_results/silhouette/scrambled_fold6/eval_4443_all_frames_seed31"

# Paths to DVIS-DAQ
DVIS_DAQ_PATH = "/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ"


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def setup_dvis_daq_imports():
    """Setup imports for DVIS-DAQ evaluation."""
    dvis_daq_path = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_Plus/data_video/datasets'
    pycocotools_path = os.path.join(dvis_daq_path, 'pycocotools')
    pycocotools_backup_path = os.path.join(dvis_daq_path, 'pycocotools_backup')
    
    # Check if pycocotools folder exists
    if not os.path.exists(pycocotools_path):
        raise FileNotFoundError(f"pycocotools folder not found at {pycocotools_path}")
    
    folder_renamed = False
    try:
        # Temporarily rename pycocotools to pycocotools_backup
        if os.path.exists(pycocotools_backup_path):
            shutil.rmtree(pycocotools_backup_path)
        shutil.move(pycocotools_path, pycocotools_backup_path)
        folder_renamed = True
        
        # Add the datasets path to sys.path
        if dvis_daq_path not in sys.path:
            sys.path.insert(0, dvis_daq_path)
        
        # Now import from pycocotools_backup
        from pycocotools_backup.oviseval import OVISeval
        from ytvis_api.ytvos import YTVOS
        
        return OVISeval, YTVOS, folder_renamed, pycocotools_path, pycocotools_backup_path
    except Exception as e:
        print(f"Error setting up DVIS-DAQ imports: {e}")
        # Restore folder if renamed
        if folder_renamed and os.path.exists(pycocotools_backup_path):
            try:
                if os.path.exists(pycocotools_path):
                    shutil.rmtree(pycocotools_path)
                shutil.move(pycocotools_backup_path, pycocotools_path)
            except:
                pass
        raise


def compute_temporal_iou(pred_segs, gt_segs):
    """
    Compute temporal IoU between prediction and ground truth sequences.
    
    Args:
        pred_segs: List of RLE masks (or None) for prediction
        gt_segs: List of RLE masks (or None) for ground truth
    
    Returns:
        IoU value (float)
    """
    intersection = 0.0
    union = 0.0
    
    # Ensure same length
    min_len = min(len(pred_segs), len(gt_segs))
    
    for i in range(min_len):
        pred_seg = pred_segs[i] if i < len(pred_segs) else None
        gt_seg = gt_segs[i] if i < len(gt_segs) else None
        
        if pred_seg is not None and gt_seg is not None:
            # Both have masks - compute intersection and union
            inter_mask = maskUtils.merge([pred_seg, gt_seg], True)
            union_mask = maskUtils.merge([pred_seg, gt_seg], False)
            intersection += maskUtils.area(inter_mask)
            union += maskUtils.area(union_mask)
        elif pred_seg is not None:
            # Only prediction has mask
            union += maskUtils.area(pred_seg)
        elif gt_seg is not None:
            # Only GT has mask
            union += maskUtils.area(gt_seg)
    
    return intersection / union if union > 0 else 0.0


def match_predictions_to_gt(predictions, ground_truth):
    """
    Match predictions to ground truth objects.
    
    Returns:
        matches: dict mapping pred_idx -> (gt_idx, iou, category_correct)
    """
    matches = {}
    gt_matched = set()
    
    # Group GT by video_id
    gt_by_video = defaultdict(list)
    for idx, gt in enumerate(ground_truth['annotations']):
        gt_by_video[gt['video_id']].append((idx, gt))
    
    # For each prediction, find best matching GT
    for pred_idx, pred in enumerate(predictions):
        video_id = pred['video_id']
        if video_id not in gt_by_video:
            continue
        
        best_iou = 0.0
        best_gt_idx = None
        best_category_correct = False
        
        for gt_idx, gt in gt_by_video[video_id]:
            if gt_idx in gt_matched:
                continue
            
            # Compute temporal IoU
            iou = compute_temporal_iou(pred['segmentations'], gt['segmentations'])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                best_category_correct = (pred['category_id'] == gt['category_id'])
        
        if best_gt_idx is not None:
            matches[pred_idx] = (best_gt_idx, best_iou, best_category_correct)
            gt_matched.add(best_gt_idx)
    
    return matches


def evaluate_with_oviseval(predictions, ground_truth_json_path, dataset_name="test_dataset"):
    """
    Evaluate predictions using OVISeval.
    
    Args:
        predictions: List of prediction dicts
        ground_truth_json_path: Path to ground truth JSON file
    
    Returns:
        metrics: dict with AP, AP50, AP, etc.
    """
    OVISeval, YTVOS, folder_renamed, pycocotools_path, pycocotools_backup_path = setup_dvis_daq_imports()
    
    try:
        # Load ground truth
        coco_gt = YTVOS(ground_truth_json_path)
        
        # Convert predictions to format expected by OVIS
        coco_results = []
        for pred in predictions:
            segmentations = []
            for seg in pred['segmentations']:
                if seg is not None:
                    segmentations.append(seg)
                else:
                    segmentations.append(None)
            
            coco_results.append({
                'video_id': pred['video_id'],
                'score': float(pred['score']),
                'category_id': int(pred['category_id']),
                'segmentations': segmentations
            })
        
        # Load predictions
        coco_dt = coco_gt.loadRes(coco_results)
        
        # Run evaluation
        coco_eval = OVISeval(coco_gt, coco_dt)
        coco_eval.params.maxDets = [1, 10, 100]
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {}
        if len(coco_eval.stats) >= 18:
            metrics['AP'] = float(coco_eval.stats[0])
            metrics['AP50'] = float(coco_eval.stats[1])
            metrics['AP75'] = float(coco_eval.stats[2])  # Keep for reference, but not used
        # Use AP as the primary metric
        metrics['primary_metric'] = metrics.get('AP', 0.0)
        
        return metrics
    
    finally:
        # Restore pycocotools folder
        if folder_renamed and os.path.exists(pycocotools_backup_path):
            try:
                if os.path.exists(pycocotools_path):
                    shutil.rmtree(pycocotools_path)
                shutil.move(pycocotools_backup_path, pycocotools_path)
            except Exception as e:
                print(f"Warning: Failed to restore pycocotools folder: {e}")


def test_a_classification_impact(predictions, ground_truth):
    """
    Test A: Classification-only impact
    Fix segmentation (use GT masks), test classification.
    """
    print("\n" + "="*80)
    print("TEST A: Classification-Only Impact")
    print("="*80)
    print("Replacing predicted masks with GT masks (perfect segmentation)...")
    
    # Group GT by video_id
    gt_by_video = {}
    for gt in ground_truth['annotations']:
        gt_by_video[gt['video_id']] = gt
    
    # Create modified predictions with GT masks
    modified_predictions = []
    for pred in predictions:
        video_id = pred['video_id']
        if video_id in gt_by_video:
            gt = gt_by_video[video_id]
            # Use GT masks, keep predicted category and score
            modified_predictions.append({
                'video_id': pred['video_id'],
                'score': pred['score'],
                'category_id': pred['category_id'],  # Keep predicted category
                'segmentations': gt['segmentations']  # Use GT masks
            })
        else:
            # No GT for this video, skip
            continue
    
    print(f"Modified {len(modified_predictions)} predictions")
    
    # Need to save ground truth to temp file for evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ground_truth, f)
        temp_gt_path = f.name
    
    try:
        # Evaluate
        metrics = evaluate_with_oviseval(modified_predictions, temp_gt_path)
        return metrics, modified_predictions
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_gt_path)
        except:
            pass


def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename (e.g., "path/to/00200.jpg" -> 200)."""
    try:
        basename = filename.split('/')[-1]
        frame_num_str = basename.replace('.jpg', '').strip()
        return int(frame_num_str)
    except (ValueError, AttributeError):
        raise ValueError(f"Could not extract frame number from {filename}")


def create_scramble_mapping(file_names: list) -> np.ndarray:
    """
    Create mapping from scrambled frame order to normal (unscrambled) frame order.
    
    Args:
        file_names: List of filenames in scrambled order (as they appear in scrambled val.json)
    
    Returns:
        Array mapping: scrambled_index -> unscrambled_index
        Example: mapping[5] = 0 means scrambled position 5 corresponds to unscrambled position 0
    """
    if not file_names:
        return None
    
    # Extract frame numbers for each position in scrambled order
    scrambled_frame_numbers = []
    for scrambled_idx, filename in enumerate(file_names):
        frame_num = extract_frame_number(filename)
        scrambled_frame_numbers.append((scrambled_idx, frame_num))
    
    # Sort by frame number to get correct (unscrambled) order
    scrambled_frame_numbers.sort(key=lambda x: x[1])
    
    # Create reverse mapping: scrambled_index -> unscrambled_index
    # We need to know: for each scrambled position, what is its position in unscrambled order?
    scramble_to_unscramble = {}
    for unscrambled_idx, (scrambled_idx, _) in enumerate(scrambled_frame_numbers):
        scramble_to_unscramble[scrambled_idx] = unscrambled_idx
    
    # Create array: scramble_to_unscramble[scrambled_idx] = unscrambled_idx
    max_scrambled_idx = max(scramble_to_unscramble.keys())
    mapping = np.zeros(max_scrambled_idx + 1, dtype=int)
    for scrambled_idx, unscrambled_idx in scramble_to_unscramble.items():
        mapping[scrambled_idx] = unscrambled_idx
    
    return mapping


def reorder_segmentations(segmentations: list, scramble_mapping: np.ndarray) -> list:
    """
    Reorder segmentations from normal (unscrambled) order to scrambled order.
    
    Args:
        segmentations: Segmentations in normal (unscrambled) order
        scramble_mapping: Mapping from scrambled_index -> unscrambled_index
    
    Returns:
        Segmentations reordered to match scrambled frame order
    """
    if len(segmentations) != len(scramble_mapping):
        raise ValueError(f"Segmentations length ({len(segmentations)}) doesn't match mapping length ({len(scramble_mapping)})")
    
    # Create scrambled segmentations: scrambled_seg[scrambled_idx] = normal_seg[unscrambled_idx]
    # scramble_mapping[scrambled_idx] gives us the unscrambled_idx
    scrambled_segmentations = [None] * len(segmentations)
    for scrambled_idx in range(len(scramble_mapping)):
        unscrambled_idx = scramble_mapping[scrambled_idx]
        if 0 <= unscrambled_idx < len(segmentations):
            scrambled_segmentations[scrambled_idx] = segmentations[unscrambled_idx]
    
    return scrambled_segmentations


def test_a2_segmentation_swap(predictions, ground_truth, reference_predictions, scrambled_val_json_path=None):
    """
    Test A2: Replace segmentations from reference predictions (normal model)
    This tests if using normal model's segmentations improves scrambled predictions.
    
    Args:
        predictions: Predictions to modify (scrambled)
        ground_truth: Ground truth for evaluation (scrambled)
        reference_predictions: Predictions to take segmentations from (normal)
        scrambled_val_json_path: Path to scrambled val.json (needed for frame mapping)
    """
    print("\n" + "="*80)
    print("TEST A2: Segmentation Swap (Use Normal Segmentations)")
    print("="*80)
    print("Replacing predicted segmentations with reference segmentations (matched by score rank per video)...")
    print("Reordering normal segmentations to match scrambled frame order...")
    
    # Load scrambled val.json to get frame mapping
    if scrambled_val_json_path is None:
        # Try to find it from ground_truth path or use a default
        raise ValueError("scrambled_val_json_path is required for frame reordering")
    
    with open(scrambled_val_json_path, 'r') as f:
        scrambled_val_data = json.load(f)
    
    # Create frame mapping for each video: video_id -> scramble_mapping
    video_frame_mappings = {}
    videos = {video['id']: video for video in scrambled_val_data['videos']}
    for video_id, video in videos.items():
        if 'file_names' in video:
            scramble_mapping = create_scramble_mapping(video['file_names'])
            if scramble_mapping is not None:
                video_frame_mappings[video_id] = scramble_mapping
                print(f"  Created frame mapping for video {video_id} ({len(scramble_mapping)} frames)")
    
    # Group predictions by video_id, keeping track of original indices
    preds_by_video = defaultdict(list)
    ref_preds_by_video = defaultdict(list)
    
    for pred_idx, pred in enumerate(predictions):
        preds_by_video[pred['video_id']].append((pred_idx, pred))
    
    for ref_idx, ref_pred in enumerate(reference_predictions):
        ref_preds_by_video[ref_pred['video_id']].append((ref_idx, ref_pred))
    
    # Sort by score within each video (descending)
    for video_id in preds_by_video:
        preds_by_video[video_id].sort(key=lambda x: x[1]['score'], reverse=True)
    
    for video_id in ref_preds_by_video:
        ref_preds_by_video[video_id].sort(key=lambda x: x[1]['score'], reverse=True)
    
    # Create segmentation mapping: for each video, assign reference segmentations to predictions by rank
    segmentation_mapping = {}  # pred_idx -> new_segmentations (reordered to scrambled frame order)
    total_matched = 0
    total_unmatched = 0
    total_reordered = 0
    
    for video_id in preds_by_video:
        video_preds = preds_by_video[video_id]
        video_ref_preds = ref_preds_by_video.get(video_id, [])
        scramble_mapping = video_frame_mappings.get(video_id)
        
        # Match by rank order
        for rank, (pred_idx, pred) in enumerate(video_preds):
            if rank < len(video_ref_preds):
                # Get reference segmentations (in normal/unscrambled order)
                ref_segmentations = video_ref_preds[rank][1]['segmentations']
                
                # Reorder to match scrambled frame order
                if scramble_mapping is not None and isinstance(ref_segmentations, list):
                    try:
                        reordered_segmentations = reorder_segmentations(ref_segmentations, scramble_mapping)
                        segmentation_mapping[pred_idx] = reordered_segmentations
                        total_reordered += 1
                    except Exception as e:
                        print(f"  Warning: Failed to reorder segmentations for video {video_id}, rank {rank}: {e}")
                        segmentation_mapping[pred_idx] = ref_segmentations  # Use as-is
                else:
                    # No mapping available or wrong format, use as-is
                    segmentation_mapping[pred_idx] = ref_segmentations
                
                total_matched += 1
            else:
                # No reference prediction at this rank, keep original segmentations
                segmentation_mapping[pred_idx] = pred['segmentations']
                total_unmatched += 1
    
    print(f"  Matched {total_matched} predictions, {total_unmatched} kept original segmentations")
    print(f"  Reordered {total_reordered} segmentations to match scrambled frame order")
    
    # Create modified predictions with reference segmentations
    modified_predictions = []
    for pred_idx, pred in enumerate(predictions):
        new_segmentations = segmentation_mapping.get(pred_idx, pred['segmentations'])
        modified_predictions.append({
            'video_id': pred['video_id'],
            'score': pred['score'],  # Keep scrambled score
            'category_id': pred['category_id'],  # Keep scrambled category
            'segmentations': new_segmentations  # Use reference segmentations (reordered)
        })
    
    print(f"Modified {len(modified_predictions)} predictions")
    print(f"  Matched segmentations from {len(ref_preds_by_video)} videos")
    
    # Need to save ground truth to temp file for evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ground_truth, f)
        temp_gt_path = f.name
    
    try:
        # Evaluate
        metrics = evaluate_with_oviseval(modified_predictions, temp_gt_path)
        return metrics, modified_predictions
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_gt_path)
        except:
            pass


def test_b_segmentation_impact(predictions, ground_truth, reference_predictions=None):
    """
    Test B: Segmentation-only impact
    Fix classification (use GT categories or reference categories), test segmentation.
    
    Args:
        predictions: Predictions to modify
        ground_truth: Ground truth for evaluation
        reference_predictions: If provided, use categories from these predictions (matched by rank)
                              If None, use GT categories
    """
    print("\n" + "="*80)
    print("TEST B: Segmentation-Only Impact")
    print("="*80)
    
    if reference_predictions is not None:
        print("Replacing predicted categories with reference categories (matched by score rank per video)...")
        
        # Group predictions by video_id
        preds_by_video = defaultdict(list)
        ref_preds_by_video = defaultdict(list)
        
        for pred_idx, pred in enumerate(predictions):
            preds_by_video[pred['video_id']].append((pred_idx, pred))
        
        for ref_idx, ref_pred in enumerate(reference_predictions):
            ref_preds_by_video[ref_pred['video_id']].append((ref_idx, ref_pred))
        
        # Sort by score within each video (descending)
        for video_id in preds_by_video:
            preds_by_video[video_id].sort(key=lambda x: x[1]['score'], reverse=True)
        
        for video_id in ref_preds_by_video:
            ref_preds_by_video[video_id].sort(key=lambda x: x[1]['score'], reverse=True)
        
        # Create category mapping: for each video, assign reference categories to predictions by rank
        category_mapping = {}  # pred_idx -> new_category_id
        total_matched = 0
        total_unmatched = 0
        for video_id in preds_by_video:
            video_preds = preds_by_video[video_id]
            video_ref_preds = ref_preds_by_video.get(video_id, [])
            
            # Match by rank order
            for rank, (pred_idx, pred) in enumerate(video_preds):
                if rank < len(video_ref_preds):
                    # Assign category from reference prediction at same rank
                    ref_category = video_ref_preds[rank][1]['category_id']
                    category_mapping[pred_idx] = ref_category
                    total_matched += 1
                else:
                    # No reference prediction at this rank, keep original category
                    category_mapping[pred_idx] = pred['category_id']
                    total_unmatched += 1
        
        print(f"  Matched {total_matched} predictions, {total_unmatched} kept original categories")
        
        # Create modified predictions with reference categories
        modified_predictions = []
        for pred_idx, pred in enumerate(predictions):
            new_category = category_mapping.get(pred_idx, pred['category_id'])
            modified_predictions.append({
                'video_id': pred['video_id'],
                'score': pred['score'],
                'category_id': new_category,  # Use reference category
                'segmentations': pred['segmentations']  # Keep predicted masks
            })
        
        print(f"Modified {len(modified_predictions)} predictions")
        print(f"  Matched categories from {len(ref_preds_by_video)} videos")
    
    else:
        print("Replacing predicted categories with GT categories (perfect classification)...")
        
        # Group GT by video_id
        gt_by_video = {}
        for gt in ground_truth['annotations']:
            gt_by_video[gt['video_id']] = gt
        
        # Create modified predictions with GT categories
        modified_predictions = []
        for pred in predictions:
            video_id = pred['video_id']
            if video_id in gt_by_video:
                gt = gt_by_video[video_id]
                # Use GT category, keep predicted masks and score
                modified_predictions.append({
                    'video_id': pred['video_id'],
                    'score': pred['score'],
                    'category_id': gt['category_id'],  # Use GT category
                    'segmentations': pred['segmentations']  # Keep predicted masks
                })
            else:
                # No GT for this video, skip
                continue
        
        print(f"Modified {len(modified_predictions)} predictions")
    
    # Need to save ground truth to temp file for evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ground_truth, f)
        temp_gt_path = f.name
    
    try:
        # Evaluate
        metrics = evaluate_with_oviseval(modified_predictions, temp_gt_path)
        return metrics, modified_predictions
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_gt_path)
        except:
            pass


def test_c_score_calibration_impact(predictions, ground_truth, reference_predictions=None):
    """
    Test C: Score calibration impact
    Replace scores with scores from reference predictions, matched by rank order within each video.
    
    Args:
        predictions: Predictions to modify (will get new scores)
        ground_truth: Ground truth for evaluation
        reference_predictions: Predictions to take scores from (if None, uses quality-based scores)
    """
    print("\n" + "="*80)
    print("TEST C: Score Calibration Impact")
    print("="*80)
    
    if reference_predictions is not None:
        print("Replacing scores with scores from reference predictions (matched by rank order per video)...")
        
        # Group predictions by video_id
        preds_by_video = defaultdict(list)
        ref_preds_by_video = defaultdict(list)
        
        for pred_idx, pred in enumerate(predictions):
            preds_by_video[pred['video_id']].append((pred_idx, pred))
        
        for ref_idx, ref_pred in enumerate(reference_predictions):
            ref_preds_by_video[ref_pred['video_id']].append((ref_idx, ref_pred))
        
        # Sort by score within each video (descending)
        for video_id in preds_by_video:
            preds_by_video[video_id].sort(key=lambda x: x[1]['score'], reverse=True)
        
        for video_id in ref_preds_by_video:
            ref_preds_by_video[video_id].sort(key=lambda x: x[1]['score'], reverse=True)
        
        # Create score mapping: for each video, assign reference scores to predictions by rank
        score_mapping = {}  # pred_idx -> new_score
        total_matched = 0
        total_unmatched = 0
        for video_id in preds_by_video:
            video_preds = preds_by_video[video_id]
            video_ref_preds = ref_preds_by_video.get(video_id, [])
            
            # Match by rank order
            for rank, (pred_idx, pred) in enumerate(video_preds):
                if rank < len(video_ref_preds):
                    # Assign score from reference prediction at same rank
                    ref_score = video_ref_preds[rank][1]['score']
                    score_mapping[pred_idx] = ref_score
                    total_matched += 1
                else:
                    # No reference prediction at this rank, keep original score
                    score_mapping[pred_idx] = pred['score']
                    total_unmatched += 1
        
        print(f"  Matched {total_matched} scores, {total_unmatched} kept original scores")
        
        # Create modified predictions with new scores
        modified_predictions = []
        for pred_idx, pred in enumerate(predictions):
            new_score = score_mapping.get(pred_idx, pred['score'])
            modified_predictions.append({
                'video_id': pred['video_id'],
                'score': new_score,
                'category_id': pred['category_id'],
                'segmentations': pred['segmentations']
            })
        
        print(f"Modified {len(modified_predictions)} predictions")
        print(f"  Score range: [{min([p['score'] for p in modified_predictions]):.4f}, "
              f"{max([p['score'] for p in modified_predictions]):.4f}]")
        print(f"  Matched scores from {len(ref_preds_by_video)} videos")
    
    else:
        # Fallback to quality-based scores if no reference provided
        print("Replacing scores with quality-based scores (IoU × category_correctness)...")
        
        # Match predictions to GT
        print("  Matching predictions to ground truth...")
        matches = match_predictions_to_gt(predictions, ground_truth)
        print(f"  Matched {len(matches)} predictions")
        
        # Group GT by video_id for category lookup
        gt_by_video = {}
        for gt in ground_truth['annotations']:
            gt_by_video[gt['video_id']] = gt
        
        # Create modified predictions with quality-based scores
        modified_predictions = []
        for pred_idx, pred in enumerate(predictions):
            video_id = pred['video_id']
            
            if pred_idx in matches:
                # Matched to GT - compute quality score
                gt_idx, iou, category_correct = matches[pred_idx]
                quality_score = iou * (1.0 if category_correct else 0.0)
            else:
                # Not matched - use IoU = 0
                if video_id in gt_by_video:
                    gt = gt_by_video[video_id]
                    # Try to compute IoU even if not matched
                    iou = compute_temporal_iou(pred['segmentations'], gt['segmentations'])
                    category_correct = (pred['category_id'] == gt['category_id'])
                    quality_score = iou * (1.0 if category_correct else 0.0)
                else:
                    quality_score = 0.0
            
            modified_predictions.append({
                'video_id': pred['video_id'],
                'score': quality_score,  # Use quality-based score
                'category_id': pred['category_id'],
                'segmentations': pred['segmentations']
            })
        
        print(f"Modified {len(modified_predictions)} predictions")
        print(f"  Score range: [{min([p['score'] for p in modified_predictions]):.4f}, "
              f"{max([p['score'] for p in modified_predictions]):.4f}]")
    
    # Need to save ground truth to temp file for evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ground_truth, f)
        temp_gt_path = f.name
    
    try:
        # Evaluate
        metrics = evaluate_with_oviseval(modified_predictions, temp_gt_path)
        return metrics, modified_predictions
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_gt_path)
        except:
            pass


def test_d_combined_fix(predictions, ground_truth, reference_predictions):
    """
    Test D: Combined fix - use both categories and scores from reference predictions
    This tests if fixing both classification and score calibration can recover performance.
    
    Args:
        predictions: Predictions to modify (scrambled)
        ground_truth: Ground truth for evaluation
        reference_predictions: Predictions to take categories and scores from (normal)
    """
    print("\n" + "="*80)
    print("TEST D: Combined Fix (Categories + Scores from Reference)")
    print("="*80)
    print("Replacing predicted categories and scores with reference values (matched by rank order per video)...")
    
    # Group predictions by video_id
    preds_by_video = defaultdict(list)
    ref_preds_by_video = defaultdict(list)
    
    for pred_idx, pred in enumerate(predictions):
        preds_by_video[pred['video_id']].append((pred_idx, pred))
    
    for ref_idx, ref_pred in enumerate(reference_predictions):
        ref_preds_by_video[ref_pred['video_id']].append((ref_idx, ref_pred))
    
    # Sort by score within each video (descending)
    for video_id in preds_by_video:
        preds_by_video[video_id].sort(key=lambda x: x[1]['score'], reverse=True)
    
    for video_id in ref_preds_by_video:
        ref_preds_by_video[video_id].sort(key=lambda x: x[1]['score'], reverse=True)
    
    # Create mappings: for each video, assign reference categories and scores to predictions by rank
    category_mapping = {}  # pred_idx -> new_category_id
    score_mapping = {}     # pred_idx -> new_score
    total_matched = 0
    total_unmatched = 0
    
    for video_id in preds_by_video:
        video_preds = preds_by_video[video_id]
        video_ref_preds = ref_preds_by_video.get(video_id, [])
        
        # Match by rank order
        for rank, (pred_idx, pred) in enumerate(video_preds):
            if rank < len(video_ref_preds):
                # Assign category and score from reference prediction at same rank
                ref_category = video_ref_preds[rank][1]['category_id']
                ref_score = video_ref_preds[rank][1]['score']
                category_mapping[pred_idx] = ref_category
                score_mapping[pred_idx] = ref_score
                total_matched += 1
            else:
                # No reference prediction at this rank, keep original
                category_mapping[pred_idx] = pred['category_id']
                score_mapping[pred_idx] = pred['score']
                total_unmatched += 1
    
    print(f"  Matched {total_matched} predictions, {total_unmatched} kept original")
    
    # Create modified predictions with reference categories and scores
    modified_predictions = []
    for pred_idx, pred in enumerate(predictions):
        new_category = category_mapping.get(pred_idx, pred['category_id'])
        new_score = score_mapping.get(pred_idx, pred['score'])
        modified_predictions.append({
            'video_id': pred['video_id'],
            'score': new_score,  # Use reference score
            'category_id': new_category,  # Use reference category
            'segmentations': pred['segmentations']  # Keep predicted masks
        })
    
    print(f"Modified {len(modified_predictions)} predictions")
    print(f"  Score range: [{min([p['score'] for p in modified_predictions]):.4f}, "
          f"{max([p['score'] for p in modified_predictions]):.4f}]")
    print(f"  Matched categories and scores from {len(ref_preds_by_video)} videos")
    
    # Need to save ground truth to temp file for evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ground_truth, f)
        temp_gt_path = f.name
    
    try:
        # Evaluate
        metrics = evaluate_with_oviseval(modified_predictions, temp_gt_path)
        return metrics, modified_predictions
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_gt_path)
        except:
            pass


def plot_ranked_scores_per_video(normal_predictions, scrambled_predictions, output_dir):
    """
    Plot ranked scores for each video, comparing normal and scrambled models.
    Creates a separate PNG file for each video.
    """
    print("\n" + "="*80)
    print("PLOTTING RANKED SCORES PER VIDEO")
    print("="*80)
    
    # Group predictions by video_id
    normal_by_video = defaultdict(list)
    scrambled_by_video = defaultdict(list)
    
    for pred in normal_predictions:
        normal_by_video[pred['video_id']].append(pred)
    
    for pred in scrambled_predictions:
        scrambled_by_video[pred['video_id']].append(pred)
    
    # Get all video IDs
    all_video_ids = sorted(set(list(normal_by_video.keys()) + list(scrambled_by_video.keys())))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating plots for {len(all_video_ids)} videos...")
    
    for video_id in all_video_ids:
        normal_video_preds = normal_by_video.get(video_id, [])
        scrambled_video_preds = scrambled_by_video.get(video_id, [])
        
        if not normal_video_preds and not scrambled_video_preds:
            continue
        
        # Sort by score (descending) for both
        normal_scores = sorted([p['score'] for p in normal_video_preds], reverse=True)
        scrambled_scores = sorted([p['score'] for p in scrambled_video_preds], reverse=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot ranked scores
        if normal_scores:
            normal_ranks = np.arange(1, len(normal_scores) + 1)
            ax.plot(normal_ranks, normal_scores, 'o-', label='Normal', color='steelblue', 
                   linewidth=2, markersize=6, alpha=0.7)
        
        if scrambled_scores:
            scrambled_ranks = np.arange(1, len(scrambled_scores) + 1)
            ax.plot(scrambled_ranks, scrambled_scores, 's-', label='Scrambled', color='coral', 
                   linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Rank (by score, descending)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Ranked Scores Comparison - Video ID: {video_id}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0.5)
        
        # Add statistics text
        stats_text = []
        if normal_scores:
            stats_text.append(f'Normal: n={len(normal_scores)}, mean={np.mean(normal_scores):.3f}, max={max(normal_scores):.3f}')
        if scrambled_scores:
            stats_text.append(f'Scrambled: n={len(scrambled_scores)}, mean={np.mean(scrambled_scores):.3f}, max={max(scrambled_scores):.3f}')
        
        if stats_text:
            ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f'ranked_scores_video_{video_id}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (list(all_video_ids).index(video_id) + 1) % 10 == 0:
            print(f"  Created plots for {list(all_video_ids).index(video_id) + 1}/{len(all_video_ids)} videos...")
    
    print(f"✓ Created {len(all_video_ids)} score comparison plots in {output_dir}")
    return output_dir


def find_val_json(directory):
    """
    Find validation JSON file in the given directory.
    Looks for common patterns: val.json, val_*.json, *_val.json
    """
    dir_path = Path(directory)
    
    # Common patterns to try
    patterns = [
        "val.json",
        "val_*.json",
        "*_val.json",
        "val_fold*.json",
        "ground_truth.json",
        "annotations.json"
    ]
    
    # First try exact matches
    for pattern in patterns:
        matches = list(dir_path.glob(pattern))
        if matches:
            return str(matches[0])
    
    # If no exact match, try case-insensitive search
    for json_file in dir_path.glob("*.json"):
        filename_lower = json_file.name.lower()
        if "val" in filename_lower or "ground" in filename_lower or "gt" in filename_lower:
            return str(json_file)
    
    raise FileNotFoundError(f"Could not find validation JSON file in {directory}")


def main():
    """Run component isolation tests."""
    print("="*80)
    print("COMPONENT ISOLATION TESTS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    normal_preds_path = os.path.join(NORMAL_DIR, "inference/results.json")
    normal_gt_path = find_val_json(NORMAL_DIR)
    scrambled_preds_path = os.path.join(SCRAMBLED_DIR, "inference/results.json")
    scrambled_gt_path = find_val_json(SCRAMBLED_DIR)
    
    print(f"  Normal predictions: {normal_preds_path}")
    print(f"  Normal ground truth: {normal_gt_path} (found)")
    print(f"  Scrambled predictions: {scrambled_preds_path}")
    print(f"  Scrambled ground truth: {scrambled_gt_path} (found)")
    
    normal_predictions = load_json(normal_preds_path)
    normal_ground_truth = load_json(normal_gt_path)
    scrambled_predictions = load_json(scrambled_preds_path)
    scrambled_ground_truth = load_json(scrambled_gt_path)
    
    print(f"  Normal: {len(normal_predictions)} predictions, {len(normal_ground_truth['annotations'])} GT")
    print(f"  Scrambled: {len(scrambled_predictions)} predictions, {len(scrambled_ground_truth['annotations'])} GT")
    
    # Plot ranked scores per video
    plots_output_dir = os.path.join(SCRAMBLED_DIR, "score_rankings_per_video")
    plot_ranked_scores_per_video(normal_predictions, scrambled_predictions, plots_output_dir)
    
    # Baseline evaluations
    print("\n" + "="*80)
    print("BASELINE EVALUATIONS")
    print("="*80)
    
    print("\nEvaluating normal predictions (baseline)...")
    normal_baseline = evaluate_with_oviseval(normal_predictions, normal_gt_path)
    print(f"  Normal AP: {normal_baseline.get('AP', 0):.4f}")
    
    print("\nEvaluating scrambled predictions (baseline)...")
    scrambled_baseline = evaluate_with_oviseval(scrambled_predictions, scrambled_gt_path)
    print(f"  Scrambled AP: {scrambled_baseline.get('AP', 0):.4f}")
    
    ap_drop = normal_baseline.get('AP', 0) - scrambled_baseline.get('AP', 0)
    print(f"\n  AP Drop: {ap_drop:.4f} ({ap_drop/normal_baseline.get('AP', 1)*100:.2f}% relative)")
    
    # Run tests on both normal and scrambled predictions
    print("\n" + "="*80)
    print("RUNNING TESTS ON NORMAL PREDICTIONS")
    print("="*80)
    
    # Test A on Normal: Classification impact
    print("\nTest A on Normal (fix segmentation, test classification)...")
    normal_test_a_metrics, _ = test_a_classification_impact(normal_predictions, normal_ground_truth)
    normal_test_a_improvement = normal_test_a_metrics.get('AP', 0) - normal_baseline.get('AP', 0)
    print(f"  Normal Test A AP: {normal_test_a_metrics.get('AP', 0):.4f} (improvement: {normal_test_a_improvement:+.4f})")
    
    # Test B on Normal: Segmentation impact (use GT categories)
    print("\nTest B on Normal (fix classification with GT, test segmentation)...")
    normal_test_b_metrics, _ = test_b_segmentation_impact(normal_predictions, normal_ground_truth, reference_predictions=None)
    normal_test_b_improvement = normal_test_b_metrics.get('AP', 0) - normal_baseline.get('AP', 0)
    print(f"  Normal Test B AP: {normal_test_b_metrics.get('AP', 0):.4f} (improvement: {normal_test_b_improvement:+.4f})")
    
    # Test C on Normal: Skipped - using baseline performance instead
    
    print("\n" + "="*80)
    print("RUNNING TESTS ON SCRAMBLED PREDICTIONS")
    print("="*80)
    
    # Test A on Scrambled: Classification impact
    print("\nTest A on Scrambled (fix segmentation, test classification)...")
    scrambled_test_a_metrics, _ = test_a_classification_impact(scrambled_predictions, scrambled_ground_truth)
    scrambled_test_a_improvement = scrambled_test_a_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    print(f"  Scrambled Test A AP: {scrambled_test_a_metrics.get('AP', 0):.4f} (improvement: {scrambled_test_a_improvement:+.4f})")
    
    # Test A2 on Scrambled: Segmentation swap (use normal segmentations)
    print("\nTest A2 on Scrambled (replace segmentations with normal segmentations, matched by rank)...")
    scrambled_test_a2_metrics, _ = test_a2_segmentation_swap(
        scrambled_predictions, scrambled_ground_truth, 
        reference_predictions=normal_predictions,
        scrambled_val_json_path=scrambled_gt_path
    )
    scrambled_test_a2_improvement = scrambled_test_a2_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    print(f"  Scrambled Test A2 AP: {scrambled_test_a2_metrics.get('AP', 0):.4f} (improvement: {scrambled_test_a2_improvement:+.4f})")
    
    # Test B on Scrambled: Segmentation impact (use GT categories)
    print("\nTest B on Scrambled (fix classification with GT, test segmentation)...")
    scrambled_test_b_metrics, _ = test_b_segmentation_impact(scrambled_predictions, scrambled_ground_truth, reference_predictions=None)
    scrambled_test_b_improvement = scrambled_test_b_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    print(f"  Scrambled Test B AP: {scrambled_test_b_metrics.get('AP', 0):.4f} (improvement: {scrambled_test_b_improvement:+.4f})")
    
    # Test B2 on Scrambled: Segmentation impact (use normal categories, matched by rank)
    print("\nTest B2 on Scrambled (fix classification with normal categories, matched by rank, test segmentation)...")
    scrambled_test_b2_metrics, _ = test_b_segmentation_impact(scrambled_predictions, scrambled_ground_truth, reference_predictions=normal_predictions)
    scrambled_test_b2_improvement = scrambled_test_b2_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    print(f"  Scrambled Test B2 AP: {scrambled_test_b2_metrics.get('AP', 0):.4f} (improvement: {scrambled_test_b2_improvement:+.4f})")
    
    # Test C on Scrambled: Score calibration impact (use normal scores)
    print("\nTest C on Scrambled (replace scores with normal scores, matched by rank)...")
    scrambled_test_c_metrics, _ = test_c_score_calibration_impact(
        scrambled_predictions, scrambled_ground_truth, reference_predictions=normal_predictions
    )
    scrambled_test_c_improvement = scrambled_test_c_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    print(f"  Scrambled Test C AP: {scrambled_test_c_metrics.get('AP', 0):.4f} (improvement: {scrambled_test_c_improvement:+.4f})")
    
    # Test D on Scrambled: Combined fix (use normal categories and scores)
    print("\nTest D on Scrambled (replace categories and scores with normal, matched by rank)...")
    scrambled_test_d_metrics, _ = test_d_combined_fix(
        scrambled_predictions, scrambled_ground_truth, reference_predictions=normal_predictions
    )
    scrambled_test_d_improvement = scrambled_test_d_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    scrambled_test_d_vs_normal = scrambled_test_d_metrics.get('AP', 0) - normal_baseline.get('AP', 0)
    print(f"  Scrambled Test D AP: {scrambled_test_d_metrics.get('AP', 0):.4f} (improvement: {scrambled_test_d_improvement:+.4f})")
    print(f"  Test D vs Normal baseline: {scrambled_test_d_vs_normal:+.4f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBaseline AP:")
    print(f"  Normal:   {normal_baseline.get('AP', 0):.4f}")
    print(f"  Scrambled: {scrambled_baseline.get('AP', 0):.4f}")
    print(f"  Drop:     {ap_drop:.4f} ({ap_drop/normal_baseline.get('AP', 1)*100:.2f}%)")
    
    # Calculate differences for comparisons
    test_a_diff = scrambled_test_a_metrics.get('AP', 0) - normal_test_a_metrics.get('AP', 0)
    test_a2_diff = scrambled_test_a2_metrics.get('AP', 0) - normal_baseline.get('AP', 0)  # Compare to normal baseline
    test_b_diff = scrambled_test_b_metrics.get('AP', 0) - normal_test_b_metrics.get('AP', 0)
    test_b2_diff = scrambled_test_b2_metrics.get('AP', 0) - normal_baseline.get('AP', 0)  # Compare to normal baseline
    test_c_diff = scrambled_test_c_metrics.get('AP', 0) - normal_baseline.get('AP', 0)  # Compare to normal baseline
    test_d_diff = scrambled_test_d_metrics.get('AP', 0) - normal_baseline.get('AP', 0)
    
    print(f"\nTest Results Comparison:")
    print(f"{'Test':<30} {'Normal':<15} {'Scrambled':<15} {'Difference':<15}")
    print("-" * 75)
    
    # Test A comparison
    print(f"{'Test A (fix segmentation)':<30} {normal_test_a_metrics.get('AP', 0):.4f}        "
          f"{scrambled_test_a_metrics.get('AP', 0):.4f}        {test_a_diff:+.4f}")
    
    # Test A2 comparison (normal baseline vs scrambled with normal segmentations)
    print(f"{'Test A2 (scrambled + normal segs)':<30} {normal_baseline.get('AP', 0):.4f}        "
          f"{scrambled_test_a2_metrics.get('AP', 0):.4f}        {test_a2_diff:+.4f}")
    
    # Test B comparison (with GT categories)
    print(f"{'Test B (fix classification, GT)':<30} {normal_test_b_metrics.get('AP', 0):.4f}        "
          f"{scrambled_test_b_metrics.get('AP', 0):.4f}        {test_b_diff:+.4f}")
    
    # Test B2 comparison (scrambled with normal categories)
    test_b2_diff_vs_baseline = scrambled_test_b2_metrics.get('AP', 0) - normal_baseline.get('AP', 0)
    print(f"{'Test B2 (scrambled + normal cats)':<30} {normal_baseline.get('AP', 0):.4f}        "
          f"{scrambled_test_b2_metrics.get('AP', 0):.4f}        {test_b2_diff_vs_baseline:+.4f}")
    
    # Test C comparison (scrambled only - uses normal scores)
    print(f"{'Test C (fix scores)':<30} {normal_baseline.get('AP', 0):.4f}        "
          f"{scrambled_test_c_metrics.get('AP', 0):.4f}        {test_c_diff:+.4f}")
    print(f"  Note: Normal shows baseline, Scrambled uses normal scores (matched by rank)")
    
    # Test D comparison (vs normal baseline)
    print(f"{'Test D (normal cats + scores)':<30} {normal_baseline.get('AP', 0):.4f}        "
          f"{scrambled_test_d_metrics.get('AP', 0):.4f}        {test_d_diff:+.4f}")
    
    print(f"\nImprovement over Baseline:")
    print(f"{'Test':<30} {'Normal':<15} {'Scrambled':<15} {'Difference':<15}")
    print("-" * 75)
    print(f"{'Test A improvement':<30} {normal_test_a_improvement:+.4f}        "
          f"{scrambled_test_a_improvement:+.4f}        "
          f"{scrambled_test_a_improvement - normal_test_a_improvement:+.4f}")
    print(f"{'Test A2 improvement (normal segs)':<30} {'N/A':<15}        "
          f"{scrambled_test_a2_improvement:+.4f}        "
          f"{'N/A':<15}")
    print(f"{'Test B improvement (GT cats)':<30} {normal_test_b_improvement:+.4f}        "
          f"{scrambled_test_b_improvement:+.4f}        "
          f"{scrambled_test_b_improvement - normal_test_b_improvement:+.4f}")
    print(f"{'Test B2 improvement (normal cats)':<30} {'N/A':<15}        "
          f"{scrambled_test_b2_improvement:+.4f}        "
          f"{'N/A':<15}")
    print(f"{'Test C improvement':<30} {'N/A':<15}        "
          f"{scrambled_test_c_improvement:+.4f}        "
          f"{'N/A':<15}")
    print(f"{'Test D improvement':<30} {'N/A':<15}        "
          f"{scrambled_test_d_improvement:+.4f}        "
          f"{'N/A':<15}")
    
    print(f"\nInterpretation:")
    # Test A fixes segmentation → tests classification impact
    # Test A2 swaps segmentations → tests if normal segmentations help scrambled
    # Test B fixes classification → tests segmentation impact  
    # Test C fixes scores → tests score calibration impact
    
    # Test A2 interpretation (compared to normal baseline)
    if abs(test_a2_diff) < 0.01:
        print("  → Test A2: Scrambled with normal segmentations matches normal baseline")
        print("    → Segmentation quality is similar between normal and scrambled")
    else:
        print(f"  → Test A2: Difference of {test_a2_diff:+.4f} vs normal baseline when using normal segmentations")
        if test_a2_diff > 0:
            print("    → Scrambled with normal segmentations performs BETTER than normal baseline (unexpected!)")
        else:
            print("    → Scrambled with normal segmentations performs WORSE than normal baseline")
            print(f"    → Remaining gap of {abs(test_a2_diff):.4f} suggests other factors (classification/scores) still matter")
    
    # Compare differences between normal and scrambled
    if abs(test_a_diff) < 0.01:
        print("  → Test A: No difference between normal/scrambled when segmentation is fixed")
        print("    → Classification quality is similar between normal and scrambled")
    else:
        print(f"  → Test A: Difference of {test_a_diff:+.4f} between normal/scrambled")
        print("    → Classification quality differs between normal and scrambled")
    
    if abs(test_b_diff) < 0.01:
        print("  → Test B (GT categories): No difference between normal/scrambled")
        print("    → Segmentation quality is similar between normal and scrambled")
    else:
        print(f"  → Test B (GT categories): Difference of {test_b_diff:+.4f} between normal/scrambled")
        print("    → Segmentation quality differs between normal and scrambled")
        if test_b_diff < 0:
            print("    → Scrambled has WORSE segmentation than normal")
        else:
            print("    → Scrambled has BETTER segmentation than normal (unexpected!)")
    
    # Test B2 interpretation (compared to normal baseline)
    if abs(test_b2_diff) < 0.01:
        print("  → Test B2 (normal categories): Scrambled matches normal baseline when using normal categories")
        print("    → Classification differences ARE causing the performance drop")
    else:
        print(f"  → Test B2 (normal categories): Difference of {test_b2_diff:+.4f} vs normal baseline")
        if test_b2_diff < 0:
            print("    → Scrambled still WORSE even with normal categories")
            print("    → Classification differences are NOT the main issue (segmentation is)")
        else:
            print("    → Scrambled performs BETTER with normal categories")
            print("    → Classification differences ARE contributing to the drop")
    
    # Test C interpretation (scrambled only)
    print(f"  → Test C: Scrambled with normal scores: {scrambled_test_c_metrics.get('AP', 0):.4f} vs baseline {scrambled_baseline.get('AP', 0):.4f}")
    scrambled_test_c_improvement = scrambled_test_c_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    if abs(scrambled_test_c_improvement) < 0.01:
        print("    → Normal scores don't help scrambled predictions")
    else:
        print(f"    → Normal scores improve scrambled by {scrambled_test_c_improvement:+.4f}")
    
    # Test D interpretation
    if abs(test_d_diff) < 0.01:
        print("  → Test D (normal categories + scores): Scrambled matches normal baseline")
        print("    → Classification and score calibration differences account for the entire drop")
        print("    → Segmentation quality is similar between normal and scrambled")
    else:
        print(f"  → Test D (normal categories + scores): Difference of {test_d_diff:+.4f} vs normal baseline")
        if test_d_diff < 0:
            print("    → Scrambled still WORSE even with normal categories and scores")
            print("    → Segmentation differences ARE the main issue")
            print(f"    → Remaining gap: {abs(test_d_diff):.4f} (out of {ap_drop:.4f} total drop)")
            print(f"    → Segmentation accounts for {abs(test_d_diff)/ap_drop*100:.1f}% of the drop")
        else:
            print("    → Scrambled performs BETTER than normal baseline (unexpected!)")
            print("    → This suggests some interaction between components")
    
    # Compare Test B2 vs Test D to see impact of adding scores
    test_d_vs_b2 = scrambled_test_d_metrics.get('AP', 0) - scrambled_test_b2_metrics.get('AP', 0)
    test_c_vs_baseline = scrambled_test_c_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    print(f"\n  Score Calibration Impact Analysis:")
    print(f"    Test B2 (normal categories only): {scrambled_test_b2_metrics.get('AP', 0):.4f}")
    print(f"    Test C (normal scores only): {scrambled_test_c_metrics.get('AP', 0):.4f} (improvement: {test_c_vs_baseline:+.4f})")
    print(f"    Test D (normal categories + scores): {scrambled_test_d_metrics.get('AP', 0):.4f}")
    print(f"    Adding scores to categories (B2→D): {test_d_vs_b2:+.4f} improvement")
    if test_d_vs_b2 < test_c_vs_baseline:
        print(f"    → Scores help more when categories are wrong (Test C) than when categories are correct (B2→D)")
        print(f"    → This suggests score calibration interacts with classification errors")
    else:
        print(f"    → Scores help similarly regardless of category correctness")
    
    # Identify primary issue
    print(f"\nPrimary Issue Identification:")
    max_diff = max(abs(test_a_diff), abs(test_b_diff), abs(test_b2_diff), abs(test_c_diff), abs(test_d_diff))
    if abs(test_d_diff) == max_diff:
        print("  → Segmentation is the PRIMARY issue (Test D shows largest remaining gap)")
        print(f"    → Even with normal categories and scores, gap is {abs(test_d_diff):.4f}")
    elif abs(test_b_diff) == max_diff:
        print("  → Segmentation is the PRIMARY issue (largest difference in Test B with GT categories)")
    elif abs(test_b2_diff) == max_diff:
        print("  → Classification is the PRIMARY issue (largest difference in Test B2 with normal categories)")
    elif abs(test_a_diff) == max_diff:
        print("  → Classification is the PRIMARY issue (largest difference in Test A)")
    elif abs(test_c_diff) == max_diff:
        print("  → Score calibration is the PRIMARY issue (largest difference in Test C)")
    else:
        print("  → Multiple components contribute to the difference")
    
    # Summary of what Test D tells us
    print(f"\nTest D Summary (Combined Fix):")
    print(f"  Normal baseline: {normal_baseline.get('AP', 0):.4f}")
    print(f"  Scrambled baseline: {scrambled_baseline.get('AP', 0):.4f}")
    print(f"  Test D (scrambled + normal cats + scores): {scrambled_test_d_metrics.get('AP', 0):.4f}")
    print(f"  Recovery: {scrambled_test_d_improvement:.4f} out of {ap_drop:.4f} total drop")
    print(f"  Recovery percentage: {scrambled_test_d_improvement/ap_drop*100:.1f}%")
    if abs(test_d_diff) < 0.01:
        print("  → FULL recovery: Classification and score calibration account for 100% of the drop")
    else:
        print(f"  → PARTIAL recovery: {abs(test_d_diff):.4f} gap remains (segmentation issue)")
    
    # Detailed breakdown of score impact
    print(f"\nScore Calibration Impact Breakdown:")
    test_c_improvement = scrambled_test_c_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    test_b2_improvement = scrambled_test_b2_metrics.get('AP', 0) - scrambled_baseline.get('AP', 0)
    test_d_vs_b2 = scrambled_test_d_metrics.get('AP', 0) - scrambled_test_b2_metrics.get('AP', 0)
    
    print(f"  Test C (scores only, wrong categories): {scrambled_test_c_metrics.get('AP', 0):.4f} (improvement: {test_c_improvement:+.4f})")
    print(f"  Test B2 (categories only, wrong scores): {scrambled_test_b2_metrics.get('AP', 0):.4f} (improvement: {test_b2_improvement:+.4f})")
    print(f"  Test D (categories + scores): {scrambled_test_d_metrics.get('AP', 0):.4f}")
    print(f"  Adding scores to categories (B2→D): {test_d_vs_b2:+.4f} additional improvement")
    print(f"\n  Interpretation:")
    print(f"    → Scores help MORE when categories are wrong (Test C: +{test_c_improvement:.4f})")
    print(f"      than when categories are correct (B2→D: +{test_d_vs_b2:.4f})")
    print(f"    → This suggests score calibration partially compensates for classification errors")
    print(f"    → When categories are already correct, better scores have diminishing returns")
    
    # Save results
    results = {
        'baseline': {
            'normal_ap': normal_baseline.get('AP', 0),
            'scrambled_ap': scrambled_baseline.get('AP', 0),
            'ap_drop': ap_drop
        },
        'normal_tests': {
            'test_a': {
                'ap': normal_test_a_metrics.get('AP', 0),
                'improvement_over_baseline': normal_test_a_improvement
            },
            'test_b': {
                'ap': normal_test_b_metrics.get('AP', 0),
                'improvement_over_baseline': normal_test_b_improvement
            }
            # Test C skipped for normal - using baseline instead
        },
        'scrambled_tests': {
            'test_a': {
                'ap': scrambled_test_a_metrics.get('AP', 0),
                'improvement_over_baseline': scrambled_test_a_improvement
            },
            'test_a2': {
                'ap': scrambled_test_a2_metrics.get('AP', 0),
                'improvement_over_baseline': scrambled_test_a2_improvement,
                'description': 'Uses normal segmentations (matched by rank)'
            },
            'test_b': {
                'ap': scrambled_test_b_metrics.get('AP', 0),
                'improvement_over_baseline': scrambled_test_b_improvement,
                'description': 'Uses GT categories'
            },
            'test_b2': {
                'ap': scrambled_test_b2_metrics.get('AP', 0),
                'improvement_over_baseline': scrambled_test_b2_improvement,
                'description': 'Uses normal categories (matched by rank)'
            },
            'test_c': {
                'ap': scrambled_test_c_metrics.get('AP', 0),
                'improvement_over_baseline': scrambled_test_c_improvement
            },
            'test_d': {
                'ap': scrambled_test_d_metrics.get('AP', 0),
                'improvement_over_baseline': scrambled_test_d_improvement,
                'vs_normal_baseline': scrambled_test_d_vs_normal,
                'description': 'Uses normal categories and scores (matched by rank)'
            }
        },
        'comparisons': {
            'test_a_difference': test_a_diff,
            'test_a2_difference': test_a2_diff,
            'test_b_difference': test_b_diff,
            'test_b2_difference': test_b2_diff,
            'test_c_difference': test_c_diff,
            'test_d_difference': test_d_diff,
            'test_a_improvement_difference': scrambled_test_a_improvement - normal_test_a_improvement,
            'test_b_improvement_difference': scrambled_test_b_improvement - normal_test_b_improvement,
            'test_b2_improvement_difference': scrambled_test_b2_improvement,
            'test_c_improvement_difference': scrambled_test_c_improvement  # Normal test C not run
        }
    }
    
    output_path = os.path.join(SCRAMBLED_DIR, "component_isolation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Score ranking plots saved to: {plots_output_dir}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

