#!/usr/bin/env python3
"""
For each video, find the top scoring prediction and identify which videos have wrong class predictions.
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from collections import defaultdict

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def find_top_scoring_prediction_per_video(results_json, val_json):
    """
    For each video, find the top scoring prediction and check if it's correct.
    """
    preds = load_json(results_json)
    gts = load_json(val_json)
    
    # Get category information
    gt_category_names = {cat['id']: cat['name'] for cat in gts['categories']}
    
    # Group predictions by video
    pred_by_video = defaultdict(list)
    for pred in preds:
        pred_by_video[pred['video_id']].append(pred)
    
    # Group ground truth by video
    gt_by_video = defaultdict(list)
    for ann in gts['annotations']:
        gt_by_video[ann['video_id']].append(ann)
    
    # Get video metadata (if available)
    video_metadata = {}
    if 'videos' in gts:
        for video in gts['videos']:
            video_metadata[video['id']] = video.get('file_name', f"video_{video['id']}")
    
    top_predictions = []
    
    for video_id in set(list(pred_by_video.keys()) + list(gt_by_video.keys())):
        video_preds = pred_by_video.get(video_id, [])
        video_gts = gt_by_video.get(video_id, [])
        
        if not video_gts:
            continue
        
        # Get ground truth class (assuming single class per video)
        gt_class_id = video_gts[0]['category_id']
        gt_class_name = gt_category_names.get(gt_class_id, f"ID {gt_class_id}")
        
        # Get video name
        video_name = video_metadata.get(video_id, f"video_{video_id}")
        
        if not video_preds:
            # No predictions for this video
            top_predictions.append({
                'video_id': video_id,
                'video_name': video_name,
                'gt_class_name': gt_class_name,
                'top_pred_class_name': 'No Prediction',
                'top_pred_score': 0.0,
                'classification_correct': False,
                'total_predictions': 0
            })
            continue
        
        # Find the top scoring prediction
        top_pred = max(video_preds, key=lambda p: p.get('score', 0.0))
        top_pred_class_id = top_pred.get('category_id')
        top_pred_class_name = gt_category_names.get(top_pred_class_id, f"ID {top_pred_class_id}")
        top_pred_score = top_pred.get('score', 0.0)
        
        # Check if classification is correct
        classification_correct = (top_pred_class_id == gt_class_id)
        
        top_predictions.append({
            'video_id': video_id,
            'video_name': video_name,
            'gt_class_name': gt_class_name,
            'top_pred_class_name': top_pred_class_name,
            'top_pred_score': top_pred_score,
            'classification_correct': classification_correct,
            'total_predictions': len(video_preds)
        })
    
    return top_predictions

def main():
    if len(sys.argv) != 3:
        print("Usage: python top_scoring_prediction_per_video.py <results_json> <val_json>")
        print("Example: python top_scoring_prediction_per_video.py results.json val.json")
        sys.exit(1)
    
    results_json = sys.argv[1]
    val_json = sys.argv[2]
    
    if not os.path.exists(results_json):
        print(f"Error: Results JSON file not found: {results_json}")
        sys.exit(1)
    
    if not os.path.exists(val_json):
        print(f"Error: Validation JSON file not found: {val_json}")
        sys.exit(1)
    
    print("Finding top scoring prediction for each video...")
    print(f"Results: {results_json}")
    print(f"Ground truth: {val_json}")
    
    top_predictions = find_top_scoring_prediction_per_video(results_json, val_json)
    
    print("=" * 100)
    print("TOP SCORING PREDICTION PER VIDEO")
    print("=" * 100)
    
    if top_predictions:
        print(f"Found {len(top_predictions)} videos:")
        print()
        print(f"{'Video ID':<8} {'Video Name':<20} {'GT Class':<15} {'Top Pred Class':<15} {'Score':<8} {'Correct':<8}")
        print("-" * 100)
        
        # Sort by video ID for consistent output
        top_predictions.sort(key=lambda x: x['video_id'])
        
        for pred in top_predictions:
            correct_str = "✓" if pred['classification_correct'] else "✗"
            print(f"{pred['video_id']:<8} {pred['video_name']:<20} {pred['gt_class_name']:<15} {pred['top_pred_class_name']:<15} {pred['top_pred_score']:<8.3f} {correct_str:<8}")
        
        # Summary statistics
        print(f"\nSummary Statistics:")
        print("-" * 50)
        
        # Overall classification accuracy
        correct_count = sum(1 for p in top_predictions if p['classification_correct'])
        total_count = len(top_predictions)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"Overall Classification Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Correct predictions: {correct_count}/{total_count}")
        print(f"Wrong predictions: {total_count - correct_count}/{total_count}")
        
        # Score statistics
        scores = [p['top_pred_score'] for p in top_predictions if p['top_pred_score'] > 0]
        if scores:
            print(f"\nScore Statistics:")
            print(f"  Mean Score: {np.mean(scores):.3f}")
            print(f"  Min Score:  {np.min(scores):.3f}")
            print(f"  Max Score:  {np.max(scores):.3f}")
            print(f"  Std Score:  {np.std(scores):.3f}")
        
        # Per-class accuracy
        print(f"\nPer-Class Classification Accuracy:")
        print("-" * 50)
        class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        for pred in top_predictions:
            class_name = pred['gt_class_name']
            class_stats[class_name]['total'] += 1
            if pred['classification_correct']:
                class_stats[class_name]['correct'] += 1
        
        for class_name, stats in sorted(class_stats.items()):
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            print(f"{class_name:15s}: {accuracy:.3f} ({accuracy*100:.1f}%) - {stats['correct']}/{stats['total']} videos")
        
        # List of wrong predictions
        wrong_predictions = [p for p in top_predictions if not p['classification_correct']]
        if wrong_predictions:
            print(f"\nVideos with WRONG top scoring predictions:")
            print("-" * 80)
            for pred in wrong_predictions:
                print(f"Video {pred['video_id']:3d} ({pred['video_name']:20s}): GT={pred['gt_class_name']:12s} | Pred={pred['top_pred_class_name']:12s} | Score={pred['top_pred_score']:.3f}")
        
        # List of correct predictions
        correct_predictions = [p for p in top_predictions if p['classification_correct']]
        if correct_predictions:
            print(f"\nVideos with CORRECT top scoring predictions:")
            print("-" * 80)
            for pred in correct_predictions:
                print(f"Video {pred['video_id']:3d} ({pred['video_name']:20s}): GT={pred['gt_class_name']:12s} | Pred={pred['top_pred_class_name']:12s} | Score={pred['top_pred_score']:.3f}")
        
        # Save results
        df = pd.DataFrame(top_predictions)
        output_path = "top_scoring_prediction_per_video.csv"
        df.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
        
    else:
        print("No videos found.")

if __name__ == "__main__":
    main()
