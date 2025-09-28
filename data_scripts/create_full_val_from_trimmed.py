#!/usr/bin/env python3
"""
Create a full validation JSON from a trimmed validation JSON.

This script takes a trimmed validation JSON (like val_stride1.json) and creates a new
validation JSON (like val_stride1_all.json) that contains the same videos but with
all their original frames and annotations from the all_videos.json file.

Usage:
    python create_full_val_from_trimmed.py \
        --trimmed-val-json /path/to/val_stride1.json \
        --all-videos-json /path/to/all_videos.json \
        --output-json /path/to/val_stride1_all.json
"""

import json
import argparse
from collections import defaultdict

def create_full_val_from_trimmed(trimmed_val_json, all_videos_json, output_json):
    """
    Create a full validation JSON from a trimmed validation JSON.
    
    Args:
        trimmed_val_json: Path to the trimmed validation JSON
        all_videos_json: Path to the all_videos.json file
        output_json: Path where to save the new full validation JSON
    """
    
    print(f"Loading trimmed validation JSON: {trimmed_val_json}")
    with open(trimmed_val_json, 'r') as f:
        trimmed_data = json.load(f)
    
    print(f"Loading all videos JSON: {all_videos_json}")
    with open(all_videos_json, 'r') as f:
        all_data = json.load(f)
    
    # Get the video IDs from the trimmed validation set
    trimmed_video_ids = set(video['id'] for video in trimmed_data['videos'])
    print(f"Found {len(trimmed_video_ids)} videos in trimmed validation set")
    
    # Get the category IDs from the trimmed validation set
    trimmed_category_ids = set(cat['id'] for cat in trimmed_data['categories'])
    print(f"Found {len(trimmed_category_ids)} categories in trimmed validation set")
    
    # Extract full videos from all_videos.json
    full_videos = []
    for video in all_data['videos']:
        if video['id'] in trimmed_video_ids:
            full_videos.append(video)
    
    print(f"Extracted {len(full_videos)} full videos from all_videos.json")
    
    # Extract full annotations from all_videos.json
    full_annotations = []
    for annotation in all_data['annotations']:
        if (annotation['video_id'] in trimmed_video_ids and 
            annotation['category_id'] in trimmed_category_ids):
            full_annotations.append(annotation)
    
    print(f"Extracted {len(full_annotations)} full annotations from all_videos.json")
    
    # Extract categories
    full_categories = []
    for category in all_data['categories']:
        if category['id'] in trimmed_category_ids:
            full_categories.append(category)
    
    print(f"Extracted {len(full_categories)} categories")
    
    # Create the new validation JSON
    full_val_data = {
        'videos': full_videos,
        'annotations': full_annotations,
        'categories': full_categories,
        'info': all_data.get('info', {}),
        'licenses': all_data.get('licenses', [])
    }
    
    # Save the new validation JSON
    print(f"Saving full validation JSON to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(full_val_data, f, indent=2)
    
    # Print statistics comparison
    print("\n" + "="*80)
    print("STATISTICS COMPARISON")
    print("="*80)
    
    # Calculate statistics for trimmed set
    trimmed_total_frames = sum(video['length'] for video in trimmed_data['videos'])
    trimmed_total_annotations = len(trimmed_data['annotations'])
    trimmed_annotated_frames = sum(
        sum(1 for seg in ann['segmentations'] if seg is not None)
        for ann in trimmed_data['annotations']
    )
    
    # Calculate statistics for full set
    full_total_frames = sum(video['length'] for video in full_val_data['videos'])
    full_total_annotations = len(full_val_data['annotations'])
    full_annotated_frames = sum(
        sum(1 for seg in ann['segmentations'] if seg is not None)
        for ann in full_val_data['annotations']
    )
    
    print(f"{'Metric':<25} {'Trimmed':<15} {'Full':<15} {'Ratio':<10}")
    print("-" * 65)
    print(f"{'Videos':<25} {len(trimmed_data['videos']):<15} {len(full_val_data['videos']):<15} {len(full_val_data['videos'])/len(trimmed_data['videos']):<10.2f}")
    print(f"{'Total Frames':<25} {trimmed_total_frames:<15} {full_total_frames:<15} {full_total_frames/trimmed_total_frames:<10.2f}")
    print(f"{'Annotations':<25} {trimmed_total_annotations:<15} {full_total_annotations:<15} {full_total_annotations/trimmed_total_annotations:<10.2f}")
    print(f"{'Annotated Frames':<25} {trimmed_annotated_frames:<15} {full_annotated_frames:<15} {full_annotated_frames/trimmed_annotated_frames:<10.2f}")
    
    # Print per-video statistics
    print(f"\n{'Video ID':<10} {'Trimmed Frames':<15} {'Full Frames':<15} {'Ratio':<10}")
    print("-" * 50)
    
    # Create mapping for easy lookup
    trimmed_video_lengths = {v['id']: v['length'] for v in trimmed_data['videos']}
    full_video_lengths = {v['id']: v['length'] for v in full_val_data['videos']}
    
    for video_id in sorted(trimmed_video_ids):
        trimmed_len = trimmed_video_lengths.get(video_id, 0)
        full_len = full_video_lengths.get(video_id, 0)
        ratio = full_len / trimmed_len if trimmed_len > 0 else 0
        print(f"{video_id:<10} {trimmed_len:<15} {full_len:<15} {ratio:<10.2f}")
    
    print(f"\nSuccessfully created full validation JSON: {output_json}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Create full validation JSON from trimmed validation JSON')
    parser.add_argument('--trimmed-val-json', type=str, required=True,
                       help='Path to the trimmed validation JSON file')
    parser.add_argument('--all-videos-json', type=str, required=True,
                       help='Path to the all_videos.json file')
    parser.add_argument('--output-json', type=str, required=True,
                       help='Path where to save the new full validation JSON')
    
    args = parser.parse_args()
    
    create_full_val_from_trimmed(
        args.trimmed_val_json,
        args.all_videos_json,
        args.output_json
    )

if __name__ == "__main__":
    main()
