#!/usr/bin/env python3
"""
Create K-fold cross-validation splits from train and validation JSON files.

This script:
1. Merges train.json and val.json into a combined dataset
2. Performs stratified K-fold splitting (preserving class proportions)
3. Creates train_foldX.json and val_foldX.json for each fold
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path
import random


def get_video_category(video_id, annotations):
    """
    Get the category ID for a video based on its annotations.
    Assumes each video has exactly one annotation with one category.
    """
    for ann in annotations:
        if ann['video_id'] == video_id:
            return ann['category_id']
    return None


def create_k_folds(
    train_json_path,
    val_json_path,
    output_dir,
    k=5,
    random_seed=42
):
    """
    Create K-fold cross-validation splits from train and validation JSONs.
    
    Args:
        train_json_path: Path to training JSON file
        val_json_path: Path to validation JSON file
        output_dir: Directory to save fold JSONs
        k: Number of folds (default: 5)
        random_seed: Random seed for reproducibility
    """
    # Load JSON files
    print(f"Loading {train_json_path}...")
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    print(f"Loading {val_json_path}...")
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    # Merge train and val data
    print("\nMerging train and validation sets...")
    all_videos = train_data['videos'] + val_data['videos']
    all_annotations = train_data['annotations'] + val_data['annotations']
    
    # Use categories from train (should be same in both)
    categories = train_data['categories']
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    print(f"Total videos: {len(all_videos)}")
    print(f"Total annotations: {len(all_annotations)}")
    
    # Count videos per category
    video_categories = {}
    for video in all_videos:
        video_id = video['id']
        category_id = get_video_category(video_id, all_annotations)
        if category_id is not None:
            video_categories[video_id] = category_id
    
    # Print category distribution
    cat_counts = defaultdict(int)
    for video_id, cat_id in video_categories.items():
        cat_counts[cat_id] += 1
    
    print("\nCategory distribution in combined dataset:")
    for cat_id in sorted(cat_counts.keys()):
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        print(f"  {cat_name}: {cat_counts[cat_id]} videos")
    
    # Prepare data for stratified splitting
    # Group videos by category for stratified splitting
    videos_by_category = defaultdict(list)
    for video_id, category_id in video_categories.items():
        videos_by_category[category_id].append(video_id)
    
    # Shuffle videos within each category for reproducibility
    random.seed(random_seed)
    for category_id in videos_by_category:
        random.shuffle(videos_by_category[category_id])
    
    # Create stratified K-fold splits manually
    print(f"\nCreating {k}-fold stratified splits (random_seed={random_seed})...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a mapping from video_id to video/annotation objects for quick lookup
    video_dict = {v['id']: v for v in all_videos}
    annotation_dict = defaultdict(list)
    for ann in all_annotations:
        annotation_dict[ann['video_id']].append(ann)
    
    # Create folds: for each category, split videos into k folds
    # This ensures each fold has similar class distribution
    folds = [[] for _ in range(k)]  # Each fold is a list of video IDs
    
    for category_id, video_ids in videos_by_category.items():
        # Split videos of this category into k folds
        n_videos = len(video_ids)
        fold_sizes = [n_videos // k] * k
        # Distribute remainder videos across first few folds
        for i in range(n_videos % k):
            fold_sizes[i] += 1
        
        # Assign videos to folds
        start_idx = 0
        for fold_idx in range(k):
            end_idx = start_idx + fold_sizes[fold_idx]
            folds[fold_idx].extend(video_ids[start_idx:end_idx])
            start_idx = end_idx
    
    # Generate folds
    fold_stats = []
    for fold_idx in range(k):
        print(f"\n--- Fold {fold_idx + 1} ---")
        
        # Get video IDs for this fold
        # Train: all folds except current fold
        train_video_ids = []
        for i in range(k):
            if i != fold_idx:
                train_video_ids.extend(folds[i])
        
        # Val: current fold
        val_video_ids = folds[fold_idx]
        
        print(f"Train videos: {len(train_video_ids)}")
        print(f"Val videos: {len(val_video_ids)}")
        
        # Count categories in each split
        train_cat_counts = defaultdict(int)
        val_cat_counts = defaultdict(int)
        
        for vid_id in train_video_ids:
            train_cat_counts[video_categories[vid_id]] += 1
        for vid_id in val_video_ids:
            val_cat_counts[video_categories[vid_id]] += 1
        
        print("Train category distribution:")
        for cat_id in sorted(train_cat_counts.keys()):
            cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            print(f"  {cat_name}: {train_cat_counts[cat_id]} videos")
        
        print("Val category distribution:")
        for cat_id in sorted(val_cat_counts.keys()):
            cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            print(f"  {cat_name}: {val_cat_counts[cat_id]} videos")
        
        # Build train fold JSON
        train_fold_videos = [video_dict[vid_id] for vid_id in train_video_ids]
        train_fold_annotations = []
        for vid_id in train_video_ids:
            train_fold_annotations.extend(annotation_dict[vid_id])
        
        # Get categories used in this fold
        used_cat_ids = set(ann['category_id'] for ann in train_fold_annotations)
        train_fold_categories = [cat for cat in categories if cat['id'] in used_cat_ids]
        
        train_fold_json = {
            'videos': train_fold_videos,
            'annotations': train_fold_annotations,
            'categories': train_fold_categories,
            'info': train_data.get('info', {}),
            'licenses': train_data.get('licenses', [])
        }
        
        # Build val fold JSON
        val_fold_videos = [video_dict[vid_id] for vid_id in val_video_ids]
        val_fold_annotations = []
        for vid_id in val_video_ids:
            val_fold_annotations.extend(annotation_dict[vid_id])
        
        # Get categories used in this fold
        used_cat_ids = set(ann['category_id'] for ann in val_fold_annotations)
        val_fold_categories = [cat for cat in categories if cat['id'] in used_cat_ids]
        
        val_fold_json = {
            'videos': val_fold_videos,
            'annotations': val_fold_annotations,
            'categories': val_fold_categories,
            'info': val_data.get('info', {}),
            'licenses': val_data.get('licenses', [])
        }
        
        # Save fold JSONs
        train_fold_path = output_dir / f"train_fold{fold_idx + 1}.json"
        val_fold_path = output_dir / f"val_fold{fold_idx + 1}.json"
        
        with open(train_fold_path, 'w') as f:
            json.dump(train_fold_json, f, indent=2)
        print(f"  Saved: {train_fold_path}")
        
        with open(val_fold_path, 'w') as f:
            json.dump(val_fold_json, f, indent=2)
        print(f"  Saved: {val_fold_path}")
        
        # Store statistics
        fold_stats.append({
            'fold': fold_idx + 1,
            'train_videos': len(train_video_ids),
            'val_videos': len(val_video_ids),
            'train_categories': dict(train_cat_counts),
            'val_categories': dict(val_cat_counts)
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total folds: {k}")
    print(f"Total videos: {len(all_videos)}")
    print(f"Random seed: {random_seed}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFold statistics:")
    print(f"{'Fold':<6} {'Train Videos':<15} {'Val Videos':<15}")
    print("-" * 40)
    for stat in fold_stats:
        print(f"{stat['fold']:<6} {stat['train_videos']:<15} {stat['val_videos']:<15}")
    print(f"{'='*80}")
    
    print("\n✅ K-fold splits created successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Create K-fold cross-validation splits from train and validation JSON files"
    )
    parser.add_argument(
        '--train-json',
        type=str,
        default='/home/simone/shared-data/fishway_ytvis/train.json',
        help='Path to training JSON file (default: /home/simone/shared-data/fishway_ytvis/train.json)'
    )
    parser.add_argument(
        '--val-json',
        type=str,
        default='/home/simone/shared-data/fishway_ytvis/val.json',
        help='Path to validation JSON file (default: /home/simone/shared-data/fishway_ytvis/val.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/simone/shared-data/fishway_ytvis',
        help='Directory to save fold JSONs (default: /home/simone/shared-data/fishway_ytvis)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of folds (default: 5)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    create_k_folds(
        train_json_path=args.train_json,
        val_json_path=args.val_json,
        output_dir=args.output_dir,
        k=args.k,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()

