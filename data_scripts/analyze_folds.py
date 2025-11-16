#!/usr/bin/env python3
"""
Analyze video counts by category for each fold in train and val sets.
"""

import json
from collections import defaultdict
from pathlib import Path

def count_videos_by_category(json_file):
    """Count videos by category in a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create category mapping
    category_map = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    
    # Get all videos
    videos = data.get('videos', [])
    video_ids = {v['id'] for v in videos}
    
    # Count categories per video
    video_categories = defaultdict(set)  # video_id -> set of category_ids
    
    for annotation in data.get('annotations', []):
        video_id = annotation.get('video_id')
        category_id = annotation.get('category_id')
        if video_id in video_ids and category_id:
            video_categories[video_id].add(category_id)
    
    # Count videos per category
    category_video_count = defaultdict(int)
    for video_id, categories in video_categories.items():
        for category_id in categories:
            category_name = category_map.get(category_id, f"Unknown_{category_id}")
            category_video_count[category_name] += 1
    
    return category_video_count, category_map

def get_category_order(categories):
    """Return categories in the specified order: Chinook, Coho, Atlantic, Rainbow Trout, Brown Trout."""
    # Define the desired order
    order = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
    
    # Create ordered list, preserving order for known categories
    ordered = []
    for cat in order:
        if cat in categories:
            ordered.append(cat)
    
    # Add any remaining categories that weren't in the order list
    for cat in sorted(categories):
        if cat not in order:
            ordered.append(cat)
    
    return ordered

def main():
    base_dir = Path("/home/simone/shared-data/fishway_ytvis")
    
    # Find all fold files
    train_folds = sorted(base_dir.glob("train_fold*.json"))
    val_folds = sorted(base_dir.glob("val_fold*.json"))
    
    print("=" * 80)
    print("VIDEO COUNTS BY CATEGORY FOR EACH FOLD")
    print("=" * 80)
    print()
    
    # Process train folds
    print("TRAIN SET:")
    print("-" * 80)
    all_train_categories = set()
    train_results = {}
    
    for train_file in train_folds:
        fold_num = train_file.stem.replace("train_fold", "")
        category_counts, category_map = count_videos_by_category(train_file)
        train_results[fold_num] = category_counts
        all_train_categories.update(category_counts.keys())
    
    # Print train results in a table format
    categories_sorted = get_category_order(all_train_categories)
    fold_nums = sorted(train_results.keys())
    
    print(f"{'Category':<20}", end="")
    for fold_num in fold_nums:
        print(f"{'Fold' + fold_num:>10}", end="")
    print(f"{'Total':>10}")
    print("-" * 80)
    
    for category in categories_sorted:
        print(f"{category:<20}", end="")
        total = 0
        for fold_num in fold_nums:
            count = train_results[fold_num].get(category, 0)
            print(f"{count:>10}", end="")
            total += count
        print(f"{total:>10}")
    
    # Print total row for train
    print(f"{'Total':<20}", end="")
    grand_total = 0
    for fold_num in fold_nums:
        fold_total = sum(train_results[fold_num].values())
        print(f"{fold_total:>10}", end="")
        grand_total += fold_total
    print(f"{grand_total:>10}")
    
    print()
    print()
    
    # Process val folds
    print("VAL SET:")
    print("-" * 80)
    all_val_categories = set()
    val_results = {}
    
    for val_file in val_folds:
        fold_num = val_file.stem.replace("val_fold", "")
        category_counts, category_map = count_videos_by_category(val_file)
        val_results[fold_num] = category_counts
        all_val_categories.update(category_counts.keys())
    
    # Print val results in a table format
    categories_sorted = get_category_order(all_val_categories)
    fold_nums = sorted(val_results.keys())
    
    print(f"{'Category':<20}", end="")
    for fold_num in fold_nums:
        print(f"{'Fold' + fold_num:>10}", end="")
    print(f"{'Total':>10}")
    print("-" * 80)
    
    for category in categories_sorted:
        print(f"{category:<20}", end="")
        total = 0
        for fold_num in fold_nums:
            count = val_results[fold_num].get(category, 0)
            print(f"{count:>10}", end="")
            total += count
        print(f"{total:>10}")
    
    # Print total row for val
    print(f"{'Total':<20}", end="")
    grand_total = 0
    for fold_num in fold_nums:
        fold_total = sum(val_results[fold_num].values())
        print(f"{fold_total:>10}", end="")
        grand_total += fold_total
    print(f"{grand_total:>10}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Print summary statistics
    print("\nTrain Set Totals:")
    for category in get_category_order(all_train_categories):
        total = sum(train_results[fn].get(category, 0) for fn in train_results.keys())
        print(f"  {category}: {total} videos")
    
    print("\nVal Set Totals:")
    for category in get_category_order(all_val_categories):
        total = sum(val_results[fn].get(category, 0) for fn in val_results.keys())
        print(f"  {category}: {total} videos")

if __name__ == "__main__":
    main()

