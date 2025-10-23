import json
import random
import copy
from collections import defaultdict
import numpy as np

def print_dataset_statistics(ytvis_data, dataset_name):
    """
    Print detailed statistics about the dataset including:
    - Number of videos per category
    - Number of total frames and annotated frames per category
    - Frame statistics (mean, median, min, max) for both total and annotated frames
    """
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} DATASET STATISTICS")
    print(f"{'='*80}")
    
    # Create category ID to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in ytvis_data['categories']}
    
    # Initialize statistics containers
    cat_video_count = defaultdict(int)
    cat_total_frames = defaultdict(int)
    cat_annotated_frames = defaultdict(int)
    cat_total_frames_per_video = defaultdict(list)  # Track total frames per video
    cat_annotated_frames_per_video = defaultdict(list)  # Track annotated frames per video
    cat_video_ids = defaultdict(set)
    
    # Process videos and annotations
    video_id_to_length = {video['id']: video['length'] for video in ytvis_data['videos']}
    
    for ann in ytvis_data['annotations']:
        cat_id = ann['category_id']
        video_id = ann['video_id']
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        
        # Count videos per category
        cat_video_ids[cat_id].add(video_id)
        
        # Count total frames per category
        total_frames = video_id_to_length.get(video_id, 0)
        cat_total_frames[cat_id] += total_frames
        cat_total_frames_per_video[cat_id].append(total_frames)
        
        # Count annotated frames per category
        annotated_frames = sum(1 for seg in ann['segmentations'] if seg is not None)
        cat_annotated_frames[cat_id] += annotated_frames
        
        # Collect annotated frames per video for this category
        cat_annotated_frames_per_video[cat_id].append(annotated_frames)
    
    # Calculate video counts
    for cat_id, video_ids in cat_video_ids.items():
        cat_video_count[cat_id] = len(video_ids)
    
    # Print statistics by category
    print(f"\n{'Category':<20} {'Videos':<8} {'Total Frames':<12} {'Ann Frames':<12} {'Total Frame Stats':<25} {'Ann Frame Stats':<25}")
    print("-" * 120)
    
    for cat_id in sorted(cat_video_count.keys()):
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        video_count = cat_video_count[cat_id]
        total_frames = cat_total_frames[cat_id]
        annotated_frames = cat_annotated_frames[cat_id]
        total_frames_per_video = cat_total_frames_per_video[cat_id]
        annotated_frames_per_video = cat_annotated_frames_per_video[cat_id]
        
        # Calculate total frame stats
        if total_frames_per_video:
            total_mean = np.mean(total_frames_per_video)
            total_median = np.median(total_frames_per_video)
            total_min = np.min(total_frames_per_video)
            total_max = np.max(total_frames_per_video)
            total_stats = f"{total_mean:.1f}/{total_median:.1f}/{total_min}/{total_max}"
        else:
            total_stats = "N/A"
        
        # Calculate annotated frame stats
        if annotated_frames_per_video:
            ann_mean = np.mean(annotated_frames_per_video)
            ann_median = np.median(annotated_frames_per_video)
            ann_min = np.min(annotated_frames_per_video)
            ann_max = np.max(annotated_frames_per_video)
            ann_stats = f"{ann_mean:.1f}/{ann_median:.1f}/{ann_min}/{ann_max}"
        else:
            ann_stats = "N/A"
        
        print(f"{cat_name:<20} {video_count:<8} {total_frames:<12} {annotated_frames:<12} {total_stats:<25} {ann_stats:<25}")
    
    
    # Print overall statistics
    print("\n" + "-" * 120)
    total_videos = len(ytvis_data['videos'])
    total_annotations = len(ytvis_data['annotations'])
    total_total_frames = sum(cat_total_frames.values())
    total_annotated_frames = sum(cat_annotated_frames.values())
    
    all_total_frames_per_video = []
    all_annotated_frames_per_video = []
    for frames_list in cat_total_frames_per_video.values():
        all_total_frames_per_video.extend(frames_list)
    for frames_list in cat_annotated_frames_per_video.values():
        all_annotated_frames_per_video.extend(frames_list)
    
    # Calculate overall total frame stats
    if all_total_frames_per_video:
        overall_total_mean = np.mean(all_total_frames_per_video)
        overall_total_median = np.median(all_total_frames_per_video)
        overall_total_min = np.min(all_total_frames_per_video)
        overall_total_max = np.max(all_total_frames_per_video)
        overall_total_stats = f"{overall_total_mean:.1f}/{overall_total_median:.1f}/{overall_total_min}/{overall_total_max}"
    else:
        overall_total_stats = "N/A"
    
    # Calculate overall annotated frame stats
    if all_annotated_frames_per_video:
        overall_ann_mean = np.mean(all_annotated_frames_per_video)
        overall_ann_median = np.median(all_annotated_frames_per_video)
        overall_ann_min = np.min(all_annotated_frames_per_video)
        overall_ann_max = np.max(all_annotated_frames_per_video)
        overall_ann_stats = f"{overall_ann_mean:.1f}/{overall_ann_median:.1f}/{overall_ann_min}/{overall_ann_max}"
    else:
        overall_ann_stats = "N/A"
    
    print(f"{'TOTAL':<20} {total_videos:<8} {total_total_frames:<12} {total_annotated_frames:<12} {overall_total_stats:<25} {overall_ann_stats:<25}")
    print(f"Total annotations: {total_annotations}")
    print(f"Total categories: {len(ytvis_data['categories'])}")
    print(f"{'='*120}")


def validate_dataset_integrity(ytvis_data, dataset_name, other_dataset=None):
    """
    Validate data integrity for the created dataset:
    - Exactly 1 annotation per video
    - Video length matches annotation length
    - Segmentation, bbox, and area lists match their stated lengths
    - Number of file_names matches video length
    - Segmentation format validation (RLE structure)
    - Train/val overlap check (if other_dataset provided)
    """
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} DATA VALIDATION")
    print(f"{'='*80}")
    
    # Get all video IDs and annotation video IDs
    video_ids = set(video['id'] for video in ytvis_data['videos'])
    annotation_video_ids = set(ann['video_id'] for ann in ytvis_data['annotations'])
    
    # Check 1: Exactly 1 annotation per video
    print("\n1. Checking annotation count per video...")
    video_ann_counts = {}
    for ann in ytvis_data['annotations']:
        video_id = ann['video_id']
        video_ann_counts[video_id] = video_ann_counts.get(video_id, 0) + 1
    
    annotation_errors = []
    for video_id in video_ids:
        ann_count = video_ann_counts.get(video_id, 0)
        if ann_count != 1:
            annotation_errors.append(f"Video {video_id}: {ann_count} annotations (expected 1)")
    
    if annotation_errors:
        print(f"❌ Found {len(annotation_errors)} annotation count errors:")
        for error in annotation_errors:
            print(f"  - {error}")
    else:
        print(f"✅ All {len(video_ids)} videos have exactly 1 annotation")
    
    # Check 2: All lengths should match (video length, annotation length, list lengths, file count)
    print("\n2. Checking all length consistency...")
    length_errors = []
    for video in ytvis_data['videos']:
        video_id = video['id']
        video_length = video['length']
        file_names_count = len(video['file_names'])
        
        # Find corresponding annotation
        ann = next((a for a in ytvis_data['annotations'] if a['video_id'] == video_id), None)
        if ann:
            ann_length = ann['length']
            seg_length = len(ann['segmentations'])
            bbox_length = len(ann['bboxes'])
            area_length = len(ann['areas'])
            
            # Check if all lengths match
            lengths = [video_length, ann_length, seg_length, bbox_length, area_length, file_names_count]
            if not all(length == lengths[0] for length in lengths):
                error_msg = f"Video {video_id}: lengths don't match - video:{video_length}, ann:{ann_length}, seg:{seg_length}, bbox:{bbox_length}, area:{area_length}, files:{file_names_count}"
                length_errors.append(error_msg)
    
    if length_errors:
        print(f"❌ Found {len(length_errors)} length consistency errors:")
        for error in length_errors:
            print(f"  - {error}")
    else:
        print(f"✅ All {len(video_ids)} videos have consistent lengths across all fields")
    
    # Check 3: Segmentation format validation
    print("\n3. Checking segmentation format...")
    segmentation_errors = []
    for ann in ytvis_data['annotations']:
        video_id = ann['video_id']
        for idx, seg in enumerate(ann.get('segmentations', [])):
            if seg is not None:
                if isinstance(seg, dict):
                    # Check RLE format
                    if "counts" not in seg or "size" not in seg:
                        segmentation_errors.append(f"Video {video_id} frame {idx}: invalid RLE segmentation (missing 'counts' or 'size')")
                elif not isinstance(seg, list):
                    # Check polygon format
                    segmentation_errors.append(f"Video {video_id} frame {idx}: segmentation has unexpected type {type(seg)} (expected dict or list)")
    
    if segmentation_errors:
        print(f"❌ Found {len(segmentation_errors)} segmentation format errors:")
        for error in segmentation_errors:
            print(f"  - {error}")
    else:
        print(f"✅ All {sum(len(ann.get('segmentations', [])) for ann in ytvis_data['annotations'])} segmentations have valid format")
    
    # Check 4: Train/val overlap check (if other dataset provided)
    overlap_errors = []
    if other_dataset is not None:
        print(f"\n4. Checking overlap with {other_dataset.get('dataset_name', 'other')} set...")
        current_video_ids = set(video['id'] for video in ytvis_data['videos'])
        other_video_ids = set(video['id'] for video in other_dataset['videos'])
        overlap = current_video_ids.intersection(other_video_ids)
        
        if overlap:
            overlap_errors = list(overlap)
            print(f"❌ Found {len(overlap_errors)} overlapping videos:")
            for video_id in sorted(overlap_errors):
                print(f"  - Video {video_id}")
        else:
            print(f"✅ No overlap found - {len(current_video_ids)} videos in {dataset_name} set are unique")
    
    # Summary
    total_errors = len(annotation_errors) + len(length_errors) + len(segmentation_errors) + len(overlap_errors)
    print(f"\n{'='*80}")
    if total_errors == 0:
        print(f"✅ {dataset_name.upper()} DATASET VALIDATION PASSED - No errors found!")
    else:
        print(f"❌ {dataset_name.upper()} DATASET VALIDATION FAILED - {total_errors} errors found!")
    print(f"{'='*80}")


def create_train_val_jsons(
    all_vids_json,
    output_train_json,
    output_val_json,
    categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic", "Rainbow Trout"],
    exclude_empty_train=True,
    balance=True,
    num_vids_val=2,
    max_vids_per_species_train=None,
    frame_skip=0,
    random_seed=None,
    max_frames_train=None,
    max_frames_val=None,
    min_anns_per_video=None
):
    """
    Create train and val YTVIS JSONs from a big all-videos YTVIS JSON.
    - categories_to_include: list of category names to include
    - exclude_empty_train: if True, trim empty frames from train set
    - balance: if True, balance number of videos per class
    - num_vids_val: number of videos per class for val set
    - max_vids_per_species_train: maximum number of videos per species for train set (None = as many as possible)
    - frame_skip: 0 = include every frame, 1 = every 2nd frame, 2 = every 3rd frame, etc.
    - random_seed: if set, ensures reproducible splits
    - max_frames_train: maximum number of annotated frames to keep per video in training set (default 75)
    - max_frames_val: maximum number of annotated frames to keep per video in validation set (default 45)
    - min_anns_per_video: minimum number of annotated frames required per video (None = no filtering)
    """
    if random_seed is not None:
        random.seed(random_seed)
    with open(all_vids_json, 'r') as f:
        data = json.load(f)
    # Map category names to ids
    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    cat_ids = [cat_name_to_id[name] for name in categories_to_include if name in cat_name_to_id]
    
    print(f"Available categories: {list(cat_name_to_id.keys())}")
    print(f"Requested categories: {categories_to_include}")
    print(f"Found category IDs: {cat_ids}")
    print(f"Found category names: {[cat_id_to_name[cid] for cid in cat_ids]}")
    # Find videos for each category
    video_to_cats = defaultdict(set)
    for ann in data['annotations']:
        if ann['category_id'] in cat_ids:
            video_to_cats[ann['video_id']].add(ann['category_id'])
    
    # Filter out videos with fewer than minimum annotations if specified
    if min_anns_per_video is not None:
        print(f"\nFiltering videos with fewer than {min_anns_per_video} annotated frames...")
        
        # Count annotated frames per video
        video_ann_counts = defaultdict(int)
        for ann in data['annotations']:
            if ann['category_id'] in cat_ids:
                annotated_frames = sum(1 for seg in ann['segmentations'] if seg is not None)
                video_ann_counts[ann['video_id']] += annotated_frames
        
        # Find videos to remove
        videos_to_remove = []
        for video_id, ann_count in video_ann_counts.items():
            if ann_count < min_anns_per_video:
                videos_to_remove.append(video_id)
        
        if videos_to_remove:
            print(f"Removing {len(videos_to_remove)} videos with insufficient annotations:")
            for video_id in videos_to_remove:
                print(f"  - Video {video_id}: {video_ann_counts[video_id]} annotated frames")
            
            # Remove videos from video_to_cats
            for video_id in videos_to_remove:
                if video_id in video_to_cats:
                    del video_to_cats[video_id]
            
            print(f"Remaining videos after filtering: {len(video_to_cats)}")
        else:
            print("No videos removed during filtering")
    
    # Build val set
    val_videos = set()
    print(f"\nBuilding validation set with num_vids_val={num_vids_val}, balance={balance}")
    for cat_id in cat_ids:
        vids = [vid for vid, cats in video_to_cats.items() if cat_id in cats]
        vids = list(set(vids) - val_videos)  # avoid overlap
        num = min(num_vids_val, len(vids))
        print(f"Category {cat_id_to_name[cat_id]}: {len(vids)} available videos, selecting {num}")
        if balance:
            vids = random.sample(vids, num) if num > 0 else []
        else:
            vids = random.sample(vids, num) if num > 0 else []
        val_videos.update(vids)
        print(f"  Selected videos: {vids}")
    print(f"Total validation videos: {len(val_videos)}")
    
    # Build train set
    train_videos = set([vid for vid in video_to_cats if vid not in val_videos])
    print(f"Initial train videos: {len(train_videos)}")
    
    # Apply max_vids_per_species_train cap to training videos
    if max_vids_per_species_train is not None:
        vids_per_cat = {cat_id: [vid for vid in train_videos if cat_id in video_to_cats[vid]] for cat_id in cat_ids}
        capped_train_videos = set()
        for cat_id, vids in vids_per_cat.items():
            n = min(max_vids_per_species_train, len(vids))
            if n > 0:
                capped_train_videos.update(random.sample(vids, n))
        train_videos = capped_train_videos
    
    if balance:
        # For each class, balance to the minimum count across all species
        vids_per_cat = {cat_id: [vid for vid in train_videos if cat_id in video_to_cats[vid]] for cat_id in cat_ids}
        print(f"Videos per category before balancing: {[(cat_id_to_name[cat_id], len(vids)) for cat_id, vids in vids_per_cat.items()]}")
        min_count = min(len(v) for v in vids_per_cat.values()) if vids_per_cat.values() else 0
        print(f"Min count for balancing: {min_count}")
        balanced_train_videos = set()
        for cat_id, vids in vids_per_cat.items():
            n = min(min_count, len(vids))
            if n > 0:
                balanced_train_videos.update(random.sample(vids, n))
        train_videos = balanced_train_videos
        print(f"Train videos after balancing: {len(train_videos)}")
    else:
        print(f"No balancing applied, keeping {len(train_videos)} train videos")
    # Filter videos/annotations for train/val
    def filter_json(videoset, trim_empty, max_frames):
        vids = [v for v in data['videos'] if v['id'] in videoset]
        anns = [a for a in data['annotations'] if a['video_id'] in videoset and a['category_id'] in cat_ids]
        # For each video, find frames with at least one annotation
        vid_to_frames_with_ann = defaultdict(set)
        for ann in anns:
            for idx, seg in enumerate(ann['segmentations']):
                if seg is not None:
                    vid_to_frames_with_ann[ann['video_id']].add(idx)
        
        # Apply frame processing (capping and/or trimming)
        new_vids = []
        new_anns = []
        for v in vids:
            frames_with_ann = sorted(vid_to_frames_with_ann[v['id']])
            if not frames_with_ann:
                continue
            
            # Apply frame capping if specified
            if max_frames is not None and len(frames_with_ann) > max_frames:
                # Center the selection around the middle of the video
                mid = len(frames_with_ann) // 2
                half = max_frames // 2
                if max_frames % 2 == 0:
                    selected = frames_with_ann[mid-half:mid+half]
                else:
                    selected = frames_with_ann[mid-half:mid+half+1]
                frames_with_ann = selected
            
            # Apply trimming logic based on trim_empty parameter
            if trim_empty:
                # For training: only keep frames with annotations
                keep_idxs = set(frames_with_ann)
            else:
                # For validation: keep all frames (but still apply capping if specified)
                keep_idxs = set(range(v['length']))
                if max_frames is not None and len(keep_idxs) > max_frames:
                    # If we're capping, center the selection
                    mid = len(keep_idxs) // 2
                    half = max_frames // 2
                    if max_frames % 2 == 0:
                        keep_idxs = set(list(keep_idxs)[mid-half:mid+half])
                    else:
                        keep_idxs = set(list(keep_idxs)[mid-half:mid+half+1])
            
            # Create new video with selected frames
            v_new = copy.deepcopy(v)
            v_new['file_names'] = [f for i, f in enumerate(v['file_names']) if i in keep_idxs]
            v_new['length'] = len(v_new['file_names'])
            new_vids.append(v_new)
                
            # Trim segmentations/bboxes/areas for each annotation
            for ann in [a for a in anns if a['video_id'] == v['id']]:
                ann_new = copy.deepcopy(ann)
                ann_new['segmentations'] = [s for i, s in enumerate(ann['segmentations']) if i in keep_idxs]
                ann_new['bboxes'] = [b for i, b in enumerate(ann['bboxes']) if i in keep_idxs]
                ann_new['areas'] = [a for i, a in enumerate(ann['areas']) if i in keep_idxs]
                ann_new['length'] = len(ann_new['segmentations'])
                new_anns.append(ann_new)
        
        vids = new_vids
        anns = new_anns
        # Apply frame skipping
        if frame_skip > 0:
            def skip_frames(seq):
                return seq[::frame_skip+1]
            # For videos
            for v in vids:
                v['file_names'] = skip_frames(v['file_names'])
                v['length'] = len(v['file_names'])
            # For annotations
            for ann in anns:
                ann['segmentations'] = skip_frames(ann['segmentations'])
                ann['bboxes'] = skip_frames(ann['bboxes'])
                ann['areas'] = skip_frames(ann['areas'])
                ann['length'] = len(ann['segmentations'])
        return vids, anns
    train_vids, train_anns = filter_json(train_videos, trim_empty=exclude_empty_train, max_frames=max_frames_train)
    val_vids, val_anns = filter_json(val_videos, trim_empty=False, max_frames=max_frames_val)
    # Build output jsons
    def build_json(vids, anns):
        used_cat_ids = set(a['category_id'] for a in anns)
        cats = [cat for cat in data['categories'] if cat['id'] in used_cat_ids]
        return {
            'videos': vids,
            'annotations': anns,
            'categories': cats,
            'info': data.get('info', {}),
            'licenses': data.get('licenses', [])
        }
    # Build the final JSONs
    train_json = build_json(train_vids, train_anns)
    val_json = build_json(val_vids, val_anns)
    
    with open(output_train_json, 'w') as f:
        json.dump(train_json, f, indent=2)
    with open(output_val_json, 'w') as f:
        json.dump(val_json, f, indent=2)
    print(f"Wrote train json to {output_train_json}")
    print(f"Wrote val json to {output_val_json}")
    
    # Print statistics for train and val sets
    print("\n--- TRAIN SET STATISTICS ---")
    print_dataset_statistics(train_json, "train")
    print("\n--- VALIDATION SET STATISTICS ---")
    print_dataset_statistics(val_json, "val")

    # Validate data integrity
    print("\n--- TRAIN SET VALIDATION ---")
    validate_dataset_integrity(train_json, "train", val_json)
    print("\n--- VALIDATION SET VALIDATION ---")
    validate_dataset_integrity(val_json, "val", train_json)


if __name__ == "__main__":
    create_train_val_jsons(
        all_vids_json="/data/fishway_ytvis/all_videos.json",
        output_train_json="/data/fishway_ytvis/train.json",
        output_val_json="/data/fishway_ytvis/val.json",
        #categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic", "Rainbow Trout"],
        exclude_empty_train=True,
        balance=False,
        num_vids_val=8,
        max_vids_per_species_train=50,
        frame_skip=0,
        random_seed=100,
        #max_frames_train=250,  # 5 windows of 15 frames for training
        #max_frames_val=250,     # 3 windows of 15 frames for validation
        min_anns_per_video=10  # Filter out videos with fewer than 30 annotated frames
    ) 