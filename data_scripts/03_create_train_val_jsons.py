import json
import random
import copy
from collections import defaultdict
import numpy as np
import re

"""
IMPORTANT FIXES APPLIED:
1. Fixed orphaned annotations bug - annotations are now properly cleaned up when videos are filtered out
2. max_anns_val/max_anns_train now properly limit BOTH annotations AND video frames to the specified number
3. Validation set now maintains the correct number of videos per species even after filtering
4. Added buffer mechanism to account for videos that might be filtered out during processing
5. Added video name matching for fine-tuning to handle different video IDs between datasets

FINE-TUNING USAGE:
To create fine-tuning datasets that exclude previously trained videos:
1. Uncomment the "FINE-TUNING DATASET CREATION" section at the bottom of this file
2. Set existing_train_json="/path/to/your/existing/train.json" to point to your trained model's train.json
3. Adjust other parameters as needed (max_vids_per_species_train, num_vids_val, etc.)
4. Run: python 03_create_train_val_jsons.py

This will create finetune_train.json and finetune_val.json with:
- No previously trained videos in validation set
- Prioritized NEW videos in training set
- Existing videos only used as minimal fallback when needed
"""

def extract_video_name(file_path):
    """
    Extract video name from file path.
    Examples:
    - "Credit__2024__08162024-08192024__24  08  16  10  57__3/00102.jpg" -> "Credit__2024__08162024-08192024__24  08  16  10  57__3"
    - "Ganaraska__Ganaraska 2023__07252023-07282023__23  07  25  13  01__10/00083.jpg" -> "Ganaraska__Ganaraska 2023__07252023-07282023__23  07  25  13  01__10"
    """
    if not file_path:
        return None
    
    # Remove the file extension and frame number
    # Pattern: video_name/frame_number.jpg -> video_name
    parts = file_path.split('/')
    if len(parts) >= 2:
        video_name = parts[0]  # Everything before the last slash
        return video_name
    
    return None

def get_video_name_to_id_mapping(videos):
    """
    Create a mapping from video names to video IDs.
    """
    name_to_id = {}
    for video in videos:
        if video['file_names']:
            video_name = extract_video_name(video['file_names'][0])
            if video_name:
                name_to_id[video_name] = video['id']
    return name_to_id

def print_dataset_statistics(ytvis_data, dataset_name):
    """
    Print detailed statistics about the dataset including:
    - Number of videos per category
    - Number of total frames and annotated frames per category
    - Frame statistics (mean, median, min, max) for both total and annotated frames per category
    - Videos with fewer than 30 annotated frames per category
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
        
        # Count total frames for this video
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
    print(f"\n{'Category':<15} {'Videos':<7} {'Total Frames':<12} {'Ann. Frames':<11} {'Total Frame Stats':<20} {'Ann. Frame Stats':<20}")
    print("-" * 95)
    
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
            total_stats = f"{total_mean:.0f}/{total_median:.0f}/{total_min}/{total_max}"
        else:
            total_stats = "N/A"
        
        # Calculate annotated frame stats
        if annotated_frames_per_video:
            ann_mean = np.mean(annotated_frames_per_video)
            ann_median = np.median(annotated_frames_per_video)
            ann_min = np.min(annotated_frames_per_video)
            ann_max = np.max(annotated_frames_per_video)
            ann_stats = f"{ann_mean:.0f}/{ann_median:.0f}/{ann_min}/{ann_max}"
        else:
            ann_stats = "N/A"
        
        print(f"{cat_name:<15} {video_count:<7} {total_frames:<12} {annotated_frames:<11} {total_stats:<20} {ann_stats:<20}")
    
    
    # Print overall statistics
    print("\n" + "-" * 95)
    total_videos = len(ytvis_data['videos'])
    total_annotations = len(ytvis_data['annotations'])
    total_total_frames = sum(cat_total_frames.values())
    total_annotated_frames = sum(cat_annotated_frames.values())
    
    # Calculate overall frame statistics
    all_total_frames_per_video = []
    all_annotated_frames_per_video = []
    for frames_list in cat_total_frames_per_video.values():
        all_total_frames_per_video.extend(frames_list)
    for frames_list in cat_annotated_frames_per_video.values():
        all_annotated_frames_per_video.extend(frames_list)
    
    # Overall total frame stats
    if all_total_frames_per_video:
        total_overall_mean = np.mean(all_total_frames_per_video)
        total_overall_median = np.median(all_total_frames_per_video)
        total_overall_min = np.min(all_total_frames_per_video)
        total_overall_max = np.max(all_total_frames_per_video)
        total_overall_stats = f"{total_overall_mean:.0f}/{total_overall_median:.0f}/{total_overall_min}/{total_overall_max}"
    else:
        total_overall_stats = "N/A"
    
    # Overall annotated frame stats
    if all_annotated_frames_per_video:
        ann_overall_mean = np.mean(all_annotated_frames_per_video)
        ann_overall_median = np.median(all_annotated_frames_per_video)
        ann_overall_min = np.min(all_annotated_frames_per_video)
        ann_overall_max = np.max(all_annotated_frames_per_video)
        ann_overall_stats = f"{ann_overall_mean:.0f}/{ann_overall_median:.0f}/{ann_overall_min}/{ann_overall_max}"
    else:
        ann_overall_stats = "N/A"
    
    print(f"{'TOTAL':<15} {total_videos:<7} {total_total_frames:<12} {total_annotated_frames:<11} {total_overall_stats:<20} {ann_overall_stats:<20}")
    print(f"Total annotations: {total_annotations}")
    print(f"Total categories: {len(ytvis_data['categories'])}")
    annotation_ratio = (total_annotated_frames / total_total_frames * 100) if total_total_frames > 0 else 0
    print(f"Annotation coverage: {annotation_ratio:.1f}% of frames have annotations")
    print("="*80)


def create_train_val_jsons(
    all_vids_json,
    output_train_json,
    output_val_json,
    categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic", "Rainbow Trout"],
    exclude_empty_train=True,
    balance=True,
    num_vids_val=2,
    num_vids_train=None,
    max_vids_per_species_train=None,
    frame_stride=1,  # 1=every frame, 2=every other frame, etc.
    random_seed=None,
    max_frames_train=None,
    max_frames_val=None,
    min_anns_per_video=None,
    max_video_length=None,
    existing_train_json=None  # Path to existing train.json to ensure those videos stay in training
):
    """
    Create train and val YTVIS JSONs from a big all-videos YTVIS JSON.
    - categories_to_include: list of category names to include
    - exclude_empty_train: if True, trim empty frames from train set
    - balance: if True, balance number of videos per class
    - num_vids_val: number of videos per class for val set
    - num_vids_train: number of videos per class for train set (None = as many as possible, only used when balance=True)
    - max_vids_per_species_train: maximum number of videos per species in training set (applies regardless of balance setting)
    - frame_stride: 1 = include every frame, 2 = every 2nd frame, 3 = every 3rd frame, etc.
    - random_seed: if set, ensures reproducible splits
    - max_frames_train: maximum total frames per video in training set - prioritizes segmented frames (None = no limit)
    - max_frames_val: maximum total frames per video in validation set - prioritizes segmented frames (None = no limit)
    - min_anns_per_video: minimum number of annotated frames required per video (None = no filtering)
    - max_video_length: maximum number of frames per video - takes first N frames (None = no cap)
    - existing_train_json: path to existing train.json file - videos in this file will be forced to stay in training set
    """
    if random_seed is not None:
        random.seed(random_seed)
    with open(all_vids_json, 'r') as f:
        data = json.load(f)
    
    # Load existing train JSON if provided
    existing_train_video_names = set()
    if existing_train_json is not None:
        print(f"\nLoading existing train JSON from {existing_train_json}...")
        with open(existing_train_json, 'r') as f:
            existing_train_data = json.load(f)
        
        # Extract video names from existing train set (more stable than IDs)
        existing_train_video_names = set()
        for video in existing_train_data['videos']:
            if video['file_names']:
                video_name = extract_video_name(video['file_names'][0])
                if video_name:
                    existing_train_video_names.add(video_name)
        
        print(f"Found {len(existing_train_video_names)} videos in existing train set")
        
        # Print breakdown by category
        existing_cat_counts = defaultdict(int)
        for ann in existing_train_data['annotations']:
            existing_cat_counts[ann['category_id']] += 1
        
        print("Existing train set breakdown by category:")
        for cat_id, count in existing_cat_counts.items():
            cat_name = next(cat['name'] for cat in existing_train_data['categories'] if cat['id'] == cat_id)
            print(f"  {cat_name}: {count} annotations")
        
        # Show sample video names
        sample_names = list(existing_train_video_names)[:3]
        print(f"Sample existing video names: {sample_names}")
    
    # Cap video lengths if specified
    if max_video_length is not None:
        print(f"\nCapping videos at {max_video_length} frames...")
        capped_count = 0
        
        # Cap videos - take only first max_video_length frames
        for video in data['videos']:
            if video['length'] > max_video_length:
                capped_count += 1
                # Trim file_names to first max_video_length frames
                video['file_names'] = video['file_names'][:max_video_length]
                video['length'] = max_video_length
        
        # Cap annotations - trim segmentations, bboxes, areas to match video length
        for ann in data['annotations']:
            if len(ann['segmentations']) > max_video_length:
                ann['segmentations'] = ann['segmentations'][:max_video_length]
                ann['bboxes'] = ann['bboxes'][:max_video_length]
                ann['areas'] = ann['areas'][:max_video_length]
                ann['length'] = max_video_length
        
        print(f"Capped {capped_count} videos to {max_video_length} frames")
    
    # Map category names to ids
    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    cat_ids = [cat_name_to_id[name] for name in categories_to_include if name in cat_name_to_id]
    
    # Create video name to ID mapping for the current dataset
    current_video_name_to_id = get_video_name_to_id_mapping(data['videos'])
    print(f"Created video name mapping for {len(current_video_name_to_id)} videos in current dataset")
    
    # Convert existing video names to current video IDs
    existing_train_videos = set()
    if existing_train_video_names:
        for video_name in existing_train_video_names:
            if video_name in current_video_name_to_id:
                existing_train_videos.add(current_video_name_to_id[video_name])
        
        print(f"Matched {len(existing_train_videos)} existing videos to current dataset IDs")
        unmatched = len(existing_train_video_names) - len(existing_train_videos)
        if unmatched > 0:
            print(f"WARNING: {unmatched} existing videos not found in current dataset")
    
    # Find videos for each category
    video_to_cats = defaultdict(set)
    for ann in data['annotations']:
        if ann['category_id'] in cat_ids:
            video_to_cats[ann['video_id']].add(ann['category_id'])
    
    # Step 1: Filter out videos with fewer than minimum annotations if specified
    if min_anns_per_video is not None:
        print(f"\nStep 1: Filtering videos with fewer than {min_anns_per_video} annotated frames...")
        
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
    
    # Step 2: Select validation videos (random selection, excluding existing train videos)
    print(f"\nStep 2: Selecting validation videos...")
    val_videos = set()
    val_videos_per_category = {}
    
    for cat_id in cat_ids:
        vids = [vid for vid, cats in video_to_cats.items() if cat_id in cats]
        vids = list(set(vids) - val_videos)  # avoid overlap
        
        # CRITICAL: Exclude videos that are already in the existing train set
        if existing_train_videos:
            vids = [vid for vid in vids if vid not in existing_train_videos]
            print(f"  Category {cat_id}: {len(vids)} NEW videos available for validation (excluding {len(existing_train_videos)} from existing train set)")
        
        # Select more videos initially to account for potential filtering
        # We'll select up to 3x the desired amount to ensure we have enough after filtering
        buffer_multiplier = 3
        initial_num = min(num_vids_val * buffer_multiplier, len(vids))
        
        if initial_num > 0:
            # Always use random sampling for validation set
            selected_vids = random.sample(vids, initial_num)
            val_videos_per_category[cat_id] = selected_vids
            val_videos.update(selected_vids)
        else:
            val_videos_per_category[cat_id] = []
    # Step 3: Select training videos (prioritize NEW videos, fall back to existing if needed)
    print(f"\nStep 3: Selecting training videos...")
    
    # Start with all available videos, excluding validation videos
    available_train_videos = set([vid for vid in video_to_cats if vid not in val_videos])
    
    # CRITICAL: Prioritize NEW videos (exclude existing train videos initially)
    new_train_videos = available_train_videos - existing_train_videos if existing_train_videos else available_train_videos
    
    print(f"  Available videos for training: {len(available_train_videos)}")
    if existing_train_videos:
        print(f"  NEW videos available: {len(new_train_videos)}")
        print(f"  Previously trained videos: {len(existing_train_videos)}")
    
    # Get videos per category for NEW training set
    new_vids_per_cat = {cat_id: [vid for vid in new_train_videos if cat_id in video_to_cats[vid]] for cat_id in cat_ids}
    
    # Get videos per category for existing train set (for fallback)
    existing_vids_per_cat = {}
    if existing_train_videos:
        existing_vids_per_cat = {cat_id: [vid for vid in existing_train_videos if cat_id in video_to_cats[vid]] for cat_id in cat_ids}
    
    # Smart training video selection: prioritize NEW videos, fall back to existing if needed
    final_train_videos = set()
    
    for cat_id in cat_ids:
        cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
        new_vids = new_vids_per_cat[cat_id]
        existing_vids = existing_vids_per_cat.get(cat_id, [])
        
        # Determine how many videos we want for this category
        # For fine-tuning, prioritize NEW videos and use existing only as minimal fallback
        if balance:
            if num_vids_train is None:
                # Use the minimum count across all categories
                min_count = min(len(v) for v in new_vids_per_cat.values())
            else:
                min_count = num_vids_train
            target_count = min_count
        else:
            # Use all available new videos (up to max cap)
            target_count = len(new_vids)
        
        # Apply max_vids_per_species_train cap if specified
        if max_vids_per_species_train is not None:
            target_count = min(target_count, max_vids_per_species_train)
        
        # For fine-tuning: if we have existing videos, be more conservative with target count
        # to prioritize NEW videos
        if existing_train_videos and len(new_vids) > 0:
            # Use at least 80% NEW videos if possible
            min_new_videos = max(1, int(target_count * 0.8))
            if len(new_vids) >= min_new_videos:
                target_count = min(target_count, len(new_vids))
        
        # Try to use NEW videos first
        if len(new_vids) >= target_count:
            # We have enough NEW videos
            selected_vids = random.sample(new_vids, target_count)
            final_train_videos.update(selected_vids)
            print(f"  {cat_name}: using {target_count} NEW videos (no fallback needed)")
        else:
            # Not enough NEW videos, use all NEW videos + minimal fallback to existing
            selected_new_vids = new_vids  # Use all available new videos
            remaining_needed = target_count - len(selected_new_vids)
            
            if remaining_needed > 0 and len(existing_vids) > 0:
                # Fall back to existing videos, but limit fallback to avoid overusing existing videos
                # Only use existing videos if we have very few NEW videos
                max_fallback = max(1, target_count // 4)  # Limit fallback to 25% of target
                fallback_count = min(remaining_needed, len(existing_vids), max_fallback)
                selected_existing_vids = random.sample(existing_vids, fallback_count)
                final_train_videos.update(selected_new_vids + selected_existing_vids)
                print(f"  {cat_name}: using {len(selected_new_vids)} NEW + {fallback_count} existing videos (limited fallback)")
            else:
                # Use only new videos (not enough existing videos for fallback)
                final_train_videos.update(selected_new_vids)
                print(f"  {cat_name}: using {len(selected_new_vids)} NEW videos (insufficient existing for fallback)")
    
    train_videos = final_train_videos
    # Step 4: Create datasets for each stride level
    print(f"\nStep 4: Creating datasets for stride {frame_stride}...")
    
    def apply_stride_to_dataset(videoset, max_frames, trim_empty=False, apply_stride=True):
        # Always start with original full videos and annotations
        vids = [v.copy() for v in data['videos'] if v['id'] in videoset]
        anns = [a.copy() for a in data['annotations'] if a['video_id'] in videoset and a['category_id'] in cat_ids]
        
        # IMPORTANT: Remove orphaned annotations (annotations for videos that don't exist)
        valid_video_ids = set(v['id'] for v in vids)
        anns = [a for a in anns if a['video_id'] in valid_video_ids]
        
        # Apply stride sampling FIRST (1=every frame, 2=every other frame, etc.)
        if apply_stride and frame_stride > 1:
            def stride_frames(seq):
                return seq[::frame_stride]
            # For videos
            for v in vids:
                original_length = v['length']
                v['file_names'] = stride_frames(v['file_names'])
                v['length'] = len(v['file_names'])
                
                # Verify stride application is correct
                # For stride sampling seq[::stride], the expected length is ceil(original_length / stride)
                # This is because we include frame 0 and then every stride-th frame
                expected_length = (original_length + frame_stride - 1) // frame_stride
                if v['length'] != expected_length:
                    print(f"ERROR: Stride {frame_stride} application incorrect for video {v['id']}")
                    print(f"  Original: {original_length} frames")
                    print(f"  Expected after stride: {expected_length} frames")
                    print(f"  Actual after stride: {v['length']} frames")
                    print(f"  This indicates a bug in stride application!")
                
                # Debug first few videos for each stride
                if v['id'] <= 3:
                    print(f"STRIDE CHECK: Video {v['id']} - Original: {original_length}, After stride {frame_stride}: {v['length']} (expected: {expected_length})")
            
            # For annotations
            for ann in anns:
                ann['segmentations'] = stride_frames(ann['segmentations'])
                ann['bboxes'] = stride_frames(ann['bboxes'])
                ann['areas'] = stride_frames(ann['areas'])
                ann['length'] = len(ann['segmentations'])
        
        # Step 5: Apply frame capping AFTER stride - prioritize segmented frames with natural padding
        if max_frames is not None:
            new_vids = []
            new_anns = []
            
            for v in vids:
                video_anns = [a for a in anns if a['video_id'] == v['id']]
                if not video_anns:
                    continue  # Skip videos with no annotations
                
                # Find all frames with segmentations (non-null) AFTER stride
                segmented_frames = set()
                for ann in video_anns:
                    for idx, seg in enumerate(ann['segmentations']):
                        if seg is not None:
                            segmented_frames.add(idx)
                
                segmented_frames = sorted(segmented_frames)
                if not segmented_frames:
                    continue  # Skip videos with no actual segmentations
                
                total_video_frames = v['length']  # This is now the post-stride length
                
                # Debug frame capping for first few videos
                if v['id'] <= 3 and frame_stride in [2, 3, 6]:
                    print(f"FRAME CAPPING DEBUG: Video {v['id']} (stride {frame_stride})")
                    print(f"  Post-stride length: {total_video_frames}")
                    print(f"  Max frames: {max_frames}")
                    print(f"  Will be capped to: {min(total_video_frames, max_frames)}")
                
                
                if len(segmented_frames) >= max_frames:
                    # Too many segmented frames - randomly choose first or last max_frames
                    if random.random() < 0.5:
                        # Take first max_frames segmented frames
                        selected_frames = segmented_frames[:max_frames]
                    else:
                        # Take last max_frames segmented frames
                        selected_frames = segmented_frames[-max_frames:]
                    keep_idxs = set(selected_frames)
                    
                    
                else:
                    # Fewer segmented frames than max_frames - add natural padding
                    num_segmented = len(segmented_frames)
                    padding_needed = max_frames - num_segmented
                    
                    first_seg_frame = segmented_frames[0]
                    last_seg_frame = segmented_frames[-1]
                    
                    
                    # Randomly distribute padding before and after segmented frames
                    if padding_needed > 0:
                        max_before = min(first_seg_frame, padding_needed)
                        max_after = min(total_video_frames - last_seg_frame - 1, padding_needed)
                        
                        if max_before + max_after <= padding_needed:
                            # Use all available padding
                            before_padding = max_before
                            after_padding = max_after
                        else:
                            # Randomly distribute available padding
                            before_padding = random.randint(0, min(max_before, padding_needed))
                            after_padding = min(max_after, padding_needed - before_padding)
                        
                        # Create frame range including padding
                        start_frame = max(0, first_seg_frame - before_padding)
                        end_frame = min(total_video_frames - 1, last_seg_frame + after_padding)
                        
                        keep_idxs = set(range(start_frame, end_frame + 1))
                        
                    else:
                        # No padding needed, just keep segmented frames
                        keep_idxs = set(segmented_frames)
                        
                
                # Apply the frame selection
                v_new = copy.deepcopy(v)
                v_new['file_names'] = [f for i, f in enumerate(v['file_names']) if i in keep_idxs]
                v_new['length'] = len(v_new['file_names'])
                new_vids.append(v_new)
                
                # Update annotations to match the selected frames
                for ann in video_anns:
                    ann_new = copy.deepcopy(ann)
                    ann_new['segmentations'] = [s for i, s in enumerate(ann['segmentations']) if i in keep_idxs]
                    ann_new['bboxes'] = [b for i, b in enumerate(ann['bboxes']) if i in keep_idxs]
                    ann_new['areas'] = [a for i, a in enumerate(ann['areas']) if i in keep_idxs]
                    ann_new['length'] = len(ann_new['segmentations'])
                    new_anns.append(ann_new)
            
            vids = new_vids
            anns = new_anns
        
        # Optionally trim empty frames (for train) - this removes frames with NO annotations
        # This is ADDITIONAL to max_anns and only removes completely empty frames
        if trim_empty:
            # For each video, find frames with at least one annotation
            vid_to_frames_with_ann = defaultdict(set)
            for ann in anns:
                for idx, seg in enumerate(ann['segmentations']):
                    if seg is not None:
                        vid_to_frames_with_ann[ann['video_id']].add(idx)
            
            # Trim frames for each video (remove frames with no annotations)
            new_vids = []
            new_anns = []
            for v in vids:
                frames_with_ann = sorted(vid_to_frames_with_ann[v['id']])
                if not frames_with_ann:
                    continue  # Skip videos with no annotated frames
                
                # Keep only frames that have annotations
                keep_idxs = set(frames_with_ann)
                v_new = copy.deepcopy(v)
                v_new['file_names'] = [f for i, f in enumerate(v['file_names']) if i in keep_idxs]
                v_new['length'] = len(v_new['file_names'])
                new_vids.append(v_new)
                
                # Update annotations to match the trimmed frames
                for ann in [a for a in anns if a['video_id'] == v['id']]:
                    ann_new = copy.deepcopy(ann)
                    ann_new['segmentations'] = [s for i, s in enumerate(ann['segmentations']) if i in keep_idxs]
                    ann_new['bboxes'] = [b for i, b in enumerate(ann['bboxes']) if i in keep_idxs]
                    ann_new['areas'] = [a for i, a in enumerate(ann['areas']) if i in keep_idxs]
                    ann_new['length'] = len(ann_new['segmentations'])
                    new_anns.append(ann_new)
            vids = new_vids
            anns = new_anns
        
        return vids, anns
    
    # Skip first pass - we'll do all processing in the re-filtering step
    
    # After filtering, ensure we have the correct number of validation videos per category
    # Some videos might have been filtered out due to insufficient data
    print(f"\nAdjusting validation set to ensure {num_vids_val} videos per category...")
    
    # First, let's get ALL available videos per category (both train and val)
    all_vids_per_cat = defaultdict(list)
    for video_id, cats in video_to_cats.items():
        for cat_id in cats:
            if cat_id in cat_ids:
                all_vids_per_cat[cat_id].append(video_id)
    
    # Remove duplicates
    for cat_id in all_vids_per_cat:
        all_vids_per_cat[cat_id] = list(set(all_vids_per_cat[cat_id]))
    
    # CRITICAL: Exclude existing train videos from validation selection
    if existing_train_videos:
        for cat_id in all_vids_per_cat:
            original_count = len(all_vids_per_cat[cat_id])
            all_vids_per_cat[cat_id] = [vid for vid in all_vids_per_cat[cat_id] if vid not in existing_train_videos]
            excluded_count = original_count - len(all_vids_per_cat[cat_id])
            if excluded_count > 0:
                cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
                print(f"  {cat_name}: excluded {excluded_count} videos already in existing train set from validation")
    
    # Check if we have enough videos per category
    for cat_id in cat_ids:
        cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
        total_available = len(all_vids_per_cat[cat_id])
        if total_available < num_vids_val:
            print(f"WARNING: Only {total_available} NEW {cat_name} videos available, but {num_vids_val} needed for validation!")
            print(f"This will cause the validation set to be incomplete for {cat_name}.")
    
    # Now select validation videos first, ensuring we get exactly num_vids_val per category
    # Add a buffer to account for videos that might be filtered out during stride application
    buffer_factor = 1.5  # Select 50% more videos than needed to ensure we have enough after filtering
    final_val_videos = set()
    
    for cat_id in cat_ids:
        cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
        available_vids = all_vids_per_cat[cat_id]
        
        # Select with buffer to account for potential filtering
        target_count = int(num_vids_val * buffer_factor)
        if len(available_vids) >= target_count:
            selected_val_vids = random.sample(available_vids, target_count)
            final_val_videos.update(selected_val_vids)
            print(f"  {cat_name}: selected {target_count} videos for validation (with buffer)")
        elif len(available_vids) >= num_vids_val:
            # Use all available videos if we have at least num_vids_val
            selected_val_vids = available_vids
            final_val_videos.update(selected_val_vids)
            print(f"  {cat_name}: selected {len(selected_val_vids)} videos for validation (all available)")
        else:
            # If we don't have enough, use all available (this should be rare)
            final_val_videos.update(available_vids)
            print(f"  {cat_name}: WARNING - only {len(available_vids)} videos available, using all for validation")
    
    # All remaining videos go to training
    final_train_videos = set()
    for video_id, cats in video_to_cats.items():
        if video_id not in final_val_videos:
            final_train_videos.add(video_id)
    
    # Apply max_vids_per_species_train cap to the final training set
    if max_vids_per_species_train is not None:
        print(f"\nApplying species cap of {max_vids_per_species_train} videos per species to final training set...")
        capped_train_videos = set()
        
        # Get videos per category for the final training set
        final_train_vids_per_cat = defaultdict(list)
        for vid in final_train_videos:
            for cat_id in cat_ids:
                if vid in video_to_cats and cat_id in video_to_cats[vid]:
                    final_train_vids_per_cat[cat_id].append(vid)
        
        for cat_id, vids in final_train_vids_per_cat.items():
            cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
            if len(vids) > max_vids_per_species_train:
                # Separate NEW videos from existing videos
                new_vids_in_final = [vid for vid in vids if vid not in existing_train_videos]
                existing_vids_in_final = [vid for vid in vids if vid in existing_train_videos]
                
                # Prioritize NEW videos
                if len(new_vids_in_final) >= max_vids_per_species_train:
                    # We have enough NEW videos, use only NEW videos
                    selected = random.sample(new_vids_in_final, max_vids_per_species_train)
                    capped_train_videos.update(selected)
                    print(f"  {cat_name}: reduced from {len(vids)} to {max_vids_per_species_train} videos (all NEW)")
                else:
                    # Use all NEW videos + some existing videos
                    selected = new_vids_in_final.copy()
                    remaining_needed = max_vids_per_species_train - len(new_vids_in_final)
                    if remaining_needed > 0 and len(existing_vids_in_final) > 0:
                        additional_existing = random.sample(existing_vids_in_final, min(remaining_needed, len(existing_vids_in_final)))
                        selected.extend(additional_existing)
                    capped_train_videos.update(selected)
                    print(f"  {cat_name}: reduced from {len(vids)} to {len(selected)} videos ({len(new_vids_in_final)} NEW + {len(selected)-len(new_vids_in_final)} existing)")
            else:
                capped_train_videos.update(vids)
                print(f"  {cat_name}: kept all {len(vids)} videos (under cap)")
        
        final_train_videos = capped_train_videos
    
    # Re-filter with the final video sets (apply stride here)
    print("Re-filtering with adjusted video sets...")
    train_vids, train_anns = apply_stride_to_dataset(final_train_videos, max_frames=max_frames_train, trim_empty=exclude_empty_train, apply_stride=True)
    val_vids, val_anns = apply_stride_to_dataset(final_val_videos, max_frames=max_frames_val, trim_empty=False, apply_stride=True)
    
    # CRITICAL FIX: Ensure we have exactly num_vids_val videos per category in validation set
    print(f"\nAdjusting validation set to exactly {num_vids_val} videos per category...")
    val_video_ids = set(v['id'] for v in val_vids)
    val_videos_per_cat = defaultdict(list)
    
    for video in val_vids:
        # Find annotations for this video
        video_anns = [ann for ann in val_anns if ann['video_id'] == video['id']]
        if video_anns:
            cat_id = video_anns[0]['category_id']
            val_videos_per_cat[cat_id].append(video['id'])
    
    # Adjust each category to have exactly num_vids_val videos
    videos_to_keep = set()
    for cat_id in cat_ids:
        cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
        current_videos = val_videos_per_cat[cat_id]
        current_count = len(current_videos)
        
        if current_count > num_vids_val:
            # Too many videos, randomly select num_vids_val
            selected = random.sample(current_videos, num_vids_val)
            videos_to_keep.update(selected)
            print(f"  {cat_name}: trimmed from {current_count} to {num_vids_val} videos")
        elif current_count == num_vids_val:
            # Perfect count
            videos_to_keep.update(current_videos)
            print(f"  {cat_name}: kept all {current_count} videos")
        else:
            # Not enough videos - keep all and try to find more
            videos_to_keep.update(current_videos)
            print(f"  {cat_name}: WARNING - only {current_count} videos available, need {num_vids_val}")
    
    # Filter val_vids and val_anns to only keep the selected videos
    val_vids = [v for v in val_vids if v['id'] in videos_to_keep]
    val_anns = [a for a in val_anns if a['video_id'] in videos_to_keep]
    
    print("Validation set adjustment complete.")
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


def select_videos_once(all_vids_json, categories_to_include, num_vids_val, balance, num_vids_train, max_vids_per_species_train, min_anns_per_video, random_seed=43, existing_train_json=None):
    """
    Select train and validation videos ONCE, then return the video IDs.
    This ensures consistent video selection across all stride datasets.
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    with open(all_vids_json, 'r') as f:
        data = json.load(f)
    
    # Load existing train JSON if provided
    existing_train_video_names = set()
    if existing_train_json is not None:
        print(f"\nLoading existing train JSON from {existing_train_json}...")
        with open(existing_train_json, 'r') as f:
            existing_train_data = json.load(f)
        
        # Extract video names from existing train set (more stable than IDs)
        for video in existing_train_data['videos']:
            if video['file_names']:
                video_name = extract_video_name(video['file_names'][0])
                if video_name:
                    existing_train_video_names.add(video_name)
        
        print(f"Found {len(existing_train_video_names)} videos in existing train set")
    
    # Map category names to ids
    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    cat_ids = [cat_name_to_id[name] for name in categories_to_include if name in cat_name_to_id]
    
    # Create video name to ID mapping for the current dataset
    current_video_name_to_id = get_video_name_to_id_mapping(data['videos'])
    print(f"Created video name mapping for {len(current_video_name_to_id)} videos in current dataset")
    
    # Convert existing video names to current video IDs
    existing_train_videos = set()
    if existing_train_video_names:
        for video_name in existing_train_video_names:
            if video_name in current_video_name_to_id:
                existing_train_videos.add(current_video_name_to_id[video_name])
        
        print(f"Matched {len(existing_train_videos)} existing videos to current dataset IDs")
        unmatched = len(existing_train_video_names) - len(existing_train_videos)
        if unmatched > 0:
            print(f"WARNING: {unmatched} existing videos not found in current dataset")
    
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
            print(f"Removing {len(videos_to_remove)} videos with insufficient annotations")
            # Remove videos from video_to_cats
            for video_id in videos_to_remove:
                if video_id in video_to_cats:
                    del video_to_cats[video_id]
    
    # Select validation videos (random selection)
    print(f"\nSelecting validation videos...")
    val_videos = set()
    val_videos_per_category = {}
    
    for cat_id in cat_ids:
        vids = [vid for vid, cats in video_to_cats.items() if cat_id in cats]
        vids = list(set(vids) - val_videos)  # avoid overlap
        
        # CRITICAL: Exclude videos that are already in the existing train set
        if existing_train_videos:
            vids = [vid for vid in vids if vid not in existing_train_videos]
            print(f"  Category {cat_id}: {len(vids)} videos available for validation (excluding {len(existing_train_videos)} from existing train set)")
        
        # Select more videos initially to account for potential filtering
        buffer_multiplier = 3
        initial_num = min(num_vids_val * buffer_multiplier, len(vids))
        
        if initial_num > 0:
            selected_vids = random.sample(vids, initial_num)
            val_videos_per_category[cat_id] = selected_vids
            val_videos.update(selected_vids)
        else:
            val_videos_per_category[cat_id] = []
    
    # Select training videos (prioritize NEW videos, fall back to existing if needed)
    print(f"\nSelecting training videos...")
    
    # Start with all available videos, excluding validation videos
    available_train_videos = set([vid for vid in video_to_cats if vid not in val_videos])
    
    # CRITICAL: Prioritize NEW videos (exclude existing train videos initially)
    new_train_videos = available_train_videos - existing_train_videos if existing_train_videos else available_train_videos
    
    print(f"  Available videos for training: {len(available_train_videos)}")
    if existing_train_videos:
        print(f"  NEW videos available: {len(new_train_videos)}")
        print(f"  Previously trained videos: {len(existing_train_videos)}")
    
    # Get videos per category for NEW training set
    new_vids_per_cat = {cat_id: [vid for vid in new_train_videos if cat_id in video_to_cats[vid]] for cat_id in cat_ids}
    
    # Get videos per category for existing train set (for fallback)
    existing_vids_per_cat = {}
    if existing_train_videos:
        existing_vids_per_cat = {cat_id: [vid for vid in existing_train_videos if cat_id in video_to_cats[vid]] for cat_id in cat_ids}
    
    if balance:
        # For each class, sample up to num_vids_train (or min count if None)
        if num_vids_train is None:
            min_count = min(len(v) for v in new_vids_per_cat.values())
        else:
            min_count = num_vids_train
        balanced_train_videos = set()
        for cat_id, vids in new_vids_per_cat.items():
            n = min(min_count, len(vids))
            # Also apply max_vids_per_species_train cap if specified
            if max_vids_per_species_train is not None:
                n = min(n, max_vids_per_species_train)
            if n > 0:
                balanced_train_videos.update(random.sample(vids, n))
        train_videos = balanced_train_videos
    else:
        # Apply max_vids_per_species_train cap even when not balancing
        if max_vids_per_species_train is not None:
            capped_train_videos = set()
            for cat_id, vids in new_vids_per_cat.items():
                n = min(max_vids_per_species_train, len(vids))
                if n > 0:
                    capped_train_videos.update(random.sample(vids, n))
                else:
                    capped_train_videos.update(vids)
            train_videos = capped_train_videos
    
    # Final adjustment to ensure correct number of validation videos per category
    print(f"\nAdjusting validation set to ensure {num_vids_val} videos per category...")
    
    # First, let's get ALL available videos per category (both train and val)
    all_vids_per_cat = defaultdict(list)
    for video_id, cats in video_to_cats.items():
        for cat_id in cats:
            if cat_id in cat_ids:
                all_vids_per_cat[cat_id].append(video_id)
    
    # Remove duplicates
    for cat_id in all_vids_per_cat:
        all_vids_per_cat[cat_id] = list(set(all_vids_per_cat[cat_id]))
    
    # CRITICAL: Exclude existing train videos from validation selection
    if existing_train_videos:
        for cat_id in all_vids_per_cat:
            original_count = len(all_vids_per_cat[cat_id])
            all_vids_per_cat[cat_id] = [vid for vid in all_vids_per_cat[cat_id] if vid not in existing_train_videos]
            excluded_count = original_count - len(all_vids_per_cat[cat_id])
            if excluded_count > 0:
                cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
                print(f"  {cat_name}: excluded {excluded_count} videos already in existing train set")
    
    # Check if we have enough videos per category
    for cat_id in cat_ids:
        cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
        total_available = len(all_vids_per_cat[cat_id])
        if total_available < num_vids_val:
            print(f"WARNING: Only {total_available} NEW {cat_name} videos available, but {num_vids_val} needed for validation!")
            print(f"This will cause the validation set to be incomplete for {cat_name}.")
    
    # Now select validation videos first, ensuring we get exactly num_vids_val per category
    # Add a buffer to account for videos that might be filtered out during stride application
    buffer_factor = 1.5  # Select 50% more videos than needed to ensure we have enough after filtering
    final_val_videos = set()
    
    for cat_id in cat_ids:
        cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
        available_vids = all_vids_per_cat[cat_id]
        
        # Select with buffer to account for potential filtering
        target_count = int(num_vids_val * buffer_factor)
        if len(available_vids) >= target_count:
            selected_val_vids = random.sample(available_vids, target_count)
            final_val_videos.update(selected_val_vids)
            print(f"  {cat_name}: selected {target_count} videos for validation (with buffer)")
        elif len(available_vids) >= num_vids_val:
            # Use all available videos if we have at least num_vids_val
            selected_val_vids = available_vids
            final_val_videos.update(selected_val_vids)
            print(f"  {cat_name}: selected {len(selected_val_vids)} videos for validation (all available)")
        else:
            # If we don't have enough, use all available (this should be rare)
            final_val_videos.update(available_vids)
            print(f"  {cat_name}: WARNING - only {len(available_vids)} videos available, using all for validation")
    
    # All remaining videos go to training
    final_train_videos = set()
    for video_id, cats in video_to_cats.items():
        if video_id not in final_val_videos:
            final_train_videos.add(video_id)
    
    # Apply max_vids_per_species_train cap to the final training set
    if max_vids_per_species_train is not None:
        print(f"\nApplying species cap of {max_vids_per_species_train} videos per species to final training set...")
        capped_train_videos = set()
        
        # Get videos per category for the final training set
        final_train_vids_per_cat = defaultdict(list)
        for vid in final_train_videos:
            for cat_id in cat_ids:
                if vid in video_to_cats and cat_id in video_to_cats[vid]:
                    final_train_vids_per_cat[cat_id].append(vid)
        
        for cat_id, vids in final_train_vids_per_cat.items():
            cat_name = next(cat['name'] for cat in data['categories'] if cat['id'] == cat_id)
            if len(vids) > max_vids_per_species_train:
                # Separate NEW videos from existing videos
                new_vids_in_final = [vid for vid in vids if vid not in existing_train_videos]
                existing_vids_in_final = [vid for vid in vids if vid in existing_train_videos]
                
                # Prioritize NEW videos
                if len(new_vids_in_final) >= max_vids_per_species_train:
                    # We have enough NEW videos, use only NEW videos
                    selected = random.sample(new_vids_in_final, max_vids_per_species_train)
                    capped_train_videos.update(selected)
                    print(f"  {cat_name}: reduced from {len(vids)} to {max_vids_per_species_train} videos (all NEW)")
                else:
                    # Use all NEW videos + some existing videos
                    selected = new_vids_in_final.copy()
                    remaining_needed = max_vids_per_species_train - len(new_vids_in_final)
                    if remaining_needed > 0 and len(existing_vids_in_final) > 0:
                        additional_existing = random.sample(existing_vids_in_final, min(remaining_needed, len(existing_vids_in_final)))
                        selected.extend(additional_existing)
                    capped_train_videos.update(selected)
                    print(f"  {cat_name}: reduced from {len(vids)} to {len(selected)} videos ({len(new_vids_in_final)} NEW + {len(selected)-len(new_vids_in_final)} existing)")
            else:
                capped_train_videos.update(vids)
                print(f"  {cat_name}: kept all {len(vids)} videos (under cap)")
        
        final_train_videos = capped_train_videos
    
    print(f"\nFinal video selection:")
    print(f"  Training videos: {len(final_train_videos)}")
    print(f"  Validation videos: {len(final_val_videos)}")
    
    return final_train_videos, final_val_videos, data


def create_stride_dataset_from_selected_videos(
    selected_train_videos, 
    selected_val_videos, 
    original_data,
    frame_stride,
    output_train_json,
    output_val_json,
    categories_to_include,
    exclude_empty_train=False,
    max_frames_train=None,
    max_frames_val=250
):
    """
    Create train and val datasets with specific stride from pre-selected videos.
    This ensures the same videos are used across all stride datasets.
    """
    # Map category names to ids
    cat_name_to_id = {cat['name']: cat['id'] for cat in original_data['categories']}
    cat_ids = [cat_name_to_id[name] for name in categories_to_include if name in cat_name_to_id]
    
    def apply_stride_to_dataset(videoset, max_frames, trim_empty=False, apply_stride=True):
        # Always start with original full videos and annotations
        vids = [v.copy() for v in original_data['videos'] if v['id'] in videoset]
        anns = [a.copy() for a in original_data['annotations'] if a['video_id'] in videoset and a['category_id'] in cat_ids]
        
        # IMPORTANT: Remove orphaned annotations (annotations for videos that don't exist)
        valid_video_ids = set(v['id'] for v in vids)
        anns = [a for a in anns if a['video_id'] in valid_video_ids]
        
        # Apply stride sampling FIRST (1=every frame, 2=every other frame, etc.)
        if apply_stride and frame_stride > 1:
            def stride_frames(seq):
                return seq[::frame_stride]
            # For videos
            for v in vids:
                original_length = v['length']
                v['file_names'] = stride_frames(v['file_names'])
                v['length'] = len(v['file_names'])
            
            # For annotations
            for ann in anns:
                ann['segmentations'] = stride_frames(ann['segmentations'])
                ann['bboxes'] = stride_frames(ann['bboxes'])
                ann['areas'] = stride_frames(ann['areas'])
                ann['length'] = len(ann['segmentations'])
        
        # Apply frame capping AFTER stride - prioritize segmented frames with natural padding
        if max_frames is not None:
            new_vids = []
            new_anns = []
            
            for v in vids:
                video_anns = [a for a in anns if a['video_id'] == v['id']]
                if not video_anns:
                    continue  # Skip videos with no annotations
                
                # Find all frames with segmentations (non-null) AFTER stride
                segmented_frames = set()
                for ann in video_anns:
                    for idx, seg in enumerate(ann['segmentations']):
                        if seg is not None:
                            segmented_frames.add(idx)
                
                segmented_frames = sorted(segmented_frames)
                if not segmented_frames:
                    continue  # Skip videos with no actual segmentations
                
                total_video_frames = v['length']  # This is now the post-stride length
                
                if len(segmented_frames) >= max_frames:
                    # Too many segmented frames - randomly choose first or last max_frames
                    if random.random() < 0.5:
                        # Take first max_frames segmented frames
                        selected_frames = segmented_frames[:max_frames]
                    else:
                        # Take last max_frames segmented frames
                        selected_frames = segmented_frames[-max_frames:]
                    keep_idxs = set(selected_frames)
                else:
                    # Fewer segmented frames than max_frames - add natural padding
                    num_segmented = len(segmented_frames)
                    padding_needed = max_frames - num_segmented
                    
                    first_seg_frame = segmented_frames[0]
                    last_seg_frame = segmented_frames[-1]
                    
                    # Randomly distribute padding before and after segmented frames
                    if padding_needed > 0:
                        max_before = min(first_seg_frame, padding_needed)
                        max_after = min(total_video_frames - last_seg_frame - 1, padding_needed)
                        
                        if max_before + max_after <= padding_needed:
                            # Use all available padding
                            before_padding = max_before
                            after_padding = max_after
                        else:
                            # Randomly distribute available padding
                            before_padding = random.randint(0, min(max_before, padding_needed))
                            after_padding = min(max_after, padding_needed - before_padding)
                        
                        # Create frame range including padding
                        start_frame = max(0, first_seg_frame - before_padding)
                        end_frame = min(total_video_frames - 1, last_seg_frame + after_padding)
                        
                        keep_idxs = set(range(start_frame, end_frame + 1))
                    else:
                        # No padding needed, just keep segmented frames
                        keep_idxs = set(segmented_frames)
                
                # Apply the frame selection
                v_new = copy.deepcopy(v)
                v_new['file_names'] = [f for i, f in enumerate(v['file_names']) if i in keep_idxs]
                v_new['length'] = len(v_new['file_names'])
                new_vids.append(v_new)
                
                # Update annotations to match the selected frames
                for ann in video_anns:
                    ann_new = copy.deepcopy(ann)
                    ann_new['segmentations'] = [s for i, s in enumerate(ann['segmentations']) if i in keep_idxs]
                    ann_new['bboxes'] = [b for i, b in enumerate(ann['bboxes']) if i in keep_idxs]
                    ann_new['areas'] = [a for i, a in enumerate(ann['areas']) if i in keep_idxs]
                    ann_new['length'] = len(ann_new['segmentations'])
                    new_anns.append(ann_new)
            
            vids = new_vids
            anns = new_anns
        
        # Optionally trim empty frames (for train) - this removes frames with NO annotations
        if trim_empty:
            # For each video, find frames with at least one annotation
            vid_to_frames_with_ann = defaultdict(set)
            for ann in anns:
                for idx, seg in enumerate(ann['segmentations']):
                    if seg is not None:
                        vid_to_frames_with_ann[ann['video_id']].add(idx)
            
            # Trim frames for each video (remove frames with no annotations)
            new_vids = []
            new_anns = []
            for v in vids:
                frames_with_ann = sorted(vid_to_frames_with_ann[v['id']])
                if not frames_with_ann:
                    continue  # Skip videos with no annotated frames
                
                # Keep only frames that have annotations
                keep_idxs = set(frames_with_ann)
                v_new = copy.deepcopy(v)
                v_new['file_names'] = [f for i, f in enumerate(v['file_names']) if i in keep_idxs]
                v_new['length'] = len(v_new['file_names'])
                new_vids.append(v_new)
                
                # Update annotations to match the trimmed frames
                for ann in [a for a in anns if a['video_id'] == v['id']]:
                    ann_new = copy.deepcopy(ann)
                    ann_new['segmentations'] = [s for i, s in enumerate(ann['segmentations']) if i in keep_idxs]
                    ann_new['bboxes'] = [b for i, b in enumerate(ann['bboxes']) if i in keep_idxs]
                    ann_new['areas'] = [a for i, a in enumerate(ann['areas']) if i in keep_idxs]
                    ann_new['length'] = len(ann_new['segmentations'])
                    new_anns.append(ann_new)
            vids = new_vids
            anns = new_anns
        
        return vids, anns
    
    # Apply stride to the pre-selected videos
    train_vids, train_anns = apply_stride_to_dataset(selected_train_videos, max_frames=max_frames_train, trim_empty=exclude_empty_train, apply_stride=True)
    val_vids, val_anns = apply_stride_to_dataset(selected_val_videos, max_frames=max_frames_val, trim_empty=False, apply_stride=True)
    
    # Build output jsons
    def build_json(vids, anns):
        used_cat_ids = set(a['category_id'] for a in anns)
        cats = [cat for cat in original_data['categories'] if cat['id'] in used_cat_ids]
        return {
            'videos': vids,
            'annotations': anns,
            'categories': cats,
            'info': original_data.get('info', {}),
            'licenses': original_data.get('licenses', [])
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


if __name__ == "__main__":
    # # Set random seed once for reproducible video selection across all stride datasets
    # random.seed(43)
    
    # # STEP 1: Select videos ONCE (this ensures consistency across all strides)
    # print(f"\n{'='*80}")
    # print(f"STEP 1: SELECTING VIDEOS ONCE FOR ALL STRIDE DATASETS")
    # print(f"{'='*80}")
    
    # selected_train_videos, selected_val_videos, original_data = select_videos_once(
    #     all_vids_json="/data/fishway_ytvis/all_videos.json",
    #     categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic", "Rainbow Trout"],
    #     num_vids_val=8,
    #     balance=False,
    #     num_vids_train=None,
    #     max_vids_per_species_train=50,
    #     min_anns_per_video=10,
    #     random_seed=43,
    #     existing_train_json=None  # Set to None for regular dataset creation
    # )
    
    # # STEP 2: Generate datasets for different stride values using the SAME selected videos
    # stride_values = [1, 2, 3, 4, 5, 6]  # 1=every frame, 2=every other frame, etc.
    
    # for stride in stride_values:
    #     print(f"\n{'='*80}")
    #     print(f"STEP 2: GENERATING DATASET WITH STRIDE {stride} (using same videos)")
    #     print(f"{'='*80}")

    #     train_json_path = f"/data/fishway_ytvis/train_stride{stride}.json"
    #     val_json_path = f"/data/fishway_ytvis/val_stride{stride}.json"
        
    #     create_stride_dataset_from_selected_videos(
    #         selected_train_videos=selected_train_videos,
    #         selected_val_videos=selected_val_videos,
    #         original_data=original_data,
    #         frame_stride=stride,
    #         output_train_json=train_json_path,
    #         output_val_json=val_json_path,
    #         categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic", "Rainbow Trout"],
    #         exclude_empty_train=True,
    #         max_frames_train=None,
    #         max_frames_val=None
    #     )

    # ============================================================================
    # FINE-TUNING DATASET CREATION
    # ============================================================================
    # Uncomment the section below to create fine-tuning datasets that exclude previously trained videos
    # This will create finetune_train.json and finetune_val.json with NEW videos only
    
    print(f"\n{'='*80}")
    print(f"FINE-TUNING DATASET CREATION")
    print(f"{'='*80}")
    
    # Create fine-tuning datasets with existing train.json exclusion
    create_train_val_jsons(
        all_vids_json="/data/fishway_ytvis/all_videos.json",
        output_train_json="/data/fishway_ytvis/finetune_train.json",
        output_val_json="/data/fishway_ytvis/finetune_val.json",
        categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic", "Rainbow Trout"],
        exclude_empty_train=True,
        balance=False,  # Don't balance for fine-tuning
        num_vids_val=8,  # Small validation set for fine-tuning
        num_vids_train=None,
        max_vids_per_species_train=40,  # Adjust as needed
        frame_stride=1,
        random_seed=100,
        max_frames_train=300,  # Increased to avoid truncation issues
        max_frames_val=300,    # Increased to avoid truncation issues
        min_anns_per_video=5,  # Reduced to avoid filtering out validation videos
        max_video_length=None,
        existing_train_json="/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/train.json"  # Key parameter!
    ) 