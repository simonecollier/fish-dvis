import json
import random
import copy
from collections import defaultdict
import numpy as np

def print_dataset_statistics(ytvis_data, dataset_name):
    """
    Print detailed statistics about the dataset including:
    - Number of videos per category
    - Number of annotated frames per category
    - Frame statistics (mean, median, min, max) per category
    """
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} DATASET STATISTICS")
    print(f"{'='*80}")
    
    # Create category ID to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in ytvis_data['categories']}
    
    # Initialize statistics containers
    cat_video_count = defaultdict(int)
    cat_annotated_frames = defaultdict(int)
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
        
        # Count annotated frames per category
        annotated_frames = sum(1 for seg in ann['segmentations'] if seg is not None)
        cat_annotated_frames[cat_id] += annotated_frames
        
        # Collect annotated frames per video for this category
        cat_annotated_frames_per_video[cat_id].append(annotated_frames)
    
    # Calculate video counts
    for cat_id, video_ids in cat_video_ids.items():
        cat_video_count[cat_id] = len(video_ids)
    
    # Print statistics by category
    print(f"\n{'Category':<20} {'Videos':<8} {'Annotated Frames':<18} {'Frame Stats (mean/median/min/max)':<35}")
    print("-" * 85)
    
    for cat_id in sorted(cat_video_count.keys()):
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        video_count = cat_video_count[cat_id]
        annotated_frames = cat_annotated_frames[cat_id]
        annotated_frames_per_video = cat_annotated_frames_per_video[cat_id]
        
        if annotated_frames_per_video:
            mean_frames = np.mean(annotated_frames_per_video)
            median_frames = np.median(annotated_frames_per_video)
            min_frames = np.min(annotated_frames_per_video)
            max_frames = np.max(annotated_frames_per_video)
            frame_stats = f"{mean_frames:.1f}/{median_frames:.1f}/{min_frames}/{max_frames}"
        else:
            frame_stats = "N/A"
        
        print(f"{cat_name:<20} {video_count:<8} {annotated_frames:<18} {frame_stats:<35}")
    
    # Print overall statistics
    print("\n" + "-" * 85)
    total_videos = len(ytvis_data['videos'])
    total_annotations = len(ytvis_data['annotations'])
    total_annotated_frames = sum(cat_annotated_frames.values())
    all_annotated_frames_per_video = []
    for frames_list in cat_annotated_frames_per_video.values():
        all_annotated_frames_per_video.extend(frames_list)
    
    if all_annotated_frames_per_video:
        overall_mean = np.mean(all_annotated_frames_per_video)
        overall_median = np.median(all_annotated_frames_per_video)
        overall_min = np.min(all_annotated_frames_per_video)
        overall_max = np.max(all_annotated_frames_per_video)
        overall_stats = f"{overall_mean:.1f}/{overall_median:.1f}/{overall_min}/{overall_max}"
    else:
        overall_stats = "N/A"
    
    print(f"{'TOTAL':<20} {total_videos:<8} {total_annotated_frames:<18} {overall_stats:<35}")
    print(f"Total annotations: {total_annotations}")
    print(f"Total categories: {len(ytvis_data['categories'])}")
    print(f"{'='*80}")


def create_train_val_jsons(
    all_vids_json,
    output_train_json,
    output_val_json,
    categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic", "Rainbow Trout"],
    exclude_empty_train=True,
    balance=True,
    num_vids_val=2,
    num_vids_train=None,
    frame_skip=0,
    random_seed=None,
    max_frames_train=75,
    max_frames_val=45
):
    """
    Create train and val YTVIS JSONs from a big all-videos YTVIS JSON.
    - categories_to_include: list of category names to include
    - exclude_empty_train: if True, trim empty frames from train set
    - balance: if True, balance number of videos per class
    - num_vids_val: number of videos per class for val set
    - num_vids_train: number of videos per class for train set (None = as many as possible)
    - frame_skip: 0 = include every frame, 1 = every 2nd frame, 2 = every 3rd frame, etc.
    - random_seed: if set, ensures reproducible splits
    - max_frames_train: maximum number of frames to keep per video in training set (default 75)
    - max_frames_val: maximum number of frames to keep per video in validation set (default 45)
    """
    if random_seed is not None:
        random.seed(random_seed)
    with open(all_vids_json, 'r') as f:
        data = json.load(f)
    # Map category names to ids
    cat_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    cat_ids = [cat_name_to_id[name] for name in categories_to_include if name in cat_name_to_id]
    # Find videos for each category
    video_to_cats = defaultdict(set)
    for ann in data['annotations']:
        if ann['category_id'] in cat_ids:
            video_to_cats[ann['video_id']].add(ann['category_id'])
    # Build val set
    val_videos = set()
    for cat_id in cat_ids:
        vids = [vid for vid, cats in video_to_cats.items() if cat_id in cats]
        vids = list(set(vids) - val_videos)  # avoid overlap
        num = min(num_vids_val, len(vids))
        if balance:
            vids = random.sample(vids, num) if num > 0 else []
        else:
            vids = vids[:num]
        val_videos.update(vids)
    # Build train set
    train_videos = set([vid for vid in video_to_cats if vid not in val_videos])
    if balance:
        # For each class, sample up to num_vids_train (or min count if None)
        vids_per_cat = {cat_id: [vid for vid in train_videos if cat_id in video_to_cats[vid]] for cat_id in cat_ids}
        if num_vids_train is None:
            min_count = min(len(v) for v in vids_per_cat.values())
        else:
            min_count = num_vids_train
        balanced_train_videos = set()
        for cat_id, vids in vids_per_cat.items():
            n = min(min_count, len(vids))
            if n > 0:
                balanced_train_videos.update(random.sample(vids, n))
        train_videos = balanced_train_videos
    # Filter videos/annotations for train/val
    def filter_json(videoset, trim_empty, max_frames):
        vids = [v for v in data['videos'] if v['id'] in videoset]
        anns = [a for a in data['annotations'] if a['video_id'] in videoset and a['category_id'] in cat_ids]
        # Optionally trim empty frames (for train)
        if trim_empty:
            # For each video, find frames with at least one annotation
            vid_to_frames_with_ann = defaultdict(set)
            for ann in anns:
                for idx, seg in enumerate(ann['segmentations']):
                    if seg is not None:
                        vid_to_frames_with_ann[ann['video_id']].add(idx)
            # Trim frames for each video
            new_vids = []
            new_anns = []
            for v in vids:
                frames_with_ann = sorted(vid_to_frames_with_ann[v['id']])
                if not frames_with_ann:
                    continue
                # Cap the number of frames if needed
                if max_frames is not None and len(frames_with_ann) > max_frames:
                    # Center the selection around the middle of the video
                    mid = len(frames_with_ann) // 2
                    half = max_frames // 2
                    if max_frames % 2 == 0:
                        selected = frames_with_ann[mid-half:mid+half]
                    else:
                        selected = frames_with_ann[mid-half:mid+half+1]
                    frames_with_ann = selected
                first = min(frames_with_ann)
                last = max(frames_with_ann)
                # Only keep the selected frames
                keep_idxs = set(frames_with_ann)
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


if __name__ == "__main__":
    create_train_val_jsons(
        all_vids_json="/data/fishway_ytvis/all_videos.json",
        output_train_json="/data/fishway_ytvis/train.json",
        output_val_json="/data/fishway_ytvis/val.json",
        categories_to_include=["Chinook", "Coho", "Brown Trout", "Atlantic", "Rainbow Trout"],
        exclude_empty_train=True,
        balance=False,
        num_vids_val=3,
        num_vids_train=None,
        frame_skip=0,
        max_frames_train=75,  # 5 windows of 15 frames for training
        max_frames_val=45     # 3 windows of 15 frames for validation
    ) 