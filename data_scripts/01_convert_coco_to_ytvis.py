import os
import json
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from pycocotools import mask as mask_utils
import numpy as np

def decode_rle(rle):
    """
    Decode RLE mask encoding, handling empty masks.
    """
    if not isinstance(rle, dict) or not rle:
        return np.array([])
    
    # Check for empty counts
    if not rle.get('counts', []):
        return np.array([])
    
    decoded_mask = mask_utils.decode({
        'size': rle['size'],
        'counts': bytes(rle['counts']) if isinstance(rle['counts'][0], int) else rle['counts']
    })
    return decoded_mask

def convert_all_segmentations_to_compressed_rle(ytvis_json):
    # Create video ID to video mapping for debugging
    videos = {v['id']: v for v in ytvis_json.get('videos', [])}
    
    for ann in ytvis_json["annotations"]:
        video_id = ann.get('video_id', 'unknown')
        video_info = videos.get(video_id, {})
        video_folder = video_info.get('file_names', ['unknown'])[0].split('/')[0] if video_info.get('file_names') else 'unknown'
        
        segs = ann.get("segmentations", [])
        for i, seg in enumerate(segs):
            if isinstance(seg, dict) and isinstance(seg.get("counts", None), list):
                try:
                    # Re-encode as compressed RLE
                    mask = decode_rle(seg)
                    compressed = mask_utils.encode(np.asfortranarray(mask))
                    
                    # Check if encoding failed
                    if compressed is None:
                        print(f"Warning: RLE encoding failed for annotation {ann['id']}, frame {i}")
                        print(f"Video ID: {video_id}, Video folder: {video_folder}")
                        continue
                    
                    compressed["counts"] = compressed["counts"].decode("ascii")
                    segs[i] = compressed
                except Exception as e:
                    print(f"Error encoding RLE for annotation {ann['id']}, frame {i}: {e}")
                    print(f"Video ID: {video_id}, Video folder: {video_folder}")
                    continue
        ann["segmentations"] = segs
    return ytvis_json

def load_existing_ytvis(json_path):
    """Load existing YTVIS JSON if it exists, otherwise return None."""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load existing JSON from {json_path}")
            return None
    return None

def get_existing_video_names(ytvis_data):
    """Extract video folder names from existing YTVIS data."""
    if not ytvis_data or 'videos' not in ytvis_data:
        return set()
    
    existing_videos = set()
    for video in ytvis_data['videos']:
        if 'file_names' in video and video['file_names']:
            # Extract video folder name from first file path
            first_file = video['file_names'][0]
            if '/' in first_file:
                video_folder = first_file.split('/')[0]
                existing_videos.add(video_folder)
    return existing_videos

def check_video_images_exist(image_out_dir, video_folder):
    """Check if all images for a video already exist in the output directory."""
    video_dir = os.path.join(image_out_dir, video_folder)
    if not os.path.exists(video_dir):
        return False
    
    # Check if directory has any image files
    image_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    return len(image_files) > 0

def filter_videos_without_annotations(ytvis, image_out_dir):
    """
    Filter out videos that don't have any annotations.
    Returns cleaned ytvis data and list of removed video folders.
    """
    print("\nFiltering videos without annotations...")
    
    # Get all video IDs that have actual annotations (not all None segmentations)
    videos_with_annotations = set()
    for ann in ytvis['annotations']:
        # Check if this annotation has any actual segmentations (not all None)
        if any(seg is not None for seg in ann['segmentations']):
            videos_with_annotations.add(ann['video_id'])
    
    # Find videos to remove
    videos_to_remove = []
    for video in ytvis['videos']:
        if video['id'] not in videos_with_annotations:
            videos_to_remove.append(video)
    
    # Remove videos from JSON
    ytvis['videos'] = [v for v in ytvis['videos'] if v['id'] in videos_with_annotations]
    
    # Remove corresponding annotations (shouldn't be any, but just in case)
    ytvis['annotations'] = [a for a in ytvis['annotations'] if a['video_id'] in videos_with_annotations]
    
    # Remove video folders from images directory
    removed_folders = []
    for video in videos_to_remove:
        if video['file_names']:
            # Extract video folder name from first file path
            first_file = video['file_names'][0]
            if '/' in first_file:
                video_folder = first_file.split('/')[0]
                video_dir = os.path.join(image_out_dir, video_folder)
                if os.path.exists(video_dir):
                    try:
                        shutil.rmtree(video_dir)
                        removed_folders.append(video_folder)
                        print(f"Removed video folder without annotations: {video_folder}")
                    except Exception as e:
                        print(f"Warning: Could not remove folder {video_folder}: {e}")
    
    print(f"Removed {len(videos_to_remove)} videos without annotations from JSON")
    print(f"Removed {len(removed_folders)} video folders from images directory")
    
    return ytvis, removed_folders

def validate_dataset_integrity(ytvis, image_out_dir):
    """
    Validate that all videos in the JSON have at least one annotation with actual segmentations.
    Returns list of problematic videos found.
    """
    print("\nValidating dataset integrity...")
    
    # Create video ID to name mapping
    video_id_to_name = {}
    for video in ytvis['videos']:
        if video['file_names']:
            first_file = video['file_names'][0]
            if '/' in first_file:
                video_folder = first_file.split('/')[0]
                video_id_to_name[video['id']] = video_folder
    
    # Check each video has annotations with actual segmentations
    problematic_videos = []
    for video in ytvis['videos']:
        video_id = video['id']
        video_name = video_id_to_name.get(video_id, f"Unknown_{video_id}")
        
        # Check if video has 0 frames (invalid)
        if video['length'] <= 0:
            problematic_videos.append((video_id, video_name, f"Invalid: {video['length']} frames"))
            continue
        
        video_annotations = [ann for ann in ytvis['annotations'] if ann['video_id'] == video_id]
        
        if not video_annotations:
            problematic_videos.append((video_id, video_name, "No annotations"))
            continue
        
        # Check if any annotation has actual segmentations
        has_actual_segmentations = False
        for ann in video_annotations:
            if any(seg is not None for seg in ann['segmentations']):
                has_actual_segmentations = True
                break
        
        if not has_actual_segmentations:
            problematic_videos.append((video_id, video_name, "All segmentations are None"))
    
    if problematic_videos:
        print(f"WARNING: Found {len(problematic_videos)} videos with issues:")
        for video_id, video_name, issue in problematic_videos:
            print(f"  - Video {video_id} ({video_name}): {issue}")
    else:
        print("✓ All videos have valid annotations with actual segmentations")
    
    return problematic_videos

def print_dataset_statistics(ytvis):
    """
    Print detailed statistics about the dataset including:
    - Number of videos per category
    - Number of annotated frames per category
    - Frame statistics (mean, median, min, max) per category
    """
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    # Create category ID to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in ytvis['categories']}
    
    # Initialize statistics containers
    cat_video_count = defaultdict(int)
    cat_annotated_frames = defaultdict(int)
    cat_annotated_frames_per_video = defaultdict(list)  # Track annotated frames per video
    cat_video_ids = defaultdict(set)
    
    # Process videos and annotations
    video_id_to_length = {video['id']: video['length'] for video in ytvis['videos']}
    
    for ann in ytvis['annotations']:
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
    total_videos = len(ytvis['videos'])
    total_annotations = len(ytvis['annotations'])
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
    print(f"Total categories: {len(ytvis['categories'])}")
    print("="*80)

def convert_coco_to_ytvis(
    metadata_csv_path,
    base_data_dir,
    output_dir,
    final_json_path
):
    # Step 1: Load and filter metadata
    df = pd.read_csv(metadata_csv_path)
    df = df[
        (df["mask annotated"] == 1) &
        (df["Mask Annotator Name"].isin(["Simone", "Bushra"])) &
        (df["Single Fish"] == 1)
    ]

    # Step 2: Create output directories
    os.makedirs(output_dir, exist_ok=True)
    image_out_dir = os.path.join(output_dir)
    os.makedirs(image_out_dir, exist_ok=True)

    # Step 3: Load existing data to check for already processed videos
    existing_ytvis = load_existing_ytvis(final_json_path)
    existing_video_names = get_existing_video_names(existing_ytvis)
    
    print(f"Found {len(existing_video_names)} existing videos: {sorted(existing_video_names)}")

    # Step 4: Initialize YTVIS json components
    if existing_ytvis:
        ytvis = existing_ytvis
        # Get the next available IDs
        video_id_counter = max([v['id'] for v in ytvis['videos']]) + 1 if ytvis['videos'] else 1
        annotation_id_counter = max([a['id'] for a in ytvis['annotations']]) + 1 if ytvis['annotations'] else 1
        used_cat_ids = set(a['category_id'] for a in ytvis['annotations'])
        
        # Filter existing data to remove videos without annotations
        print("Checking existing data for videos without annotations...")
        ytvis, existing_removed = filter_videos_without_annotations(ytvis, image_out_dir)
        if existing_removed:
            print(f"Removed {len(existing_removed)} existing videos without annotations")
            # Update existing_video_names to reflect the removal
            existing_video_names = get_existing_video_names(ytvis)
            print(f"Updated existing videos: {sorted(existing_video_names)}")
        
        # Force overwrite the problematic video if it exists
        problematic_video = "Ganaraska__Ganaraska 2024__11222024-11282024__24  11  22  14  07__93"
        if problematic_video in existing_video_names:
            print(f"Force overwriting problematic video: {problematic_video}")
            # Remove the video from existing data
            ytvis['videos'] = [v for v in ytvis['videos'] if not (v.get('file_names') and v['file_names'] and v['file_names'][0].startswith(f"{problematic_video}/"))]
            ytvis['annotations'] = [a for a in ytvis['annotations'] if not any(v.get('file_names') and v['file_names'] and v['file_names'][0].startswith(f"{problematic_video}/") for v in ytvis['videos'] if v['id'] == a['video_id'])]
            # Update existing_video_names to reflect the removal
            existing_video_names = get_existing_video_names(ytvis)
            print(f"Removed problematic video from existing data. Updated existing videos: {sorted(existing_video_names)}")
    else:
        ytvis = {
            "videos": [],
            "annotations": [],
            "categories": [],
            "info": {},
            "licenses": []
        }
        video_id_counter = 1
        annotation_id_counter = 1
        used_cat_ids = set()

    track_to_anns = defaultdict(list)

    first_categories_loaded = len(ytvis['categories']) > 0

    processed_count = 0
    skipped_count = 0
    no_annotations_videos = []  # Track videos skipped due to no annotations

    for _, row in tqdm(df.iterrows(), total=len(df)):
        annotator = row["Mask Annotator Name"].strip().lower()
        video_folder = row["Fish Computer Folder"]
        rel_path = os.path.join(annotator, video_folder)
        video_path = os.path.join(base_data_dir, rel_path)
        ann_path = os.path.join(video_path, "annotations", "edited_instances_default.json")
        img_path = os.path.join(video_path, "images")

        # Load COCO file first
        if not (os.path.exists(ann_path) and os.path.exists(img_path)):
            print(f"Skipping inaccessible: {ann_path}")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        # Check if video already exists in JSON - if so, skip adding to JSON
        if video_folder in existing_video_names:
            print(f"Skipping existing video in JSON: {video_folder}")
            # Still copy images if they don't exist in output directory
            if not check_video_images_exist(image_out_dir, video_folder):
                print(f"Copying images for video not in images directory: {video_folder}")
                # Copy images but don't add to JSON
                images = sorted(coco["images"], key=lambda x: x["file_name"])
                for img in images:
                    src_img = os.path.join(img_path, img["file_name"])
                    dst_img = os.path.join(image_out_dir, f"{video_folder}/{img['file_name']}")
                    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                    shutil.copy2(src_img, dst_img)
            else:
                print(f"Images already exist for video: {video_folder}")
            skipped_count += 1
            continue
            
        # Check if images exist in output directory - if so, skip copying them
        if check_video_images_exist(image_out_dir, video_folder):
            print(f"Images exist for video, skipping image copy: {video_folder}")
            # Still add to JSON if not already there
            print(f"Adding video to JSON: {video_folder}")
        else:
            print(f"Copying images for video: {video_folder}")
            # Copy images
            images = sorted(coco["images"], key=lambda x: x["file_name"])
            image_filenames = []
            for img in images:
                src_img = os.path.join(img_path, img["file_name"])
                dst_img = os.path.join(image_out_dir, f"{video_folder}/{img['file_name']}")
                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                shutil.copy2(src_img, dst_img)
                image_filenames.append(f"{video_folder}/{img['file_name']}")

        if not first_categories_loaded:
            ytvis["categories"] = coco["categories"]
            first_categories_loaded = True

        # Always create image_filenames for JSON, regardless of whether images were copied
        images = sorted(coco["images"], key=lambda x: x["file_name"])
        image_filenames = []
        
        # Special handling for the problematic video - skip first 100 frames
        if video_folder == "Ganaraska__Ganaraska 2024__11222024-11282024__24  11  22  14  07__93":
            print(f"Special handling: Skipping first 100 frames for {video_folder}")
            # Filter images to start from frame 100 (00100.jpg)
            filtered_images = []
            for img in images:
                # Extract frame number from filename (assuming format like 00100.jpg)
                try:
                    frame_num = int(img["file_name"].split('.')[0])
                    if frame_num >= 100:
                        filtered_images.append(img)
                        image_filenames.append(f"{video_folder}/{img['file_name']}")
                except ValueError:
                    # If filename doesn't match expected format, include it
                    filtered_images.append(img)
                    image_filenames.append(f"{video_folder}/{img['file_name']}")
            images = filtered_images
        else:
            # Normal processing for all other videos
            for img in images:
                image_filenames.append(f"{video_folder}/{img['file_name']}")

        video_id = video_id_counter
        video_id_counter += 1

        num_frames = len(images)
        
        # Check if video has 0 frames (invalid)
        if num_frames <= 0:
            no_annotations_videos.append(video_folder)
            print(f"Skipping video with 0 frames: {video_folder}")
            video_id_counter -= 1  # Decrement the counter since we didn't actually use this ID
            continue
            
        image_id_to_idx = {img["id"]: i for i, img in enumerate(images)}

        ytvis["videos"].append({
            "id": video_id,
            "file_names": image_filenames,
            "width": images[0]["width"],
            "height": images[0]["height"],
            "length": num_frames,
            "coco_url": "",
            "flickr_url": "",
            "date_captured": ""
        })

        # Group annotations by track_id for all frames (no trimming)
        track_anns = defaultdict(lambda: {"segmentations": [None]*num_frames, "bboxes": [None]*num_frames, "areas": [None]*num_frames, "category_id": None})
        
        # Create set of valid image IDs for the filtered images
        valid_image_ids = {img["id"] for img in images}
        
        for ann in coco["annotations"]:
            # Skip annotations for images that were filtered out
            if ann["image_id"] not in valid_image_ids:
                continue
                
            tid = ann["attributes"]["track_id"]
            frame_idx = image_id_to_idx[ann["image_id"]]
            track_anns[tid]["segmentations"][frame_idx] = ann["segmentation"]
            track_anns[tid]["bboxes"][frame_idx] = ann.get("bbox", None)
            track_anns[tid]["areas"][frame_idx] = ann.get("area", None)
            track_anns[tid]["category_id"] = ann["category_id"]
            track_anns[tid]["iscrowd"] = ann["iscrowd"]

        # Check if this video has any annotations
        if not track_anns:
            no_annotations_videos.append(video_folder)
            print(f"Skipping video with no annotations: {video_folder}")
            # Remove the video from the videos list since it has no annotations
            ytvis["videos"].pop()  # Remove the last added video
            video_id_counter -= 1  # Decrement the counter
            continue
        
        # Check if any track has actual annotations (not all None segmentations)
        has_actual_annotations = False
        for track_id, info in track_anns.items():
            if any(seg is not None for seg in info["segmentations"]):
                has_actual_annotations = True
                break
        
        if not has_actual_annotations:
            no_annotations_videos.append(video_folder)
            print(f"Skipping video with no actual annotations (all segmentations are None): {video_folder}")
            # Remove the video from the videos list since it has no actual annotations
            ytvis["videos"].pop()  # Remove the last added video
            video_id_counter -= 1  # Decrement the counter
            continue
        
        for track_id, info in track_anns.items():
            ytvis["annotations"].append({
                "id": annotation_id_counter,
                "video_id": video_id,
                "category_id": info["category_id"],
                "height": images[0]["height"],  # required
                "width": images[0]["width"],    # required
                "length": num_frames,    # required
                "score": 1.0,  # GT annotations
                "iscrowd": info["iscrowd"],
                "occ_score": 0.0,
                "segmentations": info["segmentations"],
                "bboxes": info["bboxes"],
                "areas": info["areas"],        
            })
            used_cat_ids.add(info["category_id"])
            annotation_id_counter += 1

        processed_count += 1

    # Only convert segmentations for newly added annotations
    if processed_count > 0:
        ytvis = convert_all_segmentations_to_compressed_rle(ytvis)
    
    # Filter categories to only those present in the annotations
    ytvis["categories"] = [cat for cat in ytvis["categories"] if cat["id"] in used_cat_ids]
    
    # Filter out videos without annotations (both new and existing)
    ytvis, removed_folders = filter_videos_without_annotations(ytvis, image_out_dir)
    
    # Validate dataset integrity
    problematic_videos = validate_dataset_integrity(ytvis, image_out_dir)
    
    # If there are still problematic videos, remove them
    if problematic_videos:
        print("\nRemoving problematic videos...")
        problematic_video_ids = {video_id for video_id, _, _ in problematic_videos}
        ytvis['videos'] = [v for v in ytvis['videos'] if v['id'] not in problematic_video_ids]
        ytvis['annotations'] = [a for a in ytvis['annotations'] if a['video_id'] not in problematic_video_ids]
        
        # Remove corresponding folders
        for video_id, video_name, _ in problematic_videos:
            video_dir = os.path.join(image_out_dir, video_name)
            if os.path.exists(video_dir):
                try:
                    shutil.rmtree(video_dir)
                    print(f"Removed problematic video folder: {video_name}")
                except Exception as e:
                    print(f"Warning: Could not remove folder {video_name}: {e}")
        
        print(f"Removed {len(problematic_videos)} problematic videos")
    
    # Save final YTVIS JSON
    with open(final_json_path, "w") as f:
        json.dump(ytvis, f, indent=2)
    
    # Print dataset statistics
    print_dataset_statistics(ytvis)
    
    # Count videos in images directory vs JSON file
    json_video_count = len(ytvis['videos'])
    
    # Count videos in images directory
    image_dir_video_count = 0
    if os.path.exists(image_out_dir):
        for item in os.listdir(image_out_dir):
            item_path = os.path.join(image_out_dir, item)
            if os.path.isdir(item_path):
                # Check if directory contains image files
                image_files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if len(image_files) > 0:
                    image_dir_video_count += 1
    
    print(f"Processed {processed_count} new videos, skipped {skipped_count} existing videos")
    
    # Report videos skipped due to no annotations
    if no_annotations_videos:
        print(f"\nSkipped {len(no_annotations_videos)} videos due to no annotations:")
        for video in no_annotations_videos:
            print(f"  - {video}")
    else:
        print(f"\nNo videos were skipped due to missing annotations.")
    
    print(f"Saved YTVIS JSON to {final_json_path}")
    print(f"Videos in JSON file: {json_video_count}")
    print(f"Videos in images directory: {image_dir_video_count}")


if __name__ == "__main__":
    convert_coco_to_ytvis(
        metadata_csv_path="/home/simone/fish-dvis/data_scripts/fishway_metadata.csv",
        base_data_dir="/data/labeled",
        output_dir="/data/fishway_ytvis/all_videos",
        final_json_path="/data/fishway_ytvis/all_videos.json"
    ) 