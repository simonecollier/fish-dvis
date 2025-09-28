import json
import numpy as np
from collections import defaultdict

def validate_ytvis(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Map video_id to its length
    video_lengths = {video["id"]: video["length"] for video in data["videos"]}
    video_names = {video["id"]: video.get("file_names", [None])[0] for video in data["videos"]}
    errors_found = False

    # Build category id to name mapping
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Build mapping: class -> list of (video_id, num_frames)
    class_to_videos = defaultdict(list)
    video_to_class = defaultdict(set)
    video_annotated_frames = defaultdict(int)  # Track annotated frames per video
    
    for ann in data["annotations"]:
        video_id = ann["video_id"]
        cat_id = ann["category_id"]
        class_to_videos[cat_id].append(video_id)
        video_to_class[video_id].add(cat_id)
        
        # Count annotated frames for this video (using the length of segmentations/bboxes)
        annotated_frames = len(ann.get("segmentations", []))
        video_annotated_frames[video_id] = max(video_annotated_frames[video_id], annotated_frames)

    # For each class, get unique videos and their frame counts
    print("\n=== Video and Frame Counts Per Class ===")
    for cat_id, video_ids in class_to_videos.items():
        unique_videos = sorted(set(video_ids))
        print(f"Class '{cat_id_to_name.get(cat_id, cat_id)}' ({cat_id}): {len(unique_videos)} videos")
        for vid in unique_videos:
            n_frames = video_annotated_frames.get(vid, 0)
            total_frames = video_lengths.get(vid, '?')
            vname = video_names.get(vid, str(vid))
            print(f"  Video ID {vid} (first file: {vname}): {n_frames} annotated frames (total: {total_frames})")
        print()

    # Print summary table
    print("\n=== Summary Table ===")
    print(f"{'Class':20} {'#Videos':>8} {'AnnotatedFrames':>15} {'MeanFrames':>12} {'MedianFrames':>12} {'MaxFrames':>10} {'MinFrames':>10}")
    for cat_id, video_ids in class_to_videos.items():
        unique_videos = sorted(set(video_ids))
        frame_counts = [video_annotated_frames.get(vid, 0) for vid in unique_videos]
        total_frames = sum(frame_counts)
        mean_frames = np.mean(frame_counts) if frame_counts else 0
        median_frames = np.median(frame_counts) if frame_counts else 0
        max_frames = max(frame_counts) if frame_counts else 0
        min_frames = min(frame_counts) if frame_counts else 0
        cname = cat_id_to_name.get(cat_id, str(cat_id))
        print(f"{cname:20} {len(unique_videos):8} {total_frames:15} {mean_frames:12.1f} {median_frames:12.1f} {max_frames:10} {min_frames:10}")

    # Existing validation logic
    for ann in data["annotations"]:
        video_id = ann["video_id"]
        expected_len = video_lengths.get(video_id)

        for field in ["segmentations", "bboxes", "areas"]:
            values = ann.get(field, [])
            if len(values) != expected_len:
                print(f"Error: Annotation ID {ann['id']} has mismatched {field} length: {len(values)} vs expected {expected_len}")
                errors_found = True

        # Optional: Check segmentation format
        for idx, seg in enumerate(ann.get("segmentations", [])):
            if seg is not None:
                if isinstance(seg, dict):
                    if "counts" not in seg or "size" not in seg:
                        print(f"Warning: Annotation ID {ann['id']} frame {idx} has invalid RLE segmentation.")
                        errors_found = True
                elif not isinstance(seg, list):
                    print(f"Warning: Annotation ID {ann['id']} frame {idx} has segmentation of unexpected type: {type(seg)}")
                    errors_found = True

    if not errors_found:
        print("✅ JSON passed validation.")
    else:
        print("❌ Errors found in JSON.")


if __name__ == "__main__":
    validate_ytvis("/data/fishway_ytvis/all_videos.json")
    validate_ytvis("/data/fishway_ytvis/train.json")
    validate_ytvis("/data/fishway_ytvis/val.json") 