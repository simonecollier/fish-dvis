import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mask_util

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Use absolute imports
from data_scripts.ytvis_loader import load_ytvis_json, register_all_ytvis_fishway
from detectron2.data import MetadataCatalog, DatasetCatalog


def debug_dataset():
    # Register datasets
    register_all_ytvis_fishway()
    
    # Check train dataset
    dataset_name = "ytvis_fishway_train"
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print(f"\nAnalyzing dataset: {dataset_name}")
    print(f"Number of videos: {len(dataset_dicts)}")
    
    # Check categories
    print("\nCategories:")
    if hasattr(metadata, "thing_classes"):
        for i, cls in enumerate(metadata.thing_classes):
            print(f"  {i}: {cls}")
    else:
        print("No categories found in metadata!")
    
    # Check a few videos
    print("\nChecking first video:")
    video = dataset_dicts[0]
    print(f"  Video ID: {video['video_id']}")
    print(f"  Length: {video['length']} frames")
    print(f"  Size: {video['height']}x{video['width']}")
    
    # Check annotations
    n_frames_with_annos = sum(1 for frame_annos in video['annotations'] if frame_annos)
    print(f"  Frames with annotations: {n_frames_with_annos}/{video['length']}")
    
    # Verify files exist
    print("\nChecking file paths:")
    for i, fname in enumerate(video['file_names'][:3]):  # Check first 3 frames
        exists = os.path.exists(fname)
        print(f"  Frame {i}: {fname} {'✓' if exists else '✗'}")

def has_consecutive_annotations(annos, num):
    count = 0
    for a in annos:
        if a:
            count += 1
            if count >= num:
                return True
        else:
            count = 0
    return False

def check_num_frames():
    # Register datasets
    register_all_ytvis_fishway()
    dataset_name = "ytvis_fishway_train"
    videos = DatasetCatalog.get(dataset_name)
    video = videos[23]["annotations"]
    print(f"Length of video 23: {video}")        

    print(f"Total videos: {len(videos)}\n")
    for idx, video in enumerate(videos):
        frame_annos = video["annotations"]
        # Count frames with at least one annotation
        frames_with_ann = sum(1 for ann in frame_annos if ann)
        total_frames = len(frame_annos)
        has_seq = has_consecutive_annotations(frame_annos, 6)  # or your SAMPLING_FRAME_NUM
        print(f"Video {idx+1} (ID: {video.get('video_id', idx)}): {frames_with_ann}/{total_frames} frames with annotations | Has 6 consecutive: {has_seq}")

def decode_rle(rle_obj, height, width):
    # Ensure counts is a string for pycocotools
    if isinstance(rle_obj['counts'], bytes):
        rle_obj['counts'] = rle_obj['counts'].decode('utf-8')
    if isinstance(rle_obj['counts'], list):
        rle = mask_util.frPyObjects(rle_obj, height, width)
    else:
        rle = rle_obj
    return mask_util.decode(rle)

def visualize_video_with_masks(json_file, image_root, dataset_name, video_id, output_path):
    # Load dataset
    dataset = load_ytvis_json(json_file, image_root, dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    thing_classes = metadata.thing_classes if hasattr(metadata, "thing_classes") else []

    # Assign a fixed color per category for consistency
    np.random.seed(42)
    category_colors = {i: tuple(np.random.randint(0, 256, 3).tolist()) for i in range(len(thing_classes))}

    # Find the video
    video = next((v for v in dataset if v["video_id"] == video_id), None)
    if video is None:
        print(f"Video ID {video_id} not found.")
        return

    # Prepare video writer
    first_frame = cv2.imread(video["file_names"][0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5, (width, height))

    for frame_idx, frame_path in enumerate(video["file_names"]):
        img = cv2.imread(frame_path)
        overlay = img.copy()
        ann_list = video["annotations"][frame_idx]
        for ann in ann_list:
            seg = ann["segmentation"]
            # Robust mask decoding
            mask = None
            if isinstance(seg, dict):
                mask = decode_rle(seg, height, width)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                # Debug: check mask shape
                if mask.shape != (height, width):
                    print(f"Warning: mask shape {mask.shape} does not match image ({height}, {width})")
                    print(f"RLE size: {seg.get('size')}")
            if mask is not None:
                cat_id = ann["category_id"]
                color = category_colors.get(cat_id, (255, 0, 0))
                alpha = 0.7
                for c in range(3):
                    overlay[..., c] = np.where(
                        mask > 0,
                        (alpha * color[c] + (1 - alpha) * overlay[..., c]).astype(np.uint8),
                        overlay[..., c]
                    )
            # Draw bbox
            x1, y1, x2, y2 = map(int, ann["bbox"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            cat_id = ann["category_id"]
            label = thing_classes[cat_id] if cat_id < len(thing_classes) else str(cat_id)
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(overlay)
    out.release()
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    # Example usage
    json_file = "/data/fishway_ytvis/train.json"
    image_root = "/data/fishway_ytvis/train"
    dataset_name = "ytvis_fishway_train"
    video_id = 1  # Change to the video ID you want to visualize
    output_path = "/home/simone/test/video1_with_masks.mp4"
    visualize_video_with_masks(json_file, image_root, dataset_name, video_id, output_path)
