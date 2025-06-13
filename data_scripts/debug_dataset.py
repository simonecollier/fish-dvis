import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Use absolute imports
from custom_data.ytvis_loader import register_ytvis_instances
from custom_data.register_datasets import register_all_ytvis_fishway
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


if __name__ == "__main__":
    check_num_frames()