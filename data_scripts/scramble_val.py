#!/usr/bin/env python3
"""
Scramble frame order in YTVIS validation JSON.

This script takes a YTVIS JSON file and randomly scrambles the order of frames
in each video while maintaining the correct annotation-to-frame mappings.

The key insight is that DVIS uses the order of frames in the 'file_names' array,
not the alphabetical order of JPEG filenames. By scrambling the file_names array
and correspondingly reordering the frame-indexed annotation arrays (segmentations,
bboxes, areas), we can test if the model relies on temporal ordering.

Usage:
    python scramble_val.py [input_json] [output_json] [--seed SEED]
    
Default: scramble_val.py /data/fishway_ytvis/val.json /data/fishway_ytvis/val_scrambled.json
"""

import json
import random
import argparse
import copy
from pathlib import Path


def scramble_video_frames(video_data, annotations_for_video, random_seed=None):
    """
    Scramble the frame order for a single video and update corresponding annotations.
    
    Args:
        video_data: Dictionary containing video information including 'file_names'
        annotations_for_video: List of annotation dictionaries for this video
        random_seed: Optional seed for reproducible scrambling
        
    Returns:
        Tuple of (scrambled_video_data, scrambled_annotations)
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create a copy to avoid modifying the original
    scrambled_video = copy.deepcopy(video_data)
    scrambled_annotations = copy.deepcopy(annotations_for_video)
    
    num_frames = len(scrambled_video['file_names'])
    
    # Create scrambling mapping: old_index -> new_index
    original_indices = list(range(num_frames))
    scrambled_indices = original_indices.copy()
    random.shuffle(scrambled_indices)
    
    # Create inverse mapping: new_index -> old_index
    # This tells us: "the frame that is now at position i came from position inverse_mapping[i]"
    inverse_mapping = [0] * num_frames
    for new_pos, old_pos in enumerate(scrambled_indices):
        inverse_mapping[new_pos] = old_pos
    
    print(f"  Scrambling {num_frames} frames...")
    print(f"  First 10 mappings: {[(i, inverse_mapping[i]) for i in range(min(10, num_frames))]}")
    
    # Scramble the file_names array
    original_file_names = scrambled_video['file_names'].copy()
    for new_pos in range(num_frames):
        old_pos = inverse_mapping[new_pos]
        scrambled_video['file_names'][new_pos] = original_file_names[old_pos]
    
    # Update annotations: reorder frame-indexed arrays
    for ann in scrambled_annotations:
        # These arrays are indexed by frame position
        frame_indexed_fields = ['segmentations', 'bboxes', 'areas']
        
        for field in frame_indexed_fields:
            if field in ann and len(ann[field]) == num_frames:
                original_array = ann[field].copy()
                for new_pos in range(num_frames):
                    old_pos = inverse_mapping[new_pos]
                    ann[field][new_pos] = original_array[old_pos]
    
    return scrambled_video, scrambled_annotations


def scramble_ytvis_json(input_json_path, output_json_path, random_seed=42):
    """
    Scramble frame order in a YTVIS JSON file.
    
    Args:
        input_json_path: Path to input YTVIS JSON file
        output_json_path: Path to output scrambled YTVIS JSON file  
        random_seed: Seed for reproducible scrambling (None for random)
    """
    print(f"Loading YTVIS JSON from: {input_json_path}")
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['videos'])} videos, {len(data['annotations'])} annotations")
    
    # Create output data structure
    scrambled_data = {
        'videos': [],
        'annotations': [],
        'categories': data['categories'],  # Categories remain unchanged
        'info': data.get('info', {}),
        'licenses': data.get('licenses', [])
    }
    
    # Group annotations by video_id for easier processing
    annotations_by_video = {}
    for ann in data['annotations']:
        video_id = ann['video_id']
        if video_id not in annotations_by_video:
            annotations_by_video[video_id] = []
        annotations_by_video[video_id].append(ann)
    
    # Process each video
    for i, video in enumerate(data['videos']):
        video_id = video['id']
        print(f"Processing video {i+1}/{len(data['videos'])}: ID {video_id}")
        
        # Get annotations for this video
        video_annotations = annotations_by_video.get(video_id, [])
        print(f"  Found {len(video_annotations)} annotations")
        
        # Generate a unique seed for each video to ensure reproducibility
        # but different scrambling per video
        video_seed = None if random_seed is None else random_seed + video_id
        
        # Scramble this video's frames
        scrambled_video, scrambled_annotations = scramble_video_frames(
            video, video_annotations, video_seed
        )
        
        # Add to output
        scrambled_data['videos'].append(scrambled_video)
        scrambled_data['annotations'].extend(scrambled_annotations)
    
    # Save scrambled data
    print(f"Saving scrambled YTVIS JSON to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(scrambled_data, f, indent=2)
    
    print("Frame scrambling completed!")
    print(f"Original videos: {len(data['videos'])}")
    print(f"Scrambled videos: {len(scrambled_data['videos'])}")
    print(f"Original annotations: {len(data['annotations'])}")  
    print(f"Scrambled annotations: {len(scrambled_data['annotations'])}")


def verify_scrambling(original_json_path, scrambled_json_path):
    """
    Verify that the scrambling was done correctly by checking a few samples.
    """
    print("\n=== Verification ===")
    
    with open(original_json_path, 'r') as f:
        original = json.load(f)
    with open(scrambled_json_path, 'r') as f:
        scrambled = json.load(f)
    
    # Check first video
    orig_video = original['videos'][0]
    scram_video = scrambled['videos'][0]
    
    print(f"Original first video frames: {orig_video['file_names'][:3]}...")
    print(f"Scrambled first video frames: {scram_video['file_names'][:3]}...")
    
    # Verify that all original frames are still present (just reordered)
    orig_set = set(orig_video['file_names'])
    scram_set = set(scram_video['file_names'])
    
    if orig_set == scram_set:
        print("✓ All original frames are present in scrambled version")
    else:
        print("✗ Frame mismatch detected!")
        
    # Check that frame count matches
    if len(orig_video['file_names']) == len(scram_video['file_names']):
        print("✓ Frame count matches")
    else:
        print("✗ Frame count mismatch!")
    
    # Check annotation lengths match
    orig_ann = next(ann for ann in original['annotations'] if ann['video_id'] == orig_video['id'])
    scram_ann = next(ann for ann in scrambled['annotations'] if ann['video_id'] == scram_video['id'])
    
    if len(orig_ann['segmentations']) == len(scram_ann['segmentations']):
        print("✓ Annotation array lengths match")
    else:
        print("✗ Annotation array length mismatch!")


def main():
    parser = argparse.ArgumentParser(description='Scramble frame order in YTVIS JSON')
    parser.add_argument('input_json', nargs='?', 
                       default='/data/fishway_ytvis/val.json',
                       help='Input YTVIS JSON file (default: /data/fishway_ytvis/val.json)')
    parser.add_argument('output_json', nargs='?',
                       default='/data/fishway_ytvis/val_scrambled_400.json', 
                       help='Output scrambled JSON file (default: /data/fishway_ytvis/val_scrambled.json)')
    parser.add_argument('--seed', type=int, default=400,
                       help='Random seed for reproducible scrambling (default: 42)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the scrambling after completion')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_json).exists():
        print(f"Error: Input file {args.input_json} does not exist!")
        return 1
    
    # Create output directory if needed
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Perform scrambling
        scramble_ytvis_json(args.input_json, args.output_json, args.seed)
        
        # Optional verification
        if args.verify:
            verify_scrambling(args.input_json, args.output_json)
            
        print(f"\n✓ Successfully created scrambled validation set: {args.output_json}")
        return 0
        
    except Exception as e:
        print(f"Error during scrambling: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
