#!/usr/bin/env python3
"""
Script to subset a YTVIS validation JSON file by video ID(s).

Usage:
    python subset_val.py <val_json_path> [video_id1] [video_id2] ... [video_idN]

Examples:
    python subset_val.py /path/to/val.json 12
    python subset_val.py /path/to/val.json 12 13 14
"""

import json
import sys
import os
import argparse
from pathlib import Path


def generate_output_filename(input_path, video_ids):
    """
    Generate output filename based on video IDs.
    
    - 1 video: val_vid_12.json
    - 2-3 videos: val_vid_12_13_14.json
    - >3 videos: val_multi.json (or val_multi_1.json, val_multi_2.json, etc. if exists)
    """
    input_dir = os.path.dirname(input_path)
    input_basename = os.path.basename(input_path)
    
    # Remove .json extension if present
    if input_basename.endswith('.json'):
        base_name = input_basename[:-5]
    else:
        base_name = input_basename
    
    num_videos = len(video_ids)
    
    if num_videos == 1:
        output_name = f"{base_name}_vid_{video_ids[0]}.json"
    elif num_videos <= 3:
        video_ids_str = "_".join(str(vid) for vid in video_ids)
        output_name = f"{base_name}_vid_{video_ids_str}.json"
    else:
        # More than 3 videos - use val_multi.json with numbering if needed
        base_output_name = f"{base_name}_multi.json"
        output_path = os.path.join(input_dir, base_output_name)
        
        # Check if file exists, if so add number suffix
        counter = 1
        while os.path.exists(output_path):
            base_output_name = f"{base_name}_multi_{counter}.json"
            output_path = os.path.join(input_dir, base_output_name)
            counter += 1
        
        return output_path
    
    return os.path.join(input_dir, output_name)


def subset_val_json(input_path, video_ids):
    """
    Create a subset of the validation JSON file containing only specified video IDs.
    
    Args:
        input_path: Path to input val.json file
        video_ids: List of video IDs to include
        
    Returns:
        Path to the output JSON file
    """
    # Convert video_ids to integers and set for fast lookup
    video_ids = set(int(vid) for vid in video_ids)
    
    # Load the input JSON
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Filter videos
    filtered_videos = [video for video in data.get('videos', []) if video.get('id') in video_ids]
    
    # Filter annotations (annotations have 'video_id' field)
    filtered_annotations = [
        ann for ann in data.get('annotations', [])
        if ann.get('video_id') in video_ids
    ]
    
    # Categories are kept as-is
    filtered_categories = data.get('categories', [])
    
    # Create output data structure
    output_data = {
        'videos': filtered_videos,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }
    
    # Generate output filename
    output_path = generate_output_filename(input_path, sorted(video_ids))
    
    # Write output JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Created subset JSON: {output_path}")
    print(f"  Videos: {len(filtered_videos)}")
    print(f"  Annotations: {len(filtered_annotations)}")
    print(f"  Categories: {len(filtered_categories)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Subset a YTVIS validation JSON file by video ID(s)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python subset_val.py /path/to/val.json 12
  python subset_val.py /path/to/val.json 12 13 14
  python subset_val.py /path/to/val.json 12 13 14 15 16 17
        """
    )
    parser.add_argument('val_json_path', type=str, help='Path to input val.json file')
    parser.add_argument('video_ids', type=int, nargs='+', help='Video ID(s) to include in subset')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.val_json_path):
        print(f"Error: Input file not found: {args.val_json_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create subset
    subset_val_json(args.val_json_path, args.video_ids)


if __name__ == '__main__':
    main()

