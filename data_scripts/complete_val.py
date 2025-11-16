#!/usr/bin/env python3
"""
Replace trimmed video data in validation JSON with complete data from all_videos.json.

This script takes a validation JSON file that may have trimmed video lengths and
replaces the file_names and annotations for each video ID with the complete data
from all_videos.json. This is useful when validation sets have been subsetted or
trimmed and you want to restore the full video data.

Usage:
    python complete_val.py <val_json_path> [all_videos_json_path]
    
Default all_videos.json path: /data/fishway_ytvis/all_videos.json
"""

import json
import argparse
import sys
from pathlib import Path


def complete_val_json(val_json_path, all_videos_json_path, output_json_path=None):
    """
    Replace trimmed video data in val JSON with complete data from all_videos.json.
    
    Args:
        val_json_path: Path to input validation JSON file
        all_videos_json_path: Path to all_videos.json with complete data
        output_json_path: Path to output JSON file (auto-generated if None)
        
    Returns:
        Path to the output JSON file
    """
    # Load validation JSON
    print(f"Loading validation JSON from: {val_json_path}")
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    # Load all_videos.json
    print(f"Loading complete video data from: {all_videos_json_path}")
    with open(all_videos_json_path, 'r') as f:
        all_videos_data = json.load(f)
    
    # Create lookup dictionaries for faster access
    all_videos_by_id = {video['id']: video for video in all_videos_data.get('videos', [])}
    all_annotations_by_video_id = {}
    for ann in all_videos_data.get('annotations', []):
        video_id = ann['video_id']
        if video_id not in all_annotations_by_video_id:
            all_annotations_by_video_id[video_id] = []
        all_annotations_by_video_id[video_id].append(ann)
    
    print(f"Loaded {len(val_data['videos'])} videos from val JSON")
    print(f"Loaded {len(all_videos_data['videos'])} videos from all_videos.json")
    
    # Create output data structure
    completed_data = {
        'videos': [],
        'annotations': [],
        'categories': val_data.get('categories', []),
        'info': val_data.get('info', {}),
        'licenses': val_data.get('licenses', [])
    }
    
    # Process each video in the val set
    missing_videos = []
    for i, video in enumerate(val_data['videos']):
        video_id = video['id']
        print(f"Processing video {i+1}/{len(val_data['videos'])}: ID {video_id}")
        
        # Find corresponding video in all_videos.json
        if video_id not in all_videos_by_id:
            print(f"  WARNING: Video ID {video_id} not found in all_videos.json! Skipping...")
            missing_videos.append(video_id)
            continue
        
        # Get complete video data
        complete_video = all_videos_by_id[video_id]
        original_frame_count = len(video.get('file_names', []))
        complete_frame_count = len(complete_video.get('file_names', []))
        
        print(f"  Original frames: {original_frame_count}, Complete frames: {complete_frame_count}")
        
        # Replace with complete video data
        completed_data['videos'].append(complete_video)
        
        # Get complete annotations for this video
        complete_annotations = all_annotations_by_video_id.get(video_id, [])
        print(f"  Found {len(complete_annotations)} complete annotations")
        completed_data['annotations'].extend(complete_annotations)
    
    if missing_videos:
        print(f"\nWARNING: {len(missing_videos)} video(s) not found in all_videos.json: {missing_videos}")
    
    # Generate output path if not provided
    if output_json_path is None:
        val_path = Path(val_json_path)
        output_json_path = val_path.parent / f"{val_path.stem}_all_frames{val_path.suffix}"
    
    # Save completed data
    print(f"\nSaving completed validation JSON to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(completed_data, f, indent=2)
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Input videos: {len(val_data['videos'])}")
    print(f"  Output videos: {len(completed_data['videos'])}")
    print(f"  Input annotations: {len(val_data['annotations'])}")
    print(f"  Output annotations: {len(completed_data['annotations'])}")
    print(f"  Missing videos: {len(missing_videos)}")
    print("="*60)
    
    return output_json_path


def main():
    parser = argparse.ArgumentParser(
        description='Replace trimmed video data in validation JSON with complete data from all_videos.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python complete_val.py /data/fishway_ytvis/val.json
  python complete_val.py /data/fishway_ytvis/val_fold1.json /data/fishway_ytvis/all_videos.json
  python complete_val.py /data/fishway_ytvis/val.json --output /custom/path/output.json
        """
    )
    parser.add_argument('val_json_path', type=str, 
                       help='Path to input validation JSON file')
    parser.add_argument('all_videos_json_path', type=str, nargs='?',
                       default='/data/fishway_ytvis/all_videos.json',
                       help='Path to all_videos.json (default: /data/fishway_ytvis/all_videos.json)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSON file path (default: <input_name>_all_frames.json)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.val_json_path).exists():
        print(f"Error: Input file not found: {args.val_json_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check if all_videos.json exists
    if not Path(args.all_videos_json_path).exists():
        print(f"Error: all_videos.json not found: {args.all_videos_json_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Complete the validation JSON
        output_path = complete_val_json(
            args.val_json_path, 
            args.all_videos_json_path,
            args.output
        )
        
        print(f"\n✓ Successfully created completed validation set: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error during completion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

