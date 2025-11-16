#!/usr/bin/env python3
"""
Script to trim val.json videos to max 300 frames, centered around frames with segmentations.

Usage:
    python trim_val_json.py <input_val.json> <output_val.json> [--max-frames 300]
"""

import json
import argparse
from typing import List, Dict, Any, Tuple


def find_segmentation_range(segmentations: List[Any]) -> Tuple[int, int]:
    """
    Find the first and last frame indices that have segmentations (non-None values).
    
    Args:
        segmentations: List of segmentation data (None for frames without segmentations)
        
    Returns:
        Tuple of (first_frame_idx, last_frame_idx) with segmentations, or (0, 0) if none found
    """
    first_idx = None
    last_idx = None
    
    for i, seg in enumerate(segmentations):
        if seg is not None:
            if first_idx is None:
                first_idx = i
            last_idx = i
    
    if first_idx is None:
        return (0, 0)
    
    return (first_idx, last_idx)


def center_window_around_range(
    seg_start: int, 
    seg_end: int, 
    total_frames: int, 
    max_frames: int
) -> Tuple[int, int]:
    """
    Center a window of max_frames around the segmentation range.
    Ensures all frames with segmentations are included in the window.
    
    Args:
        seg_start: First frame index with segmentation
        seg_end: Last frame index with segmentation
        total_frames: Total number of frames in the video
        max_frames: Maximum number of frames in the window
        
    Returns:
        Tuple of (start_idx, end_idx) for the trimmed window
    """
    seg_range = seg_end - seg_start + 1
    
    # If segmentation range is larger than max_frames, we can't include all
    # Center on the middle of the segmentation range to include as many as possible
    if seg_range >= max_frames:
        center = (seg_start + seg_end) // 2
        start_idx = max(0, center - max_frames // 2)
        end_idx = min(total_frames, start_idx + max_frames)
        # Adjust if we hit the boundaries
        if end_idx == total_frames:
            start_idx = max(0, end_idx - max_frames)
        # Verify we're including as much of the segmentation range as possible
        # The window should overlap with the segmentation range
        assert not (end_idx <= seg_start or start_idx > seg_end), \
            f"Window [{start_idx}, {end_idx}] doesn't overlap with segmentation range [{seg_start}, {seg_end}]"
        return (start_idx, end_idx)
    
    # Center the window around the segmentation range
    seg_center = (seg_start + seg_end) // 2
    window_half = max_frames // 2
    
    # Calculate ideal center position
    ideal_start = seg_center - window_half
    ideal_end = ideal_start + max_frames
    
    # Adjust if we hit boundaries, but ensure segmentation range is always included
    if ideal_start < 0:
        # Hit left boundary - shift right
        start_idx = 0
        end_idx = min(total_frames, max_frames)
        # Verify segmentation range is still included
        if seg_end >= end_idx:
            # Segmentation extends beyond window, shift window right
            end_idx = min(total_frames, seg_end + 1)
            start_idx = max(0, end_idx - max_frames)
    elif ideal_end > total_frames:
        # Hit right boundary - shift left
        end_idx = total_frames
        start_idx = max(0, end_idx - max_frames)
        # Verify segmentation range is still included
        if seg_start < start_idx:
            # Segmentation starts before window, shift window left
            start_idx = seg_start
            end_idx = min(total_frames, start_idx + max_frames)
    else:
        # No boundary issues, use ideal window
        start_idx = ideal_start
        end_idx = ideal_end
    
    # Final verification: ensure segmentation range is within the window
    assert seg_start >= start_idx and seg_end < end_idx, \
        f"Segmentation range [{seg_start}, {seg_end}] not fully within window [{start_idx}, {end_idx}]"
    
    return (start_idx, end_idx)


def trim_video_data(
    video: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    max_frames: int
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int, int]:
    """
    Trim a video and its annotations to max_frames, centered around frames with segmentations.
    
    Args:
        video: Video dictionary from JSON
        annotations: List of annotations for this video
        max_frames: Maximum number of frames to keep
        
    Returns:
        Tuple of (trimmed_video, trimmed_annotations, start_idx, end_idx)
    """
    video_id = video['id']
    file_names = video['file_names']
    total_frames = len(file_names)
    
    # Find all frames with segmentations across all annotations for this video
    frames_with_seg = set()
    for ann in annotations:
        if ann['video_id'] == video_id:
            # Only include frames that actually have non-None segmentations
            for i, seg in enumerate(ann['segmentations']):
                if seg is not None:
                    frames_with_seg.add(i)
    
    if not frames_with_seg:
        # No segmentations found, keep first max_frames
        print(f"  Warning: Video {video_id} has no segmentations, keeping first {max_frames} frames")
        start_idx = 0
        end_idx = min(total_frames, max_frames)
    else:
        # Find the overall range of frames with segmentations
        seg_start = min(frames_with_seg)
        seg_end = max(frames_with_seg)
        
        # Center window around this range
        start_idx, end_idx = center_window_around_range(
            seg_start, seg_end, total_frames, max_frames
        )
    
    # Trim the video
    trimmed_file_names = file_names[start_idx:end_idx]
    trimmed_video = video.copy()
    trimmed_video['file_names'] = trimmed_file_names
    trimmed_video['length'] = len(trimmed_file_names)
    
    # Trim annotations
    trimmed_annotations = []
    for ann in annotations:
        if ann['video_id'] == video_id:
            trimmed_ann = ann.copy()
            
            # Trim segmentations, bboxes, and areas
            trimmed_ann['segmentations'] = ann['segmentations'][start_idx:end_idx]
            if 'bboxes' in ann and ann['bboxes']:
                trimmed_ann['bboxes'] = ann['bboxes'][start_idx:end_idx]
            if 'areas' in ann and ann['areas']:
                trimmed_ann['areas'] = ann['areas'][start_idx:end_idx]
            
            trimmed_ann['length'] = len(trimmed_ann['segmentations'])
            trimmed_annotations.append(trimmed_ann)
    
    return (trimmed_video, trimmed_annotations, start_idx, end_idx)


def main():
    parser = argparse.ArgumentParser(
        description='Trim val.json videos to max frames, centered around frames with segmentations'
    )
    parser.add_argument('input_json', type=str, help='Input val.json file path')
    parser.add_argument('output_json', type=str, help='Output val.json file path')
    parser.add_argument('--max-frames', type=int, default=300,
                       help='Maximum number of frames to keep per video (default: 300)')
    
    args = parser.parse_args()
    
    # Load input JSON
    print(f"Loading {args.input_json}...")
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    videos = data['videos']
    annotations = data['annotations']
    
    print(f"Found {len(videos)} videos and {len(annotations)} annotations")
    print(f"Trimming to max {args.max_frames} frames per video...")
    
    # Group annotations by video_id
    annotations_by_video = {}
    for ann in annotations:
        video_id = ann['video_id']
        if video_id not in annotations_by_video:
            annotations_by_video[video_id] = []
        annotations_by_video[video_id].append(ann)
    
    # Trim each video
    trimmed_videos = []
    trimmed_annotations = []
    stats = {
        'total_videos': len(videos),
        'trimmed_videos': 0,
        'unchanged_videos': 0,
        'total_frames_before': 0,
        'total_frames_after': 0
    }
    
    for video in videos:
        video_id = video['id']
        original_frames = len(video['file_names'])
        stats['total_frames_before'] += original_frames
        
        video_annotations = annotations_by_video.get(video_id, [])
        
        if original_frames <= args.max_frames:
            # Video is already short enough
            trimmed_videos.append(video)
            trimmed_annotations.extend(video_annotations)
            stats['unchanged_videos'] += 1
            stats['total_frames_after'] += original_frames
            print(f"Video {video_id}: {original_frames} frames (unchanged)")
        else:
            # Trim the video
            trimmed_video, trimmed_anns, start_idx, end_idx = trim_video_data(
                video, video_annotations, args.max_frames
            )
            trimmed_videos.append(trimmed_video)
            trimmed_annotations.extend(trimmed_anns)
            stats['trimmed_videos'] += 1
            trimmed_frames = len(trimmed_video['file_names'])
            stats['total_frames_after'] += trimmed_frames
            print(f"Video {video_id}: {original_frames} -> {trimmed_frames} frames "
                  f"(trimmed from {start_idx} to {end_idx})")
    
    # Create output data
    output_data = data.copy()
    output_data['videos'] = trimmed_videos
    output_data['annotations'] = trimmed_annotations
    
    # Save output JSON
    print(f"\nSaving to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print statistics
    print("\n" + "="*60)
    print("Statistics:")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Trimmed videos: {stats['trimmed_videos']}")
    print(f"  Unchanged videos: {stats['unchanged_videos']}")
    print(f"  Total frames before: {stats['total_frames_before']}")
    print(f"  Total frames after: {stats['total_frames_after']}")
    print(f"  Frames removed: {stats['total_frames_before'] - stats['total_frames_after']}")
    print("="*60)
    print(f"\nDone! Output saved to {args.output_json}")


if __name__ == '__main__':
    main()

