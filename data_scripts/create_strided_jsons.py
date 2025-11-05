#!/usr/bin/env python3
"""
Create a stride-adjusted JSON file for proper evaluation when using frame stride.

This script creates a new JSON file (train or val) that only includes frames that would be
selected by the frame stride during inference. This ensures that ground truth annotations
match the predictions frame-for-frame during evaluation.
"""

import json
import os
import argparse
from typing import Dict, List, Any


def create_strided_json(
    input_json_path: str,
    output_json_path: str,
    frame_stride: int,
    verbose: bool = True
) -> bool:
    """
    Create a stride-adjusted JSON file.
    
    Args:
        input_json_path: Path to the original JSON file (train or val)
        output_json_path: Path where the stride-adjusted JSON will be saved
        frame_stride: Frame stride value (1 = every frame, 2 = every other frame, etc.)
        verbose: Whether to print progress information
        
    Returns:
        True if successful, False otherwise
    """
    if frame_stride == 1:
        if verbose:
            print("Frame stride is 1, no adjustment needed.")
        return True
    
    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON file not found: {input_json_path}")
        return False
    
    try:
        # Load the original JSON
        if verbose:
            print(f"Loading JSON from: {input_json_path}")
        with open(input_json_path, 'r') as f:
            json_data = json.load(f)
        
        # Create a copy for the stride-adjusted version
        stride_data = {
            'info': json_data['info'].copy() if 'info' in json_data else {},
            'licenses': json_data['licenses'].copy() if 'licenses' in json_data else [],
            'categories': json_data['categories'].copy() if 'categories' in json_data else [],
            'videos': [],
            'annotations': []
        }
        
        # Add metadata about the stride adjustment
        if 'info' not in stride_data:
            stride_data['info'] = {}
        stride_data['info']['description'] = f"Stride-adjusted set (stride={frame_stride})"
        stride_data['info']['original_json'] = input_json_path
        stride_data['info']['frame_stride'] = frame_stride
        
        total_original_frames = 0
        total_adjusted_frames = 0
        total_original_annotations = 0
        total_adjusted_annotations = 0
        
        # Process each video
        for video in json_data['videos']:
            video_id = video['id']
            video_length = video['length']
            
            # Calculate which frames to keep based on stride
            # This matches the logic in dataset_mapper.py line 314:
            # selected_idx = range(0, video_length, self.test_sampling_frame_stride)
            selected_frame_indices = list(range(0, video_length, frame_stride))
            
            total_original_frames += video_length
            total_adjusted_frames += len(selected_frame_indices)
            
            # Update video info
            adjusted_video = video.copy()
            adjusted_video['length'] = len(selected_frame_indices)
            
            # CRITICAL FIX: Also adjust the file_names list to match the stride
            if 'file_names' in video:
                original_file_names = video['file_names']
                adjusted_file_names = [original_file_names[i] for i in selected_frame_indices if i < len(original_file_names)]
                adjusted_video['file_names'] = adjusted_file_names
                
                if verbose:
                    print(f"Video {video_id}: {video_length} -> {len(selected_frame_indices)} frames, {len(original_file_names)} -> {len(adjusted_file_names)} file names (stride={frame_stride})")
            else:
                if verbose:
                    print(f"Video {video_id}: {video_length} -> {len(selected_frame_indices)} frames (stride={frame_stride})")
            
            stride_data['videos'].append(adjusted_video)
        
        # Process annotations - only keep annotations for frames that will be used
        video_frame_mapping = {}  # video_id -> set of frame_indices to keep
        for video in json_data['videos']:
            video_id = video['id']
            video_length = video['length']
            selected_frame_indices = set(range(0, video_length, frame_stride))
            video_frame_mapping[video_id] = selected_frame_indices
        
        # Filter annotations
        for annotation in json_data['annotations']:
            video_id = annotation['video_id']
            
            if video_id not in video_frame_mapping:
                continue
            
            selected_frames = video_frame_mapping[video_id]
            
            # Filter segmentations to only include selected frames
            original_segmentations = annotation.get('segmentations', [])
            adjusted_segmentations = []
            
            for frame_idx, segmentation in enumerate(original_segmentations):
                if frame_idx in selected_frames:
                    adjusted_segmentations.append(segmentation)
            
            # Also filter bboxes and areas if they exist
            original_bboxes = annotation.get('bboxes', [])
            adjusted_bboxes = []
            if original_bboxes:
                for frame_idx, bbox in enumerate(original_bboxes):
                    if frame_idx in selected_frames:
                        adjusted_bboxes.append(bbox)
            
            original_areas = annotation.get('areas', [])
            adjusted_areas = []
            if original_areas:
                for frame_idx, area in enumerate(original_areas):
                    if frame_idx in selected_frames:
                        adjusted_areas.append(area)
            
            total_original_annotations += len(original_segmentations)
            total_adjusted_annotations += len(adjusted_segmentations)
            
            # Only keep annotation if it has segmentations in the selected frames
            if adjusted_segmentations:
                adjusted_annotation = annotation.copy()
                adjusted_annotation['segmentations'] = adjusted_segmentations
                if adjusted_bboxes:
                    adjusted_annotation['bboxes'] = adjusted_bboxes
                if adjusted_areas:
                    adjusted_annotation['areas'] = adjusted_areas
                
                # CRITICAL FIX: Update the length field to match the stride-adjusted count
                if 'length' in adjusted_annotation:
                    adjusted_annotation['length'] = len(adjusted_segmentations)
                
                stride_data['annotations'].append(adjusted_annotation)
        
        # Save the stride-adjusted JSON
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        if verbose:
            print(f"Saving stride-adjusted JSON to: {output_json_path}")
        
        with open(output_json_path, 'w') as f:
            json.dump(stride_data, f, indent=2)
        
        if verbose:
            print(f"\n✓ Successfully created stride-adjusted JSON:")
            print(f"  Original frames: {total_original_frames}")
            print(f"  Adjusted frames: {total_adjusted_frames}")
            print(f"  Frame reduction: {(1 - total_adjusted_frames/total_original_frames)*100:.1f}%")
            print(f"  Original annotations: {total_original_annotations}")
            print(f"  Adjusted annotations: {total_adjusted_annotations}")
            print(f"  Videos: {len(stride_data['videos'])}")
            print(f"  Annotation objects: {len(stride_data['annotations'])}")
        
        return True
        
    except Exception as e:
        print(f"Error creating stride-adjusted JSON: {e}")
        return False


def get_strided_json_path(original_json_path: str, frame_stride: int) -> str:
    """
    Get the path where the stride-adjusted JSON should be located.
    
    Args:
        original_json_path: Path to the original JSON file
        frame_stride: Frame stride value
        
    Returns:
        Path where the stride-adjusted JSON should be located
    """
    if frame_stride == 1:
        return original_json_path
    
    base_path, ext = os.path.splitext(original_json_path)
    return f"{base_path}_stride{frame_stride}{ext}"


def main():
    parser = argparse.ArgumentParser(description="Create stride-adjusted JSON file (train or val)")
    parser.add_argument('--input-json', type=str, required=True,
                       help='Input JSON file (train or val)')
    parser.add_argument('--output-json', type=str,
                       help='Output stride-adjusted JSON file (default: input_json with _stride{N} suffix)')
    parser.add_argument('--frame-stride', type=int, required=True,
                       help='Frame stride value (1=every frame, 2=every other frame, etc.)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print progress information')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output_json:
        args.output_json = get_strided_json_path(args.input_json, args.frame_stride)
    
    print(f"Creating stride-adjusted JSON:")
    print(f"  Input: {args.input_json}")
    print(f"  Output: {args.output_json}")
    print(f"  Frame stride: {args.frame_stride}")
    
    success = create_strided_json(
        args.input_json,
        args.output_json,
        args.frame_stride,
        args.verbose
    )
    
    if success:
        print(f"\n✓ Success! Stride-adjusted JSON created at: {args.output_json}")
    else:
        print("\n✗ Failed to create stride-adjusted JSON")
        exit(1)


if __name__ == "__main__":
    main()
