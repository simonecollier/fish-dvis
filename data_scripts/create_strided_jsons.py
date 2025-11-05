#!/usr/bin/env python3
"""
Create a stride-adjusted validation JSON file for proper evaluation when using frame stride.

This script creates a new validation JSON file that only includes frames that would be
selected by the frame stride during inference. This ensures that ground truth annotations
match the predictions frame-for-frame during evaluation.
"""

import json
import os
import argparse
from typing import Dict, List, Any


def create_stride_adjusted_val_json(
    input_json_path: str,
    output_json_path: str,
    frame_stride: int,
    verbose: bool = True
) -> bool:
    """
    Create a stride-adjusted validation JSON file.
    
    Args:
        input_json_path: Path to the original validation JSON file
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
        # Load the original validation JSON
        if verbose:
            print(f"Loading validation JSON from: {input_json_path}")
        with open(input_json_path, 'r') as f:
            val_data = json.load(f)
        
        # Create a copy for the stride-adjusted version
        stride_data = {
            'info': val_data['info'].copy() if 'info' in val_data else {},
            'licenses': val_data['licenses'].copy() if 'licenses' in val_data else [],
            'categories': val_data['categories'].copy() if 'categories' in val_data else [],
            'videos': [],
            'annotations': []
        }
        
        # Add metadata about the stride adjustment
        if 'info' not in stride_data:
            stride_data['info'] = {}
        stride_data['info']['description'] = f"Stride-adjusted validation set (stride={frame_stride})"
        stride_data['info']['original_json'] = input_json_path
        stride_data['info']['frame_stride'] = frame_stride
        
        total_original_frames = 0
        total_adjusted_frames = 0
        total_original_annotations = 0
        total_adjusted_annotations = 0
        
        # Process each video
        for video in val_data['videos']:
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
        for video in val_data['videos']:
            video_id = video['id']
            video_length = video['length']
            selected_frame_indices = set(range(0, video_length, frame_stride))
            video_frame_mapping[video_id] = selected_frame_indices
        
        # Filter annotations
        for annotation in val_data['annotations']:
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
            print(f"\n✓ Successfully created stride-adjusted validation JSON:")
            print(f"  Original frames: {total_original_frames}")
            print(f"  Adjusted frames: {total_adjusted_frames}")
            print(f"  Frame reduction: {(1 - total_adjusted_frames/total_original_frames)*100:.1f}%")
            print(f"  Original annotations: {total_original_annotations}")
            print(f"  Adjusted annotations: {total_adjusted_annotations}")
            print(f"  Videos: {len(stride_data['videos'])}")
            print(f"  Annotation objects: {len(stride_data['annotations'])}")
        
        return True
        
    except Exception as e:
        print(f"Error creating stride-adjusted validation JSON: {e}")
        return False


def get_stride_adjusted_json_path(original_json_path: str, frame_stride: int) -> str:
    """
    Get the path where the stride-adjusted JSON should be located.
    
    Args:
        original_json_path: Path to the original validation JSON
        frame_stride: Frame stride value
        
    Returns:
        Path where the stride-adjusted JSON should be located
    """
    if frame_stride == 1:
        return original_json_path
    
    base_path, ext = os.path.splitext(original_json_path)
    return f"{base_path}_stride{frame_stride}{ext}"


def ensure_stride_adjusted_json(
    original_json_path: str, 
    frame_stride: int,
    model_output_dir: str = None,
    verbose: bool = True
) -> str:
    """
    Ensure that a stride-adjusted validation JSON exists when needed.
    
    Args:
        original_json_path: Path to the original validation JSON
        frame_stride: Frame stride value from config
        model_output_dir: Model output directory (only used for initial training, not for copying during eval)
        verbose: Whether to print progress information
        
    Returns:
        Path to the JSON file that should be used (original or stride-adjusted)
    """
    
    # If stride is 1, no adjustment needed
    if frame_stride == 1:
        if verbose:
            print("Frame stride is 1, using original validation JSON")
        return original_json_path
    
    # Determine where the stride-adjusted JSON should be
    stride_json_path = get_stride_adjusted_json_path(original_json_path, frame_stride)
    
    # Check if stride-adjusted JSON already exists in model directory FIRST
    # This prevents unnecessary copying during checkpoint evaluations
    if model_output_dir and os.path.exists(model_output_dir):
        model_stride_json = os.path.join(model_output_dir, f"val_stride{frame_stride}.json")
        if os.path.exists(model_stride_json):
            if verbose:
                print(f"✓ Using existing stride-adjusted JSON from model directory: {model_stride_json}")
                print(f"  → This JSON has pre-applied stride {frame_stride}, data loader stride will be set to 1")
            return model_stride_json
    
    # Check if stride-adjusted JSON exists in the data directory
    if os.path.exists(stride_json_path):
        if verbose:
            print(f"✓ Using existing stride-adjusted JSON: {stride_json_path}")
        
        # Only copy to model directory during TRAINING (when it doesn't already exist there)
        # This prevents copying during checkpoint evaluations
        # Don't copy if the output directory looks like a checkpoint evaluation directory
        is_checkpoint_eval_dir = (model_output_dir and 
                                 ('checkpoint_evaluations' in model_output_dir or 
                                  'checkpoint_' in os.path.basename(model_output_dir)))
        
        if (model_output_dir and os.path.exists(model_output_dir) and 
            not os.path.exists(os.path.join(model_output_dir, f"val_stride{frame_stride}.json")) and
            not is_checkpoint_eval_dir):
            model_stride_json = os.path.join(model_output_dir, f"val_stride{frame_stride}.json")
            try:
                import shutil
                shutil.copy2(stride_json_path, model_stride_json)
                if verbose:
                    print(f"✓ Copied stride-adjusted JSON to model directory: {model_stride_json}")
                return model_stride_json
            except Exception as e:
                if verbose:
                    print(f"⚠ Failed to copy to model directory: {e}")
        elif is_checkpoint_eval_dir and verbose:
            print("✓ Checkpoint evaluation detected - using existing stride JSON without copying")
        
        return stride_json_path
    
    # Need to create stride-adjusted JSON
    if verbose:
        print(f"Creating stride-adjusted validation JSON (stride={frame_stride})...")
    
    success = create_stride_adjusted_val_json(
        input_json_path=original_json_path,
        output_json_path=stride_json_path,
        frame_stride=frame_stride,
        verbose=verbose
    )
    
    if not success:
        if verbose:
            print(f"⚠ Failed to create stride-adjusted JSON, using original: {original_json_path}")
        return original_json_path
    
    if verbose:
        print(f"✓ Created stride-adjusted JSON: {stride_json_path}")
    
    # Copy to model directory if specified and it doesn't already exist
    # Don't copy if the output directory looks like a checkpoint evaluation directory
    is_checkpoint_eval_dir = (model_output_dir and 
                             ('checkpoint_evaluations' in model_output_dir or 
                              'checkpoint_' in os.path.basename(model_output_dir)))
    
    if (model_output_dir and os.path.exists(model_output_dir) and 
        not os.path.exists(os.path.join(model_output_dir, f"val_stride{frame_stride}.json")) and
        not is_checkpoint_eval_dir):
        model_stride_json = os.path.join(model_output_dir, f"val_stride{frame_stride}.json")
        try:
            import shutil
            shutil.copy2(stride_json_path, model_stride_json)
            if verbose:
                print(f"✓ Copied stride-adjusted JSON to model directory: {model_stride_json}")
            return model_stride_json
        except Exception as e:
            if verbose:
                print(f"⚠ Failed to copy to model directory: {e}")
    elif is_checkpoint_eval_dir and verbose:
        print("✓ Checkpoint evaluation detected - using stride JSON without copying")
    
    return stride_json_path


def main():
    parser = argparse.ArgumentParser(description="Create stride-adjusted validation JSON")
    parser.add_argument('--input-json', type=str, 
                       default='/data/fishway_ytvis/val.json',
                       help='Input validation JSON file')
    parser.add_argument('--output-json', type=str,
                       help='Output stride-adjusted JSON file (default: input_json with _stride{N} suffix)')
    parser.add_argument('--frame-stride', type=int, required=True,
                       help='Frame stride value (1=every frame, 2=every other frame, etc.)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print progress information')
    
    # Support for ensure mode
    parser.add_argument('--ensure', action='store_true',
                       help='Ensure stride-adjusted JSON exists (create only if needed)')
    parser.add_argument('--model-dir', type=str,
                       help='Model output directory (for ensure mode)')
    
    args = parser.parse_args()
    
    if args.ensure:
        # Use ensure mode
        result_path = ensure_stride_adjusted_json(
            original_json_path=args.input_json,
            frame_stride=args.frame_stride,
            model_output_dir=args.model_dir,
            verbose=args.verbose
        )
        print(f"\nResult: Use JSON file at {result_path}")
    else:
        # Use create mode (original functionality)
        # Generate output filename if not provided
        if not args.output_json:
            base_path, ext = os.path.splitext(args.input_json)
            args.output_json = f"{base_path}_stride{args.frame_stride}{ext}"
        
        print(f"Creating stride-adjusted validation JSON:")
        print(f"  Input: {args.input_json}")
        print(f"  Output: {args.output_json}")
        print(f"  Frame stride: {args.frame_stride}")
        
        success = create_stride_adjusted_val_json(
            args.input_json,
            args.output_json,
            args.frame_stride,
            args.verbose
        )
        
        if success:
            print(f"\n✓ Success! Stride-adjusted validation JSON created at: {args.output_json}")
        else:
            print("\n✗ Failed to create stride-adjusted validation JSON")
            exit(1)


if __name__ == "__main__":
    main()
