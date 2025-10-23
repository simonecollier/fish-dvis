#!/usr/bin/env python3
"""
Script to visualize videos from different stride datasets with segmentations and labels overlaid.
This helps verify that the stride sampling is working correctly and the data looks good.

Usage:
    python /home/simone/fish-dvis/visualization_scripts/visualize_stride_videos.py --stride 1 --max-videos 300 --output-dir /store/simone/data_viz
"""

import json
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import colorsys

# Try to import pycocotools for RLE decoding
try:
    from pycocotools import mask as coco_mask
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Segmentation masks will not be displayed.")

def generate_colors(n_colors):
    """Generate n distinct colors for different categories."""
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([int(c * 255) for c in rgb])
    return colors

def draw_segmentation_mask(image, segmentation, color, alpha=0.3):
    """Draw segmentation mask outline on image (no fill)."""
    if segmentation is None:
        return image
    
    try:
        # Handle COCO RLE format
        if isinstance(segmentation, dict) and 'size' in segmentation and 'counts' in segmentation:
            if PYCOCOTOOLS_AVAILABLE:
                # Decode RLE to binary mask
                rle = segmentation
                binary_mask = coco_mask.decode(rle)
                
                # Ensure binary mask is uint8
                binary_mask = binary_mask.astype(np.uint8)
                
                # Get 2D mask for contour detection
                if len(binary_mask.shape) == 3:
                    mask_2d = binary_mask[:, :, 0]
                else:
                    mask_2d = binary_mask
                
                # Find and draw contours only (no fill)
                contours, _ = cv2.findContours(mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)
                
                return image
            else:
                print("Warning: pycocotools not available, skipping RLE segmentation")
                return image
        
        # Handle polygon format
        elif isinstance(segmentation, list):
            # Check if it's a simple list of coordinates
            if len(segmentation) >= 4 and len(segmentation) % 2 == 0:
                # Flatten the list and reshape to (N, 2) for polygon points
                points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                
                # Draw polygon outline only (no fill)
                cv2.polylines(image, [points], True, color, 2)
                
                return image
            else:
                # Invalid segmentation format
                return image
        else:
            # Try to convert to numpy array
            points = np.array(segmentation).reshape(-1, 2).astype(np.int32)
            
            # Draw polygon outline only (no fill)
            cv2.polylines(image, [points], True, color, 2)
            
            return image
            
    except Exception as e:
        print(f"Warning: Could not draw segmentation: {e}")
        return image

def draw_bbox(image, bbox, color, label, thickness=2):
    """Draw bounding box and label on image."""
    if bbox is None:
        return image
    
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Draw bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(image, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
    
    # Draw label text
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

def load_video_frames(video_path, frame_names):
    """Load frames from video file based on frame names."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return frames
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame is in our frame_names list
        if frame_count < len(frame_names):
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    return frames

def create_video_visualization(video_data, annotations, categories, output_path, stride, fps=10, base_path="/data/fishway_ytvis/all_videos"):
    """Create a video with segmentations and labels overlaid."""
    
    # Get video info
    video_id = video_data['id']
    file_names = video_data['file_names']
    video_length = video_data['length']
    
    # Convert relative paths to absolute paths
    file_names = [os.path.join(base_path, fname) for fname in file_names]
    
    print(f"Processing video {video_id} with {video_length} frames (stride {stride})")
    
    # Create category ID to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # Generate colors for categories
    colors = generate_colors(len(categories))
    cat_id_to_color = {cat['id']: colors[i] for i, cat in enumerate(categories)}
    
    # Group annotations by frame
    frame_annotations = defaultdict(list)
    for ann in annotations:
        if ann['video_id'] == video_id:
            for frame_idx, segmentation in enumerate(ann['segmentations']):
                if segmentation is not None:
                    frame_annotations[frame_idx].append({
                        'category_id': ann['category_id'],
                        'segmentation': segmentation,
                        'bbox': ann['bboxes'][frame_idx] if frame_idx < len(ann['bboxes']) else None,
                        'area': ann['areas'][frame_idx] if frame_idx < len(ann['areas']) else None
                    })
    
    # Create output video
    if not file_names:
        print(f"Warning: No frames found for video {video_id}")
        return
    
    # Try to load the first frame to get dimensions
    first_frame_path = file_names[0] if file_names else None
    if not first_frame_path or not os.path.exists(first_frame_path):
        print(f"Warning: First frame not found: {first_frame_path}")
        return
    
    # Load first frame to get dimensions
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Warning: Could not load first frame: {first_frame_path}")
        return
    
    height, width, _ = first_frame.shape
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_idx, frame_name in enumerate(file_names):
        if not os.path.exists(frame_name):
            print(f"Warning: Frame not found: {frame_name}")
            continue
        
        # Load frame
        frame = cv2.imread(frame_name)
        if frame is None:
            print(f"Warning: Could not load frame: {frame_name}")
            continue
        
        # Add frame info text
        info_text = f"Video {video_id} | Frame {frame_idx+1}/{len(file_names)} | Stride {stride}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add FPS info
        actual_fps = fps / stride if stride > 1 else fps
        fps_text = f"Effective FPS: {actual_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw annotations for this frame
        if frame_idx in frame_annotations:
            for ann in frame_annotations[frame_idx]:
                category_id = ann['category_id']
                category_name = cat_id_to_name.get(category_id, f"Unknown_{category_id}")
                color = cat_id_to_color.get(category_id, [255, 255, 255])
                
                # Draw segmentation
                if ann['segmentation'] is not None:
                    frame = draw_segmentation_mask(frame, ann['segmentation'], color)
                
                # Draw bounding box and label
                if ann['bbox'] is not None:
                    frame = draw_bbox(frame, ann['bbox'], color, category_name)
        
        # Write frame to output video
        out.write(frame)
    
    out.release()
    print(f"Saved visualization to: {output_path}")

def visualize_stride_dataset(stride, max_videos=5, output_dir="./stride_visualizations"):
    """Visualize videos from a specific stride dataset."""
    
    # Paths
    train_json_path = f"/data/fishway_ytvis/train_stride{stride}.json"
    val_json_path = f"/data/fishway_ytvis/val_stride{stride}.json"
    
    # Check if files exist
    if not os.path.exists(train_json_path):
        print(f"Error: Training JSON not found: {train_json_path}")
        return
    
    if not os.path.exists(val_json_path):
        print(f"Error: Validation JSON not found: {val_json_path}")
        return
    
    # Create output directories
    stride_output_dir = os.path.join(output_dir, f"stride_{stride}")
    train_output_dir = os.path.join(stride_output_dir, "train")
    val_output_dir = os.path.join(stride_output_dir, "val")
    
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    
    # Load datasets
    print(f"Loading stride {stride} datasets...")
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    print(f"Train: {len(train_data['videos'])} videos, {len(train_data['annotations'])} annotations")
    print(f"Val: {len(val_data['videos'])} videos, {len(val_data['annotations'])} annotations")
    
    # Process training videos
    print(f"\nProcessing training videos (max {max_videos})...")
    train_videos = random.sample(train_data['videos'], min(max_videos, len(train_data['videos'])))
    
    for video in train_videos:
        output_path = os.path.join(train_output_dir, f"video_{video['id']}_stride{stride}.mp4")
        create_video_visualization(
            video, 
            train_data['annotations'], 
            train_data['categories'], 
            output_path, 
            stride,
            base_path="/data/fishway_ytvis/all_videos"
        )
    
    # Process validation videos
    print(f"\nProcessing validation videos (max {max_videos})...")
    val_videos = random.sample(val_data['videos'], min(max_videos, len(val_data['videos'])))
    
    for video in val_videos:
        output_path = os.path.join(val_output_dir, f"video_{video['id']}_stride{stride}.mp4")
        create_video_visualization(
            video, 
            val_data['annotations'], 
            val_data['categories'], 
            output_path, 
            stride,
            base_path="/data/fishway_ytvis/all_videos"
        )
    
    print(f"\nCompleted stride {stride} visualization!")
    print(f"Output directory: {stride_output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize stride datasets with segmentations and labels')
    parser.add_argument('--stride', type=int, required=True, 
                       help='Stride level to visualize (1, 2, 3, 4, 5, 6)')
    parser.add_argument('--max-videos', type=int, default=5,
                       help='Maximum number of videos to process per dataset (default: 5)')
    parser.add_argument('--output-dir', type=str, default='./stride_visualizations',
                       help='Output directory for visualizations (default: ./stride_visualizations)')
    parser.add_argument('--all-strides', action='store_true',
                       help='Process all stride levels (1-6)')
    
    args = parser.parse_args()
    
    if args.all_strides:
        print("Processing all stride levels (1-6)...")
        for stride in range(1, 7):
            print(f"\n{'='*60}")
            print(f"PROCESSING STRIDE {stride}")
            print(f"{'='*60}")
            visualize_stride_dataset(stride, args.max_videos, args.output_dir)
    else:
        if args.stride not in range(1, 7):
            print("Error: Stride must be between 1 and 6")
            return
        
        print(f"Processing stride {args.stride}...")
        visualize_stride_dataset(args.stride, args.max_videos, args.output_dir)

if __name__ == "__main__":
    main()
