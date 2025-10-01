#!/usr/bin/env python3
"""
Script to convert a folder of images into a video with FPS matching the original video.

Usage:
    python convert_images_to_video.py <image_folder_path>

Example:
    python /home/simone/fish-dvis/visualization_scripts/convert_images_to_video.py "/home/simone/shared-data/fishway_ytvis/all_videos_mask/Ganaraska__Ganaraska 2024__11222024-11282024__24  11  22  14  07__32"

The script will:
1. Parse the folder name to find the corresponding original video
2. Extract FPS from the original video
3. Convert images to video with matching FPS
4. Save both videos to /home/simone/masked_unmasked_vids/
"""

import os
import sys
import re
import cv2
import argparse
from pathlib import Path
from typing import Optional, Tuple


def parse_folder_name(folder_path: str) -> Tuple[str, str, str, str, str]:
    """
    Parse the folder name to extract video path components.
    
    Expected format: "Ganaraska__Ganaraska 2024__11222024-11282024__24  11  22  14  07__32"
    
    Returns:
        Tuple of (location, year, date_range, time, video_number)
    """
    folder_name = os.path.basename(folder_path)
    
    # Split by double underscores
    parts = folder_name.split('__')
    
    if len(parts) != 5:
        raise ValueError(f"Invalid folder name format: {folder_name}. Expected 5 parts separated by '__'")
    
    location = parts[0]  # e.g., "Ganaraska"
    year_info = parts[1]  # e.g., "Ganaraska 2024"
    date_range = parts[2]  # e.g., "11222024-11282024"
    time = parts[3]  # e.g., "24  11  22  14  07"
    video_number = parts[4]  # e.g., "32"
    
    return location, year_info, date_range, time, video_number


def construct_video_path(location: str, year_info: str, date_range: str, time: str, video_number: str) -> str:
    """
    Construct the path to the original video file.
    
    Args:
        location: Location name (e.g., "Ganaraska")
        year_info: Year information (e.g., "Ganaraska 2024")
        date_range: Date range (e.g., "11222024-11282024")
        time: Time string (e.g., "24  11  22  14  07")
        video_number: Video number (e.g., "32")
    
    Returns:
        Path to the original video file
    """
    base_path = "/home/simone/shared-data/remora"
    video_path = os.path.join(base_path, location, year_info, date_range, time, f"{video_number}.mp4")
    return video_path


def get_video_fps(video_path: str) -> float:
    """
    Extract FPS from a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        FPS value as float
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        Exception: If FPS cannot be extracted
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps <= 0:
        raise Exception(f"Invalid FPS value ({fps}) extracted from video: {video_path}")
    
    return fps


def get_sorted_image_files(image_folder: str) -> list:
    """
    Get sorted list of image files from the folder.
    
    Args:
        image_folder: Path to folder containing images
    
    Returns:
        List of image file paths sorted by name
    """
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        raise Exception(f"No image files found in folder: {image_folder}")
    
    # Sort by filename (numerical order)
    image_files.sort(key=lambda x: os.path.basename(x))
    
    return image_files


def create_video_from_images(image_files: list, output_path: str, fps: float) -> None:
    """
    Create a video from a list of image files.
    
    Args:
        image_files: List of image file paths
        output_path: Output video file path
        fps: Frames per second for the video
    """
    if not image_files:
        raise ValueError("No image files provided")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        raise Exception(f"Could not read first image: {image_files[0]}")
    
    height, width, layers = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise Exception(f"Could not create video writer for: {output_path}")
    
    try:
        print(f"Processing {len(image_files)} images...")
        for i, image_path in enumerate(image_files):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}, skipping...")
                continue
            
            out.write(img)
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images...")
    
    finally:
        out.release()
    
    print(f"Video created successfully: {output_path}")


def copy_original_video(original_video_path: str, output_path: str) -> None:
    """
    Copy the original video to the output directory.
    
    Args:
        original_video_path: Path to original video
        output_path: Destination path
    """
    import shutil
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    shutil.copy2(original_video_path, output_path)
    print(f"Original video copied to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert images to video with matching FPS")
    parser.add_argument("image_folder", help="Path to folder containing images")
    parser.add_argument("--output-dir", default="/home/simone/masked_unmasked_vids",
                       help="Output directory for videos (default: /home/simone/masked_unmasked_vids)")
    
    args = parser.parse_args()
    
    try:
        # Parse folder name to get video path components
        print(f"Parsing folder name: {args.image_folder}")
        location, year_info, date_range, time, video_number = parse_folder_name(args.image_folder)
        
        # Construct original video path
        original_video_path = construct_video_path(location, year_info, date_range, time, video_number)
        print(f"Original video path: {original_video_path}")
        
        # Get FPS from original video
        fps = get_video_fps(original_video_path)
        print(f"Extracted FPS: {fps}")
        
        # Get sorted image files
        image_files = get_sorted_image_files(args.image_folder)
        print(f"Found {len(image_files)} image files")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate output filenames
        folder_name = os.path.basename(args.image_folder)
        masked_video_name = f"{folder_name}_masked.mp4"
        original_video_name = f"{folder_name}.mp4"
        
        masked_video_path = os.path.join(args.output_dir, masked_video_name)
        original_video_output_path = os.path.join(args.output_dir, original_video_name)
        
        # Create video from images
        print(f"Creating masked video: {masked_video_path}")
        create_video_from_images(image_files, masked_video_path, fps)
        
        # Copy original video
        print(f"Copying original video: {original_video_output_path}")
        copy_original_video(original_video_path, original_video_output_path)
        
        print("\nProcessing completed successfully!")
        print(f"Masked video: {masked_video_path}")
        print(f"Original video: {original_video_output_path}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
