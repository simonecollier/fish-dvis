#!/usr/bin/env python3
"""
Create side-by-side GIF comparing two videos or video frames.

This script creates a GIF that shows two videos side by side:
- Left: First video/frames
- Right: Second video/frames

Features:
- Support for two video files OR frame directories
- Configurable frame rate (with frame skipping for smaller file sizes)
- Configurable output resolution
- Automatic frame selection from JSON metadata (directory mode)
- Progress tracking and file size optimization
- Built-in presets for common use cases

Usage Examples:
    # Video mode: Create GIF from two video files
    python3 create_side_by_side_gif.py --video1 video1.mp4 --video2 video2.mp4 --preset web
    python3 create_side_by_side_gif.py --video1 video1.mp4 --video2 video2.mp4 --label1 "Original" --label2 "Processed" --output-fps 5
    
    # Directory mode: Create GIF from frame directories (legacy)
    python3 create_side_by_side_gif.py --preset small      # Small file (2-5 MB)
    python3 create_side_by_side_gif.py --preset web        # Web optimized (5-10 MB)
    python3 create_side_by_side_gif.py --preset high       # High quality (20-40 MB)
    
    # Custom parameters
    python3 create_side_by_side_gif.py --output-fps 3 --width 640 --height 240 --max-frames 30
    
    # Show all presets
    python3 create_side_by_side_gif.py --list-presets
"""

import json
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import sys
import cv2

def load_video_frames(json_path, video_id):
    """Load frame information for a specific video from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for video in data['videos']:
        if video['id'] == video_id:
            return video['file_names']
    
    raise ValueError(f"Video ID {video_id} not found in {json_path}")

def extract_frames_from_video(video_path, start_frame=0, end_frame=None):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        start_frame: Starting frame index (0-based)
        end_frame: Ending frame index (None for end of video)
    
    Returns:
        List of PIL Images, fps
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Validate start_frame
    if start_frame < 0:
        start_frame = 0
    if start_frame >= frame_count:
        raise ValueError(f"Start frame {start_frame} is beyond video length ({frame_count} frames)")
    
    # Set end_frame if not specified
    if end_frame is None:
        end_frame = frame_count
    else:
        end_frame = min(end_frame, frame_count)
    
    if end_frame <= start_frame:
        raise ValueError(f"End frame {end_frame} must be greater than start frame {start_frame}")
    
    print(f"Video info: {frame_count} frames, {fps:.2f} FPS")
    print(f"Extracting frames {start_frame} to {end_frame-1} (inclusive)")
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames, fps

def create_side_by_side_frame(original_path, mask_path, output_width=None, output_height=None):
    """Create a side-by-side frame from original and mask images."""
    try:
        # Load images
        original_img = Image.open(original_path)
        mask_img = Image.open(mask_path)
        
        # Get original dimensions
        orig_width, orig_height = original_img.size
        mask_width, mask_height = mask_img.size
        
        # Calculate output dimensions
        if output_width is None:
            output_width = orig_width + mask_width
        if output_height is None:
            output_height = max(orig_height, mask_height)
        
        # Resize images to fit output dimensions
        # Each side gets half the width
        side_width = output_width // 2
        side_height = output_height
        
        # Resize maintaining aspect ratio
        orig_ratio = orig_width / orig_height
        mask_ratio = mask_width / mask_height
        
        # Calculate new dimensions for each side
        orig_new_width = min(side_width, int(side_height * orig_ratio))
        orig_new_height = min(side_height, int(side_width / orig_ratio))
        
        mask_new_width = min(side_width, int(side_height * mask_ratio))
        mask_new_height = min(side_height, int(side_width / mask_ratio))
        
        # Resize images
        original_resized = original_img.resize((orig_new_width, orig_new_height), Image.Resampling.LANCZOS)
        mask_resized = mask_img.resize((mask_new_width, mask_new_height), Image.Resampling.LANCZOS)
        
        # Create output image
        output_img = Image.new('RGB', (output_width, output_height), (0, 0, 0))
        
        # Calculate positions to center images
        orig_x = (side_width - orig_new_width) // 2
        orig_y = (side_height - orig_new_height) // 2
        
        mask_x = side_width + (side_width - mask_new_width) // 2
        mask_y = (side_height - mask_new_height) // 2
        
        # Paste images
        output_img.paste(original_resized, (orig_x, orig_y))
        output_img.paste(mask_resized, (mask_x, mask_y))
        
        # Add labels
        draw = ImageDraw.Draw(output_img)
        try:
            # Try to use a default font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Add text labels
        if font:
            draw.text((10, 10), "Camera", fill=(255, 255, 255), font=font)
            draw.text((side_width + 10, 10), "Silhouette", fill=(255, 255, 255), font=font)
        
        return output_img
        
    except Exception as e:
        print(f"Error creating side-by-side frame: {e}")
        return None

def create_gif_from_videos(video1_path, video2_path, output_path,
                           target_fps=10, output_fps=5, output_width=1280, output_height=480,
                           start_frame=0, end_frame=None, max_frames=None, quality=85, 
                           label1=None, label2=None):
    """
    Create a side-by-side GIF from two video files.
    
    Args:
        video1_path: Path to first video file
        video2_path: Path to second video file
        output_path: Output GIF path
        target_fps: Original video frame rate (will be detected if None)
        output_fps: Desired output frame rate
        output_width: Output GIF width
        output_height: Output GIF height
        start_frame: Starting frame index (0-based, applies to both videos)
        end_frame: Ending frame index (None for end of video, applies to both videos)
        max_frames: Maximum number of frames (None for all)
        quality: GIF quality (1-100)
        label1: Label for first video (None to omit)
        label2: Label for second video (None to omit)
    """
    print("Extracting frames from videos...")
    
    # Extract frames from both videos with same start/end frame range
    frames1, fps1 = extract_frames_from_video(video1_path, start_frame, end_frame)
    frames2, fps2 = extract_frames_from_video(video2_path, start_frame, end_frame)
    
    # Use detected FPS if target_fps not specified
    if target_fps is None:
        target_fps = min(fps1, fps2)
    
    # Ensure both videos have the same number of frames (use minimum)
    min_frames = min(len(frames1), len(frames2))
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]
    
    print(f"Extracted {len(frames1)} frames from video 1, {len(frames2)} frames from video 2")
    print(f"Using {min_frames} frames (minimum of both videos)")
    
    # Calculate frame skipping
    frame_skip = max(1, int(target_fps / output_fps))
    print(f"Frame skip: {frame_skip} (using every {frame_skip}th frame)")
    
    # Select frames to use
    selected_indices = list(range(0, min_frames, frame_skip))
    
    # Limit frames if specified
    if max_frames and len(selected_indices) > max_frames:
        # Evenly sample frames
        step = len(selected_indices) // max_frames
        selected_indices = selected_indices[::step][:max_frames]
    
    print(f"Using {len(selected_indices)} frames out of {min_frames} total frames")
    
    # Create combined frames
    combined_frames = []
    for i, idx in enumerate(selected_indices):
        print(f"Processing frame {i+1}/{len(selected_indices)} (frame {idx+1})")
        
        # Create side-by-side frame from PIL Images
        combined_frame = create_side_by_side_frame_from_images(
            frames1[idx], frames2[idx], output_width, output_height, label1, label2
        )
        
        if combined_frame:
            combined_frames.append(combined_frame)
        else:
            print(f"Failed to create frame {idx+1}")
    
    if not combined_frames:
        print("No frames were successfully created!")
        return False
    
    print(f"Creating GIF with {len(combined_frames)} frames...")
    
    # Calculate duration (in milliseconds)
    duration = int(1000 / output_fps)
    
    # Save GIF
    try:
        combined_frames[0].save(
            output_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
            quality=quality
        )
        
        # Get file size
        file_size = os.path.getsize(output_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"GIF created successfully!")
        print(f"Output: {output_path}")
        print(f"Frames: {len(combined_frames)}")
        print(f"Duration: {len(combined_frames) / output_fps:.1f} seconds")
        print(f"File size: {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error saving GIF: {e}")
        return False

def create_side_by_side_frame_from_images(original_img, mask_img, output_width=None, output_height=None, label1=None, label2=None):
    """Create a side-by-side frame from two PIL Images."""
    try:
        # Get original dimensions
        orig_width, orig_height = original_img.size
        mask_width, mask_height = mask_img.size
        
        # Calculate output dimensions
        if output_width is None:
            output_width = orig_width + mask_width
        if output_height is None:
            output_height = max(orig_height, mask_height)
        
        # Resize images to fit output dimensions
        # Each side gets half the width
        side_width = output_width // 2
        side_height = output_height
        
        # Resize maintaining aspect ratio
        orig_ratio = orig_width / orig_height
        mask_ratio = mask_width / mask_height
        
        # Calculate new dimensions for each side
        orig_new_width = min(side_width, int(side_height * orig_ratio))
        orig_new_height = min(side_height, int(side_width / orig_ratio))
        
        mask_new_width = min(side_width, int(side_height * mask_ratio))
        mask_new_height = min(side_height, int(side_width / mask_ratio))
        
        # Resize images
        original_resized = original_img.resize((orig_new_width, orig_new_height), Image.Resampling.LANCZOS)
        mask_resized = mask_img.resize((mask_new_width, mask_new_height), Image.Resampling.LANCZOS)
        
        # Create output image
        output_img = Image.new('RGB', (output_width, output_height), (0, 0, 0))
        
        # Calculate positions to center images
        orig_x = (side_width - orig_new_width) // 2
        orig_y = (side_height - orig_new_height) // 2
        
        mask_x = side_width + (side_width - mask_new_width) // 2
        mask_y = (side_height - mask_new_height) // 2
        
        # Paste images
        output_img.paste(original_resized, (orig_x, orig_y))
        output_img.paste(mask_resized, (mask_x, mask_y))
        
        # Add labels only if provided
        if label1 or label2:
            draw = ImageDraw.Draw(output_img)
            try:
                # Try to use a default font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Add text labels
            if font:
                if label1:
                    draw.text((10, 10), label1, fill=(255, 255, 255), font=font)
                if label2:
                    draw.text((side_width + 10, 10), label2, fill=(255, 255, 255), font=font)
        
        return output_img
        
    except Exception as e:
        print(f"Error creating side-by-side frame: {e}")
        return None

def create_gif(original_dir, mask_dir, frame_list, output_path, 
               target_fps=10, output_fps=5, output_width=1280, output_height=480,
               max_frames=None, quality=85):
    """
    Create a side-by-side GIF from frame directories.
    
    Args:
        original_dir: Directory containing original frames
        mask_dir: Directory containing mask frames
        frame_list: List of frame filenames
        output_path: Output GIF path
        target_fps: Original video frame rate
        output_fps: Desired output frame rate
        output_width: Output GIF width
        output_height: Output GIF height
        max_frames: Maximum number of frames (None for all)
        quality: GIF quality (1-100)
    """
    
    # Calculate frame skipping
    frame_skip = max(1, int(target_fps / output_fps))
    print(f"Frame skip: {frame_skip} (using every {frame_skip}th frame)")
    
    # Select frames to use
    selected_frames = frame_list[::frame_skip]
    
    # Limit frames if specified
    if max_frames and len(selected_frames) > max_frames:
        # Evenly sample frames
        step = len(selected_frames) // max_frames
        selected_frames = selected_frames[::step][:max_frames]
    
    print(f"Using {len(selected_frames)} frames out of {len(frame_list)} total frames")
    
    # Create frames
    frames = []
    for i, frame_name in enumerate(selected_frames):
        print(f"Processing frame {i+1}/{len(selected_frames)}: {frame_name}")
        
        # Construct paths
        original_path = os.path.join(original_dir, frame_name)
        mask_path = os.path.join(mask_dir, frame_name)
        
        # Check if files exist
        if not os.path.exists(original_path):
            print(f"Warning: Original frame not found: {original_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"Warning: Mask frame not found: {mask_path}")
            continue
        
        # Create side-by-side frame
        combined_frame = create_side_by_side_frame(
            original_path, mask_path, output_width, output_height
        )
        
        if combined_frame:
            frames.append(combined_frame)
        else:
            print(f"Failed to create frame for {frame_name}")
    
    if not frames:
        print("No frames were successfully created!")
        return False
    
    print(f"Creating GIF with {len(frames)} frames...")
    
    # Calculate duration (in milliseconds)
    duration = int(1000 / output_fps)
    
    # Save GIF
    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
            quality=quality
        )
        
        # Get file size
        file_size = os.path.getsize(output_path)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"GIF created successfully!")
        print(f"Output: {output_path}")
        print(f"Frames: {len(frames)}")
        print(f"Duration: {len(frames) / output_fps:.1f} seconds")
        print(f"File size: {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error saving GIF: {e}")
        return False

def get_presets():
    """Return dictionary of preset configurations."""
    return {
        'small': {
            'description': 'Small file size (2-5 MB)',
            'target_fps': 10, 'output_fps': 3, 'width': 640, 'height': 240,
            'max_frames': 30, 'quality': 60
        },
        'medium': {
            'description': 'Balanced quality (8-15 MB)',
            'target_fps': 10, 'output_fps': 5, 'width': 1280, 'height': 480,
            'max_frames': 60, 'quality': 75
        },
        'high': {
            'description': 'High quality (20-40 MB)',
            'target_fps': 10, 'output_fps': 8, 'width': 1920, 'height': 720,
            'max_frames': 100, 'quality': 90
        },
        'full': {
            'description': 'Full quality (50-100 MB)',
            'target_fps': 10, 'output_fps': 10, 'width': 2560, 'height': 960,
            'max_frames': None, 'quality': 95
        },
        'web': {
            'description': 'Web optimized (5-10 MB)',
            'target_fps': 10, 'output_fps': 6, 'width': 1024, 'height': 384,
            'max_frames': 50, 'quality': 80
        },
        'demo': {
            'description': 'Demo version (1-3 MB)',
            'target_fps': 10, 'output_fps': 4, 'width': 800, 'height': 300,
            'max_frames': 20, 'quality': 70
        }
    }

def list_presets():
    """Print available presets."""
    presets = get_presets()
    print("Available Presets:")
    print("==================")
    for name, config in presets.items():
        print(f"  {name:8} - {config['description']}")
        print(f"           Resolution: {config['width']}x{config['height']}")
        print(f"           FPS: {config['output_fps']}, Max frames: {config['max_frames'] or 'All'}, Quality: {config['quality']}%")
        print()

def main():
    parser = argparse.ArgumentParser(
        description='Create side-by-side GIF from video frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Video mode: Create GIF from two video files (no labels)
  python3 create_side_by_side_gif.py --video1 video1.mp4 --video2 video2.mp4 --preset web
  
  # Video mode: With custom frame range
  python3 create_side_by_side_gif.py --video1 video1.mp4 --video2 video2.mp4 --start-frame 100 --end-frame 500
  
  # Video mode: With labels
  python3 create_side_by_side_gif.py --video1 video1.mp4 --video2 video2.mp4 --label1 "Original" --label2 "Processed"
  
  # Directory mode: Quick presets
  python3 create_side_by_side_gif.py --preset small      # Small file (2-5 MB)
  python3 create_side_by_side_gif.py --preset web        # Web optimized (5-10 MB)
  python3 create_side_by_side_gif.py --preset high       # High quality (20-40 MB)
  
  # Custom parameters
  python3 create_side_by_side_gif.py --output-fps 3 --width 640 --height 240 --max-frames 30
  
  # Show all presets
  python3 create_side_by_side_gif.py --list-presets
        """
    )
    
    # Preset options
    parser.add_argument('--preset', type=str, choices=list(get_presets().keys()),
                       help='Use a preset configuration (small, medium, high, full, web, demo)')
    parser.add_argument('--list-presets', action='store_true',
                       help='List all available presets and exit')
    
    # File paths - Video mode
    parser.add_argument('--video1', type=str, default=None,
                       help='Path to first video file (enables video mode)')
    parser.add_argument('--video2', type=str, default=None,
                       help='Path to second video file (enables video mode)')
    parser.add_argument('--label1', type=str, default=None,
                       help='Label for first video (omit to not show labels)')
    parser.add_argument('--label2', type=str, default=None,
                       help='Label for second video (omit to not show labels)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame index (0-based, applies to both videos)')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='Ending frame index (None for end of video, applies to both videos)')
    
    # File paths - Directory mode (legacy)
    parser.add_argument('--json-path', type=str, 
                       default='/home/simone/shared-data/fishway_ytvis/train.json',
                       help='Path to JSON file containing frame information')
    parser.add_argument('--video-id', type=int, default=122,
                       help='Video ID to process')
    parser.add_argument('--original-dir', type=str,
                       default='/home/simone/shared-data/fishway_ytvis/all_videos',
                       help='Directory containing original frames')
    parser.add_argument('--mask-dir', type=str,
                       default='/home/simone/shared-data/fishway_ytvis/all_videos_mask',
                       help='Directory containing mask frames')
    parser.add_argument('--output', type=str, default='side_by_side_comparison.gif',
                       help='Output GIF filename')
    
    # GIF parameters
    parser.add_argument('--target-fps', type=int, default=10,
                       help='Original video frame rate')
    parser.add_argument('--output-fps', type=int, default=5,
                       help='Output GIF frame rate (lower = smaller file)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Output GIF width')
    parser.add_argument('--height', type=int, default=480,
                       help='Output GIF height')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames (None for all)')
    parser.add_argument('--quality', type=int, default=85,
                       help='GIF quality (1-100)')
    
    args = parser.parse_args()
    
    # Handle list presets
    if args.list_presets:
        list_presets()
        return
    
    # Apply preset if specified
    if args.preset:
        presets = get_presets()
        preset_config = presets[args.preset]
        print(f"Using preset: {args.preset} - {preset_config['description']}")
        
        # Override arguments with preset values
        args.target_fps = preset_config['target_fps']
        args.output_fps = preset_config['output_fps']
        args.width = preset_config['width']
        args.height = preset_config['height']
        args.max_frames = preset_config['max_frames']
        args.quality = preset_config['quality']
        
        # Update output filename to include preset name
        if args.output == 'side_by_side_comparison.gif':
            args.output = f'side_by_side_comparison_{args.preset}.gif'
    
    print("=== Side-by-Side GIF Creator ===")
    
    # Determine mode: video files or directories
    video_mode = args.video1 is not None and args.video2 is not None
    
    if video_mode:
        print("Mode: Video files")
        print(f"Video 1: {args.video1}")
        print(f"Video 2: {args.video2}")
        if args.label1 or args.label2:
            print(f"Label 1: {args.label1}")
            print(f"Label 2: {args.label2}")
        print(f"Start frame: {args.start_frame}")
        print(f"End frame: {args.end_frame if args.end_frame else 'end of video'}")
    else:
        print("Mode: Frame directories")
        print(f"Video ID: {args.video_id}")
        print(f"Original directory: {args.original_dir}")
        print(f"Mask directory: {args.mask_dir}")
    
    print(f"Output: {args.output}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Output FPS: {args.output_fps}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Max frames: {args.max_frames}")
    print(f"Quality: {args.quality}")
    print()
    
    try:
        if video_mode:
            # Video mode: create GIF from two video files
            if not os.path.exists(args.video1):
                raise FileNotFoundError(f"Video 1 not found: {args.video1}")
            if not os.path.exists(args.video2):
                raise FileNotFoundError(f"Video 2 not found: {args.video2}")
            
            success = create_gif_from_videos(
                args.video1,
                args.video2,
                args.output,
                target_fps=args.target_fps,
                output_fps=args.output_fps,
                output_width=args.width,
                output_height=args.height,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                max_frames=args.max_frames,
                quality=args.quality,
                label1=args.label1,
                label2=args.label2
            )
        else:
            # Directory mode: create GIF from frame directories
            print("Loading frame information...")
            frame_list = load_video_frames(args.json_path, args.video_id)
            print(f"Found {len(frame_list)} frames for video {args.video_id}")
            
            success = create_gif(
                args.original_dir,
                args.mask_dir,
                frame_list,
                args.output,
                target_fps=args.target_fps,
                output_fps=args.output_fps,
                output_width=args.width,
                output_height=args.height,
                max_frames=args.max_frames,
                quality=args.quality
            )
        
        if success:
            print("\n✅ GIF creation completed successfully!")
        else:
            print("\n❌ GIF creation failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
