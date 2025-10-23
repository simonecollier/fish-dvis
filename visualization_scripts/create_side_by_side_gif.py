#!/usr/bin/env python3
"""
Create side-by-side GIF comparing original video frames with mask frames.

This script creates a GIF that shows two videos side by side:
- Left: Original video frames
- Right: Corresponding mask frames

Features:
- Configurable frame rate (with frame skipping for smaller file sizes)
- Configurable output resolution
- Automatic frame selection from JSON metadata
- Progress tracking and file size optimization
- Built-in presets for common use cases

Usage Examples:
    # Quick presets
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

def load_video_frames(json_path, video_id):
    """Load frame information for a specific video from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for video in data['videos']:
        if video['id'] == video_id:
            return video['file_names']
    
    raise ValueError(f"Video ID {video_id} not found in {json_path}")

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

def create_gif(original_dir, mask_dir, frame_list, output_path, 
               target_fps=10, output_fps=5, output_width=1280, output_height=480,
               max_frames=None, quality=85):
    """
    Create a side-by-side GIF.
    
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
  # Quick presets
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
    
    # File paths
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
        # Load frame information
        print("Loading frame information...")
        frame_list = load_video_frames(args.json_path, args.video_id)
        print(f"Found {len(frame_list)} frames for video {args.video_id}")
        
        # Create GIF
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
        sys.exit(1)

if __name__ == "__main__":
    main()
