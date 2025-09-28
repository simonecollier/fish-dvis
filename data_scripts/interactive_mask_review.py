#!/usr/bin/env python3
"""
Interactive Mask Review Tool for DVIS Dataset

This script allows you to review multi-part masks in your dataset and decide
whether to keep all components or merge them to the largest component.

Usage:
    python interactive_mask_review.py --json-file /path/to/dataset.json --image-root /path/to/images --output-dir /path/to/output
"""

import json
import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from pycocotools import mask as mask_util
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
import tempfile

# Configure matplotlib backend
def setup_matplotlib():
    """Setup matplotlib for different environments"""
    import os
    
    # Always use Agg backend for saving images
    matplotlib.use('Agg')
    print("Using 'Agg' backend for image saving")
    
    # Set up the plot style
    plt.rcParams['figure.figsize'] = (18, 12)
    plt.rcParams['figure.dpi'] = 100

def save_and_display_image(fig, output_dir, mask_info, i, auto_open=True):
    """Save the visualization image and optionally open it in Cursor"""
    # Create a temporary file
    temp_filename = f"mask_review_{i+1:03d}_{mask_info['video_id']}_{mask_info['frame_idx']}.png"
    temp_path = os.path.join(output_dir, temp_filename)
    
    # Save the figure with optimized settings
    fig.savefig(temp_path, dpi=100, bbox_inches='tight', format='png')
    print(f"✓ Visualization saved to: {temp_path}")
    
    # Try to open in Cursor (non-blocking)
    if auto_open:
        try:
            subprocess.Popen(['cursor', temp_path], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            print("✓ Image opened in Cursor (non-blocking)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠️  Could not open in Cursor. Image saved at: {temp_path}")
            print("   Please open the image manually to review the mask.")
    
    return temp_path

def load_previous_review_results(resume_file):
    """Load previous review results for resuming"""
    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as f:
                previous_results = json.load(f)
            print(f"✓ Loaded previous review results from: {resume_file}")
            return previous_results
        except Exception as e:
            print(f"⚠️  Could not load previous results: {e}")
    return None

def get_reviewed_mask_ids(previous_results):
    """Get set of mask IDs that have already been reviewed"""
    if not previous_results:
        return set()
    
    reviewed_ids = set()
    for category in ['keep_all', 'merge_to_largest', 'skipped']:
        for mask_info in previous_results.get(category, []):
            # Create unique ID for each mask
            mask_id = f"{mask_info['video_id']}_{mask_info['frame_idx']}_{mask_info['annotation_idx']}"
            reviewed_ids.add(mask_id)
    
    return reviewed_ids

def decode_rle(rle_obj, height, width):
    """Decode RLE mask to binary array"""
    if isinstance(rle_obj['counts'], list):
        rle = mask_util.frPyObjects(rle_obj, height, width)
    else:
        rle = rle_obj
    return mask_util.decode(rle)

def analyze_mask_components(mask):
    """Analyze mask components and return detailed information"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    if num_labels <= 1:  # Only background
        return {
            'num_components': 0,
            'component_sizes': [],
            'is_multi_part': False,
            'largest_component_size': 0
        }
    
    # Analyze each component (skip background label 0)
    component_sizes = []
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        size = np.sum(component_mask)
        component_sizes.append(size)
    
    return {
        'num_components': len(component_sizes),
        'component_sizes': component_sizes,
        'is_multi_part': len(component_sizes) > 1,
        'largest_component_size': max(component_sizes) if component_sizes else 0
    }

def find_multi_part_masks(json_file):
    """Find all multi-part masks in the dataset"""
    print(f"Loading dataset from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract video and annotation information
    videos = {v['id']: v for v in data['videos']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    multi_part_masks = []
    
    print(f"Found {len(data['videos'])} videos")
    print(f"Found {len(data['annotations'])} annotations")
    
    # Process each annotation
    for ann_idx, annotation in enumerate(tqdm(data['annotations'], desc="Finding multi-part masks")):
        video_id = annotation['video_id']
        category_id = annotation['category_id']
        video = videos[video_id]
        
        # Get video dimensions
        height, width = video['height'], video['width']
        
        # Analyze each frame's segmentation
        for frame_idx, segmentation in enumerate(annotation['segmentations']):
            if segmentation is None:
                continue
                
            try:
                # Decode the mask
                mask = decode_rle(segmentation, height, width)
                
                # Analyze components
                analysis = analyze_mask_components(mask)
                
                # Record multi-part masks
                if analysis['is_multi_part']:
                    multi_part_masks.append({
                        'annotation_idx': ann_idx,
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'num_components': analysis['num_components'],
                        'component_sizes': [int(size) for size in analysis['component_sizes']],
                        'largest_component_size': int(analysis['largest_component_size']),
                        'total_mask_size': int(np.sum(mask > 0))
                    })
                    
            except Exception as e:
                print(f"Error processing annotation {ann_idx}, frame {frame_idx}: {e}")
    
    print(f"\nFound {len(multi_part_masks)} multi-part masks")
    return multi_part_masks, data

def create_component_mask(labels, component_id):
    """Create a binary mask for a specific component"""
    return (labels == component_id).astype(np.uint8)

def interactive_mask_review(json_file, image_root, multi_part_masks, data, output_dir, no_viz=False, no_auto_open=False, fast_mode=False, batch_mode=False, resume_file=None):
    """Interactive review of multi-part masks with option to keep or merge components"""
    
    # Setup matplotlib
    if not no_viz:
        setup_matplotlib()
        if fast_mode:
            plt.rcParams['figure.figsize'] = (12, 8)  # Smaller images
            plt.rcParams['figure.dpi'] = 80  # Lower DPI
    
    # Load previous results if resuming
    previous_results = None
    reviewed_ids = set()
    if resume_file:
        previous_results = load_previous_review_results(resume_file)
        if previous_results:
            reviewed_ids = get_reviewed_mask_ids(previous_results)
            print(f"Found {len(reviewed_ids)} previously reviewed masks")
    
    # Filter out already reviewed masks
    if reviewed_ids:
        original_count = len(multi_part_masks)
        multi_part_masks = [
            mask for mask in multi_part_masks 
            if f"{mask['video_id']}_{mask['frame_idx']}_{mask['annotation_idx']}" not in reviewed_ids
        ]
        print(f"Filtered to {len(multi_part_masks)} unreviewed masks (skipped {original_count - len(multi_part_masks)} already reviewed)")
    
    # Initialize results from previous session or create new
    if previous_results:
        review_results = previous_results
        print(f"Resuming with {len(review_results.get('keep_all', []))} kept, {len(review_results.get('merge_to_largest', []))} merged, {len(review_results.get('skipped', []))} skipped")
    else:
        review_results = {
            'keep_all': [],
            'merge_to_largest': [],
            'skipped': [],
            'manual_edits': []
        }
    
    print(f"\n=== INTERACTIVE MASK REVIEW ===")
    print(f"Found {len(multi_part_masks)} multi-part masks to review")
    print(f"Controls:")
    print(f"  'k' - Keep all components")
    print(f"  'm' - Merge to largest component")
    print(f"  's' - Skip this mask")
    print(f"  'q' - Quit review")
    print(f"  'i' - Show image info only (no visualization)")
    
    videos = {v['id']: v for v in data['videos']}
    
    # Results tracking
    review_results = {
        'keep_all': [],
        'merge_to_largest': [],
        'skipped': [],
        'manual_edits': []
    }
    
    for i, mask_info in enumerate(multi_part_masks):
        print(f"\n--- Reviewing mask {i+1}/{len(multi_part_masks)} ---")
        print(f"Video ID: {mask_info['video_id']}, Category: {mask_info['category_name']}")
        print(f"Frame: {mask_info['frame_idx']}, Components: {mask_info['num_components']}")
        
        # Get the original frame and mask
        video = videos[mask_info['video_id']]
        annotation = data['annotations'][mask_info['annotation_idx']]
        segmentation = annotation['segmentations'][mask_info['frame_idx']]
        
        # Decode mask
        mask = decode_rle(segmentation, video['height'], video['width'])
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        
        # Show component information
        component_sizes = mask_info['component_sizes']
        largest_component_id = np.argmax(component_sizes) + 1
        
        print(f"\nComponent Analysis:")
        print(f"  Total components: {mask_info['num_components']}")
        print(f"  Component sizes: {component_sizes}")
        print(f"  Largest component: {component_sizes[largest_component_id-1]} pixels")
        print(f"  Other components: {sum(component_sizes) - component_sizes[largest_component_id-1]} pixels")
        print(f"  Largest component ratio: {component_sizes[largest_component_id-1]/sum(component_sizes)*100:.1f}%")
        
        # Batch mode: auto-merge small components
        if batch_mode:
            largest_ratio = component_sizes[largest_component_id-1]/sum(component_sizes)*100
            if largest_ratio > 99.0:  # If largest component is >99% of total
                print(f"  🚀 Batch mode: Auto-merging (largest component is {largest_ratio:.1f}%)")
                review_results['merge_to_largest'].append(mask_info)
                continue
            else:
                print(f"  🔍 Batch mode: Manual review needed (largest component is {largest_ratio:.1f}%)")
                # In batch mode, auto-open images for manual review
                no_auto_open = False
        
        # Create visualization
        fig = None
        temp_image_path = None
        
        if not no_viz:
            try:
                # Create visualization with 6 panels: 3 frames + 3 mask views
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'Multi-part Mask Review: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
                
                # Helper function to load and display frame
                def load_frame(frame_idx):
                    if 0 <= frame_idx < len(video['file_names']):
                        frame_path = os.path.join(image_root, video['file_names'][frame_idx])
                        if os.path.exists(frame_path):
                            frame = cv2.imread(frame_path)
                            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return None
                
                # Frame before
                prev_frame = load_frame(mask_info['frame_idx'] - 1)
                if prev_frame is not None:
                    axes[0, 0].imshow(prev_frame)
                    axes[0, 0].set_title(f'Frame {mask_info["frame_idx"] - 1} (Before)')
                else:
                    axes[0, 0].text(0.5, 0.5, 'No previous frame', ha='center', va='center', transform=axes[0, 0].transAxes)
                    axes[0, 0].set_title(f'Frame {mask_info["frame_idx"] - 1} (Before)')
                axes[0, 0].axis('off')
                
                # Current frame
                current_frame = load_frame(mask_info['frame_idx'])
                if current_frame is not None:
                    axes[0, 1].imshow(current_frame)
                    axes[0, 1].set_title(f'Frame {mask_info["frame_idx"]} (Current)')
                else:
                    axes[0, 1].text(0.5, 0.5, 'Frame not found', ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title(f'Frame {mask_info["frame_idx"]} (Current)')
                axes[0, 1].axis('off')
                
                # Frame after
                next_frame = load_frame(mask_info['frame_idx'] + 1)
                if next_frame is not None:
                    axes[0, 2].imshow(next_frame)
                    axes[0, 2].set_title(f'Frame {mask_info["frame_idx"] + 1} (After)')
                else:
                    axes[0, 2].text(0.5, 0.5, 'No next frame', ha='center', va='center', transform=axes[0, 2].transAxes)
                    axes[0, 2].set_title(f'Frame {mask_info["frame_idx"] + 1} (After)')
                axes[0, 2].axis('off')
                
                # Connected components (different colors)
                colored_components = np.zeros((*labels.shape, 3), dtype=np.uint8)
                colors = plt.cm.Set3(np.linspace(0, 1, num_labels))
                for label in range(1, num_labels):
                    colored_components[labels == label] = (colors[label][:3] * 255).astype(np.uint8)
                
                axes[1, 0].imshow(colored_components)
                axes[1, 0].set_title('Connected Components')
                axes[1, 0].axis('off')
                
                # Largest component
                largest_mask = create_component_mask(labels, largest_component_id)
                
                # All other components combined
                other_components = np.zeros_like(labels, dtype=np.uint8)
                for label in range(1, num_labels):
                    if label != largest_component_id:
                        other_components = np.logical_or(other_components, labels == label).astype(np.uint8)
                
                axes[1, 1].imshow(other_components, cmap='gray')
                axes[1, 1].set_title(f'Other Components (combined)')
                axes[1, 1].axis('off')
                
                # Merged mask (largest component only)
                merged_mask = largest_mask
                axes[1, 2].imshow(merged_mask, cmap='gray')
                axes[1, 2].set_title('Merged (Largest Only)')
                axes[1, 2].axis('off')
                
                # Add component size information
                info_text = f"Component sizes: {component_sizes}\n"
                info_text += f"Largest: {component_sizes[largest_component_id-1]} pixels\n"
                info_text += f"Others: {sum(component_sizes) - component_sizes[largest_component_id-1]} pixels"
                
                fig.text(0.02, 0.02, info_text, fontsize=10, transform=fig.transFigure, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                
                plt.tight_layout()
                
                # Save and display the image
                temp_image_path = save_and_display_image(fig, output_dir, mask_info, i, not no_auto_open)
                
            except Exception as e:
                print(f"⚠️  Could not create visualization: {e}")
                print("   Continuing with text-based interface...")
                fig = None
        
        # Get user input
        while True:
            if fig is not None:
                # Wait for key press with visualization
                key = input(f"\nMask {i+1}/{len(multi_part_masks)} - Action (k/m/s/q/i): ").lower().strip()
            else:
                # Text-only interface
                key = input(f"\nMask {i+1}/{len(multi_part_masks)} - Action (k/m/s/q): ").lower().strip()
            
            if key == 'k':  # Keep all components
                review_results['keep_all'].append(mask_info)
                print("✓ Keeping all components")
                break
            elif key == 'm':  # Merge to largest
                review_results['merge_to_largest'].append(mask_info)
                print("✓ Merging to largest component")
                break
            elif key == 's':  # Skip
                review_results['skipped'].append(mask_info)
                print("✓ Skipped")
                break
            elif key == 'i' and fig is not None:  # Show info only
                print("Showing component information only...")
                continue
            elif key == 'q':  # Quit
                print("Quitting review...")
                if fig is not None:
                    plt.close('all')
                return review_results
            else:
                if fig is not None:
                    print("Invalid input. Use 'k' (keep), 'm' (merge), 's' (skip), 'i' (info), or 'q' (quit)")
                else:
                    print("Invalid input. Use 'k' (keep), 'm' (merge), 's' (skip), or 'q' (quit)")
        
        # Clean up temporary image
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"✓ Cleaned up temporary image: {temp_image_path}")
            except Exception as e:
                print(f"⚠️  Could not clean up temporary image: {e}")
        
        if fig is not None:
            plt.close(fig)
    
    # Save review results
    review_file = os.path.join(output_dir, "mask_review_results.json")
    with open(review_file, 'w') as f:
        json.dump(review_results, f, indent=2)
    
    print(f"\n=== REVIEW COMPLETE ===")
    print(f"Kept all components: {len(review_results['keep_all'])}")
    print(f"Merged to largest: {len(review_results['merge_to_largest'])}")
    print(f"Skipped: {len(review_results['skipped'])}")
    print(f"Results saved to: {review_file}")
    
    return review_results

def apply_mask_edits(data, review_results, output_file):
    """Apply the mask edits based on review results"""
    
    print(f"\n=== APPLYING MASK EDITS ===")
    
    # Track changes
    changes_made = 0
    
    # Process masks to be merged
    for mask_info in review_results['merge_to_largest']:
        video_id = mask_info['video_id']
        ann_idx = mask_info['annotation_idx']
        frame_idx = mask_info['frame_idx']
        
        video = data['videos'][video_id - 1]  # Adjust for 0-based indexing
        annotation = data['annotations'][ann_idx]
        segmentation = annotation['segmentations'][frame_idx]
        
        # Decode original mask
        mask = decode_rle(segmentation, video['height'], video['width'])
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        
        # Find largest component
        component_sizes = []
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8)
            size = np.sum(component_mask)
            component_sizes.append(size)
        
        largest_component_id = np.argmax(component_sizes) + 1
        
        # Create new mask with only largest component
        new_mask = (labels == largest_component_id).astype(np.uint8)
        
        # Encode back to RLE
        new_rle = mask_util.encode(np.asfortranarray(new_mask))
        new_rle['counts'] = new_rle['counts'].decode('utf-8')
        
        # Update the segmentation
        data['annotations'][ann_idx]['segmentations'][frame_idx] = new_rle
        changes_made += 1
        
        print(f"✓ Merged mask in Video {video_id}, Frame {frame_idx}")
    
    # Save modified dataset
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"\n=== EDITS APPLIED ===")
    print(f"Total changes made: {changes_made}")
    print(f"Modified dataset saved to: {output_file}")
    
    return changes_made

def main():
    parser = argparse.ArgumentParser(description='Interactive mask review tool')
    parser.add_argument('--json-file', required=True, help='Path to dataset JSON file')
    parser.add_argument('--image-root', required=True, help='Path to image directory')
    parser.add_argument('--output-dir', required=True, help='Path to output directory')
    parser.add_argument('--apply-edits', action='store_true', help='Apply edits after review')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization (text-only mode)')
    parser.add_argument('--no-auto-open', action='store_true', help='Disable auto-opening images in Cursor')
    parser.add_argument('--fast-mode', action='store_true', help='Fast mode: smaller images, no auto-open')
    parser.add_argument('--batch-mode', action='store_true', help='Batch mode: auto-merge small components (<1% of largest)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous review session')
    parser.add_argument('--resume-file', help='Path to resume file (default: output_dir/mask_review_results.json)')
    
    args = parser.parse_args()
    
    # Setup matplotlib
    if not args.no_viz:
        setup_matplotlib()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find multi-part masks
    multi_part_masks, data = find_multi_part_masks(args.json_file)
    
    if not multi_part_masks:
        print("No multi-part masks found!")
        return
    
    # Determine resume file path
    resume_file = None
    if args.resume:
        if args.resume_file:
            resume_file = args.resume_file
        else:
            resume_file = os.path.join(args.output_dir, "mask_review_results.json")
    
    # Run interactive review
    review_results = interactive_mask_review(
        args.json_file, 
        args.image_root, 
        multi_part_masks, 
        data, 
        args.output_dir, 
        args.no_viz,
        args.no_auto_open or args.fast_mode,
        args.fast_mode,
        args.batch_mode,
        resume_file
    )
    
    # Apply edits if requested
    if args.apply_edits and review_results['merge_to_largest']:
        output_file = os.path.join(args.output_dir, "train_edited.json")
        changes_made = apply_mask_edits(data, review_results, output_file)
        print(f"✓ Applied {changes_made} mask edits to {output_file}")
    elif review_results['merge_to_largest']:
        print(f"\nTo apply edits later, run:")
        print(f"python apply_mask_edits.py --json-file {args.json_file} --review-results {os.path.join(args.output_dir, 'mask_review_results.json')}")

if __name__ == "__main__":
    main() 