import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_util
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import messagebox

def decode_rle(rle_obj, height, width):
    """Decode RLE mask to binary array"""
    if isinstance(rle_obj['counts'], list):
        rle = mask_util.frPyObjects(rle_obj, height, width)
    else:
        rle = rle_obj
    return mask_util.decode(rle)

def count_connected_components(mask):
    """Count the number of connected components in a binary mask"""
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    # Subtract 1 because the first label (0) is the background
    return num_labels - 1

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

def validate_dataset_masks(json_file, image_root, output_dir=None):
    """Validate all masks in the dataset and identify multi-part masks"""
    
    print(f"Loading dataset from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract video and annotation information
    videos = {v['id']: v for v in data['videos']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Statistics
    total_annotations = 0
    multi_part_masks = []
    component_stats = defaultdict(list)
    
    print(f"Found {len(data['videos'])} videos")
    print(f"Found {len(data['annotations'])} annotations")
    
    # Process each annotation
    for ann_idx, annotation in enumerate(tqdm(data['annotations'], desc="Analyzing masks")):
        video_id = annotation['video_id']
        category_id = annotation['category_id']
        video = videos[video_id]
        
        # Get video dimensions
        height, width = video['height'], video['width']
        
        # Analyze each frame's segmentation
        for frame_idx, segmentation in enumerate(annotation['segmentations']):
            if segmentation is None:
                continue
                
            total_annotations += 1
            
            try:
                # Decode the mask
                mask = decode_rle(segmentation, height, width)
                
                # Analyze components
                analysis = analyze_mask_components(mask)
                
                # Store statistics
                component_stats['num_components'].append(analysis['num_components'])
                component_stats['largest_component_size'].append(analysis['largest_component_size'])
                
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
    
    # Print summary statistics
    print(f"\n=== MASK VALIDATION SUMMARY ===")
    print(f"Total annotations processed: {total_annotations}")
    print(f"Multi-part masks found: {len(multi_part_masks)}")
    print(f"Percentage of multi-part masks: {len(multi_part_masks)/total_annotations*100:.2f}%")
    
    if component_stats['num_components']:
        print(f"Average components per mask: {np.mean(component_stats['num_components']):.2f}")
        print(f"Max components in a single mask: {max(component_stats['num_components'])}")
    
    # Print detailed multi-part mask information
    if multi_part_masks:
        print(f"\n=== MULTI-PART MASKS DETAILS ===")
        for i, mask_info in enumerate(multi_part_masks[:10]):  # Show first 10
            print(f"Mask {i+1}:")
            print(f"  Video ID: {mask_info['video_id']}")
            print(f"  Category: {mask_info['category_name']}")
            print(f"  Frame: {mask_info['frame_idx']}")
            print(f"  Components: {mask_info['num_components']}")
            print(f"  Component sizes: {mask_info['component_sizes']}")
            print(f"  Largest component: {mask_info['largest_component_size']}")
            print(f"  Total mask size: {mask_info['total_mask_size']}")
            print()
        
        if len(multi_part_masks) > 10:
            print(f"... and {len(multi_part_masks) - 10} more multi-part masks")
    
    # Save detailed results to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save multi-part mask details
        multi_part_file = os.path.join(output_dir, "multi_part_masks.json")
        with open(multi_part_file, 'w') as f:
            json.dump(multi_part_masks, f, indent=2)
        print(f"Multi-part mask details saved to: {multi_part_file}")
        
        # Save summary statistics
        summary_file = os.path.join(output_dir, "mask_validation_summary.json")
        # Convert component_stats to JSON-serializable format
        serializable_component_stats = {}
        for key, values in component_stats.items():
            if values:  # Only include non-empty lists
                serializable_component_stats[key] = [float(val) for val in values]
        
        summary = {
            'total_annotations': total_annotations,
            'multi_part_masks_count': len(multi_part_masks),
            'multi_part_percentage': len(multi_part_masks)/total_annotations*100 if total_annotations > 0 else 0,
            'component_stats': serializable_component_stats,
            'multi_part_masks': multi_part_masks
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to: {summary_file}")
    
    return multi_part_masks, component_stats

def visualize_multi_part_mask(json_file, image_root, mask_info, output_path):
    """Visualize a specific multi-part mask"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    videos = {v['id']: v for v in data['videos']}
    video = videos[mask_info['video_id']]
    
    # Get the frame image
    frame_path = os.path.join(image_root, video['file_names'][mask_info['frame_idx']])
    if not os.path.exists(frame_path):
        print(f"Frame not found: {frame_path}")
        return
    
    # Load frame
    frame = cv2.imread(frame_path)
    
    # Get the annotation
    annotation = data['annotations'][mask_info['annotation_idx']]
    segmentation = annotation['segmentations'][mask_info['frame_idx']]
    
    # Decode mask
    mask = decode_rle(segmentation, video['height'], video['width'])
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original frame
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Frame')
    axes[0].axis('off')
    
    # Binary mask
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title(f'Binary Mask ({mask_info["num_components"]} components)')
    axes[1].axis('off')
    
    # Connected components (different colors)
    colored_components = np.zeros((*labels.shape, 3), dtype=np.uint8)
    colors = plt.cm.Set3(np.linspace(0, 1, num_labels))  # Generate colors
    for label in range(1, num_labels):
        colored_components[labels == label] = (colors[label][:3] * 255).astype(np.uint8)
    
    axes[2].imshow(colored_components)
    axes[2].set_title('Connected Components')
    axes[2].axis('off')
    
    plt.suptitle(f'Multi-part Mask: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def create_component_mask(labels, component_id):
    """Create a binary mask for a specific component"""
    return (labels == component_id).astype(np.uint8)

def get_component_bbox(component_mask):
    """Get bounding box for a component"""
    rows = np.any(component_mask, axis=1)
    cols = np.any(component_mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return x1, y1, x2, y2

def interactive_mask_review(json_file, image_root, multi_part_masks, output_dir):
    """Interactive review of multi-part masks with option to keep or merge components"""
    
    print(f"\n=== INTERACTIVE MASK REVIEW ===")
    print(f"Found {len(multi_part_masks)} multi-part masks to review")
    print(f"Press 'k' to keep all components, 'm' to merge to largest, 's' to skip, 'q' to quit")
    
    # Load dataset
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    videos = {v['id']: v for v in data['videos']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
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
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Multi-part Mask Review: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
        
        # Load and display original frame
        frame_path = os.path.join(image_root, video['file_names'][mask_info['frame_idx']])
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[0, 0].imshow(frame_rgb)
            axes[0, 0].set_title('Original Frame')
            axes[0, 0].axis('off')
        else:
            axes[0, 0].text(0.5, 0.5, 'Frame not found', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Original Frame')
            axes[0, 0].axis('off')
        
        # Original binary mask
        axes[0, 1].imshow(binary_mask, cmap='gray')
        axes[0, 1].set_title(f'Original Mask ({mask_info["num_components"]} components)')
        axes[0, 1].axis('off')
        
        # Connected components (different colors)
        colored_components = np.zeros((*labels.shape, 3), dtype=np.uint8)
        colors = plt.cm.Set3(np.linspace(0, 1, num_labels))
        for label in range(1, num_labels):
            colored_components[labels == label] = (colors[label][:3] * 255).astype(np.uint8)
        
        axes[0, 2].imshow(colored_components)
        axes[0, 2].set_title('Connected Components')
        axes[0, 2].axis('off')
        
        # Show individual components
        component_sizes = mask_info['component_sizes']
        largest_component_id = np.argmax(component_sizes) + 1
        
        # Largest component
        largest_mask = create_component_mask(labels, largest_component_id)
        axes[1, 0].imshow(largest_mask, cmap='gray')
        axes[1, 0].set_title(f'Largest Component (size: {component_sizes[largest_component_id-1]})')
        axes[1, 0].axis('off')
        
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
        
        # Get user input
        while True:
            plt.show(block=False)
            plt.pause(0.1)
            
            # Wait for key press
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
            elif key == 'q':  # Quit
                print("Quitting review...")
                plt.close('all')
                return review_results
            else:
                print("Invalid input. Use 'k' (keep), 'm' (merge), 's' (skip), or 'q' (quit)")
        
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

def apply_mask_edits(json_file, review_results, output_file):
    """Apply the mask edits based on review results"""
    
    print(f"\n=== APPLYING MASK EDITS ===")
    
    # Load original dataset
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    videos = {v['id']: v for v in data['videos']}
    
    # Track changes
    changes_made = 0
    
    # Process masks to be merged
    for mask_info in review_results['merge_to_largest']:
        video_id = mask_info['video_id']
        ann_idx = mask_info['annotation_idx']
        frame_idx = mask_info['frame_idx']
        
        video = videos[video_id]
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

if __name__ == "__main__":
    # Configuration
    json_file = "/data/fishway_ytvis/train.json"  # Change to your dataset path
    image_root = "/data/fishway_ytvis/all_videos"      # Change to your image root
    output_dir = "/store/simone/mask_validation_results"
    
    # Run validation
    multi_part_masks, stats = validate_dataset_masks(json_file, image_root, output_dir)
    
    # Visualize first few multi-part masks if any found
    if multi_part_masks:
        print(f"\nGenerating visualizations for first 5 multi-part masks...")
        for i, mask_info in enumerate(multi_part_masks[:5]):
            viz_path = os.path.join(output_dir, f"multi_part_mask_{i+1}.png")
            visualize_multi_part_mask(json_file, image_root, mask_info, viz_path)
        
        # Ask if user wants to review masks interactively
        print(f"\n=== INTERACTIVE REVIEW OPTION ===")
        print(f"Found {len(multi_part_masks)} multi-part masks")
        review_choice = input("Would you like to review these masks interactively? (y/n): ").lower().strip()
        
        if review_choice == 'y':
            # Run interactive review
            review_results = interactive_mask_review(json_file, image_root, multi_part_masks, output_dir)
            
            # Ask if user wants to apply the edits
            if review_results['merge_to_largest']:
                apply_choice = input(f"\nApply edits to merge {len(review_results['merge_to_largest'])} masks? (y/n): ").lower().strip()
                
                if apply_choice == 'y':
                    output_file = os.path.join(output_dir, "train_edited.json")
                    changes_made = apply_mask_edits(json_file, review_results, output_file)
                    print(f"✓ Applied {changes_made} mask edits to {output_file}")
                else:
                    print("Edits not applied. You can run the apply_mask_edits function manually later.")
            else:
                print("No masks selected for merging.")
        else:
            print("Skipping interactive review.") 