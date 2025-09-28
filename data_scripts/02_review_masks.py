import json
import os
import cv2
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_util
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib
from datetime import datetime
from scipy.spatial.distance import directed_hausdorff
import math
import subprocess
import tempfile
from mask_editor import (
    get_mask_options, apply_mask_edit, ensure_cleaned_json_exists,
    load_cleaned_data, save_cleaned_data, decode_rle, get_non_editable_options,
    calculate_bbox_from_mask
)
from mask_analysis_utils import check_temporal_boundary_consistency

# Configure matplotlib backend for remote workstation
matplotlib.use('Agg')

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

def decode_rle(rle_obj, height, width):
    """Decode RLE mask to binary array"""
    if isinstance(rle_obj['counts'], list):
        rle = mask_util.frPyObjects(rle_obj, height, width)
    else:
        rle = rle_obj
    return mask_util.decode(rle)

def load_review_tracking(tracking_file):
    """Load existing review tracking data"""
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load tracking file from {tracking_file}")
    return {
        'reviewed_videos': {},
        'edited_frames': {},
        'review_criteria_version': '1.0',
        'last_updated': datetime.now().isoformat(),
        'data_hash': None,
        'script_hash': None
    }

def save_review_tracking(tracking_data, tracking_file):
    """Save review tracking data"""
    tracking_data['last_updated'] = datetime.now().isoformat()
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f, indent=2)

def detect_skinny_masks(mask, min_width=10):
    """Detect skinny parts of masks that might be artifacts"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Use morphological operations to detect skinny parts
    kernel = np.ones((min_width, min_width), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    # The difference shows skinny parts that disappear after erosion
    skinny_parts = binary_mask - dilated
    
    # Calculate skinny part statistics
    skinny_area = np.sum(skinny_parts)
    total_area = np.sum(binary_mask)
    skinny_ratio = skinny_area / total_area if total_area > 0 else 0
    
    # Check if any skinny parts exist
    has_skinny_parts = skinny_area > 0
    
    return {
        'has_skinny_parts': has_skinny_parts,
        'skinny_area': int(skinny_area),
        'total_area': int(total_area),
        'skinny_ratio': float(skinny_ratio),
        'skinny_mask': skinny_parts
    }

def group_issues_by_mask(mask_issues):
    """Group mask issues by unique mask (video_id, annotation_id, frame_idx)
    
    Note: This groups by individual fish annotations, not by frame.
    If a frame has two fish, they will be treated as separate masks.
    This is correct behavior as each fish annotation is independent.
    """
    mask_groups = {}
    
    for issue in mask_issues:
        # Create unique key for each mask
        # This groups by individual fish annotation, not by frame
        mask_key = (issue['video_id'], issue['annotation_id'], issue['frame_idx'])
        
        if mask_key not in mask_groups:
            mask_groups[mask_key] = []
        
        mask_groups[mask_key].append(issue)
    
    # Convert to list of groups, each group contains all issues for one mask
    grouped_issues = []
    for mask_key, issues in mask_groups.items():
        # Sort issues by type for consistent ordering
        # Priority: editable issues first, then by type name
        editable_types = ['multi_part', 'skinny', 'smoothness', 'internal', 'internal_holes']
        
        def sort_key(issue):
            is_editable = issue['issue_type'] in editable_types
            return (not is_editable, issue['issue_type'])
        
        issues.sort(key=sort_key)
        
        grouped_issues.append({
            'mask_key': mask_key,
            'issues': issues,
            'primary_issue': issues[0]  # Use first issue as primary for display
        })
    
    return grouped_issues

def get_all_issues_text(mask_group, data, video):
    """Get text representation of all issues for a mask group"""
    issues_text = []
    
    for issue in mask_group['issues']:
        issue_text = get_all_issues_for_mask(issue, data, video)
        issues_text.extend(issue_text)
    
    return issues_text

def create_multi_issue_visualization(mask_group, video, binary_mask, image_root, issue_idx, output_dir, data, display_mode='editable'):
    """Create visualization showing all issues for a mask with different display modes"""
    
    # Get the primary issue for basic info
    primary_issue = mask_group['primary_issue']
    
    # Determine if any issues are editable
    editable_types = ['multi_part', 'skinny', 'smoothness', 'internal', 'internal_holes']
    has_editable_issues = any(issue['issue_type'] in editable_types for issue in mask_group['issues'])
    
    if display_mode == 'editable' and has_editable_issues:
        # Show editable options (original 1-3 style)
        # Use the first editable issue for visualization
        editable_issue = next((issue for issue in mask_group['issues'] if issue['issue_type'] in editable_types), primary_issue)
        
        # Use the unified visualization function for all editable issue types
        return create_unified_visualization(editable_issue, video, binary_mask, image_root, issue_idx, output_dir, editable_issue['issue_type'])
    else:
        # Show non-editable visualization with mask sequence
        return create_non_editable_multi_issue_visualization(mask_group, video, binary_mask, image_root, issue_idx, output_dir, data)

def create_non_editable_multi_issue_visualization(mask_group, video, binary_mask, image_root, issue_idx, output_dir, data):
    """Create visualization showing all issues with mask sequence"""
    
    primary_issue = mask_group['primary_issue']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Load and display frames (no mask overlay)
    load_and_display_frames(primary_issue, video, image_root, axes[:3])
    
    # Show mask sequence (current, previous, next if available)
    annotation = data['annotations'][primary_issue['annotation_idx']]
    frame_idx = primary_issue['frame_idx']
    
    # Current mask (colored)
    colored_current_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    colored_current_mask[binary_mask > 0] = [255, 0, 0]  # Red for the fish being reviewed
    axes[3].imshow(colored_current_mask)
    axes[3].set_title(f'Current Frame {frame_idx}')
    axes[3].axis('off')
    
    # Previous frame mask (colored)
    if frame_idx > 0 and annotation['segmentations'][frame_idx - 1] is not None:
        prev_mask = decode_rle(annotation['segmentations'][frame_idx - 1], video['height'], video['width'])
        colored_prev_mask = np.zeros((prev_mask.shape[0], prev_mask.shape[1], 3), dtype=np.uint8)
        colored_prev_mask[prev_mask > 0] = [255, 0, 0]  # Red for the fish being reviewed
        axes[4].imshow(colored_prev_mask)
        axes[4].set_title(f'Previous Frame {frame_idx - 1}')
    else:
        axes[4].text(0.5, 0.5, 'No previous frame', ha='center', va='center', transform=axes[4].transAxes)
        axes[4].set_title('No Previous Frame')
    axes[4].axis('off')
    
    # Next frame mask (colored)
    if frame_idx + 1 < len(annotation['segmentations']) and annotation['segmentations'][frame_idx + 1] is not None:
        next_mask = decode_rle(annotation['segmentations'][frame_idx + 1], video['height'], video['width'])
        colored_next_mask = np.zeros((next_mask.shape[0], next_mask.shape[1], 3), dtype=np.uint8)
        colored_next_mask[next_mask > 0] = [255, 0, 0]  # Red for the fish being reviewed
        axes[5].imshow(colored_next_mask)
        axes[5].set_title(f'Next Frame {frame_idx + 1}')
    else:
        axes[5].text(0.5, 0.5, 'No next frame', ha='center', va='center', transform=axes[5].transAxes)
        axes[5].set_title('No Next Frame')
    axes[5].axis('off')
    
    # Add title with all issue types
    issue_types = [issue['issue_type'] for issue in mask_group['issues']]
    fig.suptitle(f'Mask Issues: {", ".join(issue_types)}', fontsize=16)
    
    plt.tight_layout()
    
    # Save and display
    temp_image_path = save_and_display_image(fig, output_dir, primary_issue, issue_idx, auto_open=True)
    plt.close(fig)
    
    return temp_image_path, None  # No mask options for non-editable display

def interactive_mask_review_with_tracking(json_file, cleaned_json_file, image_root, mask_issues, output_dir, tracking_file=None):
    """Interactive review of mask issues with immediate editing functionality"""
    
    if tracking_file is None:
        tracking_file = os.path.join(output_dir, "mask_review_tracking.json") if output_dir else "mask_review_tracking.json"
    
    # Load tracking data
    tracking_data = load_review_tracking(tracking_file)
    
    # Ensure cleaned JSON file exists and is up to date
    ensure_cleaned_json_exists(json_file, cleaned_json_file)
    
    # Load cleaned data
    cleaned_data = load_cleaned_data(cleaned_json_file)
    
    print(f"\n=== INTERACTIVE MASK REVIEW ===")
    print(f"Found {len(mask_issues)} mask issues to review")
    print(f"Edits will be applied immediately to: {cleaned_json_file}")
    print(f"Use number keys to select mask options, 's' to skip, 'p' for progress, 'S' to save, 'q' to quit")
    print(f"Use '4' to toggle display mode (editable options vs. non-editable visualization)")
    
    # Group issues by mask
    grouped_issues = group_issues_by_mask(mask_issues)
    print(f"Grouped into {len(grouped_issues)} unique masks to review")
    
    # Check for existing review progress
    if 'review_progress' in tracking_data:
        last_reviewed = tracking_data['review_progress'].get('last_reviewed_index', -1)
        if last_reviewed >= 0:
            resume_choice = input(f"\nFound previous review progress. Last reviewed: mask {last_reviewed + 1}/{len(grouped_issues)}. Resume from here? (y/n): ").lower().strip()
            if resume_choice == 'y':
                start_index = last_reviewed + 1
                print(f"Resuming from mask {start_index + 1}/{len(grouped_issues)}")
            else:
                start_index = 0
                print("Starting from the beginning")
        else:
            start_index = 0
    else:
        start_index = 0
        tracking_data['review_progress'] = {}
    
    # Load dataset
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    videos = {v['id']: v for v in data['videos']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Results tracking
    review_results = {
        'edited': [],
        'skipped': []
    }
    
    for i, mask_group in enumerate(grouped_issues[start_index:], start=start_index):
        primary_issue = mask_group['primary_issue']
        video_id = primary_issue['video_id']
        video = videos[video_id]
        video_folder = video['file_names'][0].split('/')[0] if video['file_names'] else f"video_{video_id}"
        
        print(f"\n--- Reviewing mask {i+1}/{len(grouped_issues)} ---")
        print(f"Video ID: {video_id}, Category: {primary_issue['category_name']}")
        print(f"Frame: {primary_issue['frame_idx']}")
        
        # Display all issues for this mask
        all_issues_text = get_all_issues_text(mask_group, data, video)
        print(f"\n📊 ALL ISSUES FOR THIS MASK:")
        for issue_text in all_issues_text:
            print(f"   • {issue_text}")
        
        # Get the original frame and mask
        annotation = data['annotations'][primary_issue['annotation_idx']]
        segmentation = annotation['segmentations'][primary_issue['frame_idx']]
        
        # Decode mask
        mask = decode_rle(segmentation, video['height'], video['width'])
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Check if any issues are editable
        editable_types = ['multi_part', 'skinny', 'smoothness', 'internal', 'internal_holes']
        has_editable_issues = any(issue['issue_type'] in editable_types for issue in mask_group['issues'])
        
        # Initialize display mode
        display_mode = 'editable' if has_editable_issues else 'non_editable'
        
        # Create visualization based on display mode
        temp_image_path = None
        mask_options = None
        
        # Get user input
        while True:
            # Create visualization based on current display mode
            temp_image_path, mask_options = create_multi_issue_visualization(
                mask_group, video, binary_mask, image_root, i, output_dir, data, display_mode
            )
            
            # Determine if this is a non-editable issue
            non_editable_types = ['size', 'convexity', 'motion', 'boundary', 'temporal_boundary', 'inconsistent', 'complexity']
            is_non_editable = all(issue['issue_type'] in non_editable_types for issue in mask_group['issues'])
            
            if mask_options and display_mode == 'editable':
                # Show options for editable issues
                print(f"\nMask {i+1}/{len(grouped_issues)} - Available options:")
                for option in mask_options['options']:
                    print(f"  {option['id']}: {option['name']} - {mask_options['descriptions'][option['id']-1]}")
                print("  4: Toggle display mode, s: Skip, e: Manual edit, p: Progress, q: Quit")
                
                key = input(f"Select option (1-{len(mask_options['options'])}, 4/s/e/p/q): ").strip()
                
                # Check if it's a number (option selection)
                if key.isdigit():
                    option_id = int(key)
                    if option_id == 4:
                        # Toggle display mode
                        display_mode = 'non_editable' if display_mode == 'editable' else 'editable'
                        print(f"Switched to {display_mode} display mode")
                        continue
                    elif 1 <= option_id <= len(mask_options['options']):
                        # Apply the selected mask edit
                        selected_option = mask_options['options'][option_id - 1]
                        new_mask = selected_option['mask']
                        
                        # Apply the edit to the cleaned data
                        cleaned_data = apply_mask_edit(
                            cleaned_data, 
                            primary_issue['annotation_idx'], 
                            primary_issue['frame_idx'], 
                            new_mask, 
                            video['height'], 
                            video['width']
                        )
                        
                        # Print the changes made
                        original_area = np.sum(binary_mask)
                        new_area = np.sum(new_mask)
                        area_change = new_area - original_area
                        print(f"  Area changed: {original_area} → {new_area} ({area_change:+d} pixels)")
                        
                        # Calculate bounding box changes
                        original_bbox = calculate_bbox_from_mask(binary_mask)
                        new_bbox = calculate_bbox_from_mask(new_mask)
                        print(f"  Bounding box: {original_bbox} → {new_bbox}")
                        
                        # Save the updated cleaned data
                        save_cleaned_data(cleaned_data, cleaned_json_file)
                        
                        print(f"✓ Applied {selected_option['name']} to mask")
                        review_results['edited'].append({
                            'mask_group': mask_group,
                            'option_selected': selected_option['name'],
                            'option_id': option_id
                        })
                        break
                    else:
                        print(f"Invalid option number. Please select 1-{len(mask_options['options'])} or 4")
                        continue
            else:
                # Show options for non-editable issues or non-editable display mode
                if display_mode == 'editable':
                    print(f"\nMask {i+1}/{len(grouped_issues)} - Non-editable issues detected")
                    print("  1: Keep original mask")
                    print("  4: Toggle display mode, s: Skip, e: Manual edit, p: Progress, q: Quit")
                    key = input(f"Select action (1/4/s/e/p/q): ").strip()
                    
                    if key == '1':
                        print("✓ Kept original mask")
                        review_results['edited'].append({
                            'mask_group': mask_group,
                            'option_selected': 'keep_original',
                            'option_id': 1
                        })
                        break
                    elif key == '4':
                        # Toggle display mode
                        display_mode = 'non_editable' if display_mode == 'editable' else 'editable'
                        print(f"Switched to {display_mode} display mode")
                        continue
                else:
                    # Non-editable display mode
                    print(f"\nMask {i+1}/{len(grouped_issues)} - Non-editable display mode")
                    print("  1: Keep original mask")
                    print("  4: Toggle display mode, s: Skip, e: Manual edit, p: Progress, q: Quit")
                    key = input(f"Select action (1/4/s/e/p/q): ").strip()
                    
                    if key == '1':
                        print("✓ Kept original mask")
                        review_results['edited'].append({
                            'mask_group': mask_group,
                            'option_selected': 'keep_original',
                            'option_id': 1
                        })
                        break
                    elif key == '4':
                        # Toggle display mode
                        display_mode = 'editable' if display_mode == 'non_editable' else 'non_editable'
                        print(f"Switched to {display_mode} display mode")
                        continue
            
            if key.lower() == 's':  # Skip (temporarily, don't count as reviewed)
                print("✓ Temporarily skipped - will return to this mask later")
                # Don't add to skipped results - this mask will be reviewed again
                break
            elif key.lower() == 'e':  # Manual edit required
                review_results['skipped'].append(mask_group)
                print("✓ Marked for manual editing")
                break
            elif key.lower() == 'p':  # Show progress
                show_review_progress(i, len(grouped_issues), review_results)
                continue
            elif key.lower() == 'q':  # Quit with confirmation
                # Save progress first
                tracking_data['review_progress']['last_reviewed_index'] = i
                tracking_data['review_progress']['total_issues'] = len(grouped_issues)
                tracking_data['review_progress']['reviewed_count'] = i + 1
                tracking_data['review_progress']['last_updated'] = datetime.now().isoformat()
                save_review_tracking(tracking_data, tracking_file)
                print(f"✓ Progress saved at mask {i + 1}/{len(grouped_issues)}")
                
                # Ask for confirmation
                confirm = input("Do you really want to quit? (y/n): ").strip().lower()
                if confirm == 'y':
                    print("Quitting review...")
                    break
                else:
                    print("Continuing review...")
                    continue
            else:
                if mask_options and display_mode == 'editable':
                    print(f"Invalid input. Please select 1-{len(mask_options['options'])}, 4, s, e, p, or q")
                else:
                    print("Invalid input. Use '1' (keep original), '4' (toggle display), 's' (skip), 'e' (manual edit), 'p' (progress), or 'q' (quit)")
        
        if key == 'q':
            # This should not happen as 'q' is handled in the inner loop
            print("Unexpected quit command")
            break
        elif key == 'p':  # Show progress
            show_review_progress(i, len(grouped_issues), review_results)
            continue
        
        # Clean up temporary image after decision
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
                print(f"✓ Cleaned up temporary image: {temp_image_path}")
            except Exception as e:
                print(f"⚠️  Could not clean up temporary image: {e}")
    
    # Save detailed tracking data including original and edited mask information
    save_detailed_tracking_data(tracking_data, review_results, data, cleaned_data, tracking_file)
    
    print(f"\n=== REVIEW COMPLETED ===")
    print(f"Total masks reviewed: {len(review_results['edited']) + len(review_results['skipped'])}")
    print(f"Masks edited: {len(review_results['edited'])}")
    print(f"Masks marked for manual editing: {len(review_results['skipped'])}")
    print(f"Results saved to: {cleaned_json_file}")
    print(f"Tracking data saved to: {tracking_file}")
    
    return review_results

def load_and_display_frames(mask_info, video, image_root, axes):
    """Helper function to load and display previous, current, and next frames"""
    # Get frame indices
    frame_idx = mask_info['frame_idx']
    prev_frame_idx = max(0, frame_idx - 1)
    next_frame_idx = min(len(video['file_names']) - 1, frame_idx + 1)
    
    # Load and display previous frame
    prev_frame_path = os.path.join(image_root, video['file_names'][prev_frame_idx])
    if os.path.exists(prev_frame_path):
        prev_frame = cv2.imread(prev_frame_path)
        prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(prev_frame_rgb)
        axes[0, 0].set_title(f'Frame {prev_frame_idx} (Previous)')
        axes[0, 0].axis('off')
    else:
        axes[0, 0].text(0.5, 0.5, 'Frame not found', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title(f'Frame {prev_frame_idx} (Previous)')
        axes[0, 0].axis('off')
    
    # Load and display current frame
    frame_path = os.path.join(image_root, video['file_names'][frame_idx])
    if os.path.exists(frame_path):
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(frame_rgb)
        axes[0, 1].set_title(f'Frame {frame_idx} (Current)')
        axes[0, 1].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'Frame not found', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title(f'Frame {frame_idx} (Current)')
        axes[0, 1].axis('off')
    
    # Load and display next frame
    next_frame_path = os.path.join(image_root, video['file_names'][next_frame_idx])
    if os.path.exists(next_frame_path):
        next_frame = cv2.imread(next_frame_path)
        next_frame_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
        axes[0, 2].imshow(next_frame_rgb)
        axes[0, 2].set_title(f'Frame {next_frame_idx} (Next)')
        axes[0, 2].axis('off')
    else:
        axes[0, 2].text(0.5, 0.5, 'Frame not found', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title(f'Frame {next_frame_idx} (Next)')
        axes[0, 2].axis('off')

# Legacy wrapper functions for backward compatibility
# These now use the unified visualization function

def create_multi_part_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir):
    """Create visualization for multi-part mask issues with numbered options"""
    return create_unified_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir, 'multi_part')

def create_skinny_mask_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir):
    """Create visualization for skinny mask issues with numbered options"""
    return create_unified_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir, 'skinny')

def create_inconsistent_mask_visualization(mask_info, video, binary_mask, image_root, data, issue_idx, output_dir):
    """Create visualization for inconsistent mask issues with numbered options"""
    return create_unified_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir, 'inconsistent')

def create_smoothness_issue_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir):
    """Create visualization for smoothness issue masks with numbered options"""
    return create_unified_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir, 'smoothness')

def create_internal_issue_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir):
    """Create visualization for internal consistency issue masks with numbered options"""
    return create_unified_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir, 'internal')



def create_non_editable_issue_visualization(mask_info, video, binary_mask, image_root, data, issue_idx, output_dir, cleaned_data):
    """Create visualization for non-editable issues (size, convexity, motion, boundary, inconsistent, complexity)"""
    
    # Get mask options for non-editable issues
    mask_options = get_non_editable_options(binary_mask)
    
    # Get frame indices
    frame_idx = mask_info['frame_idx']
    prev_frame_idx = max(0, frame_idx - 1)
    next_frame_idx = min(len(video['file_names']) - 1, frame_idx + 1)
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Non-Editable Issues: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
    
    # Load and display frames using helper function with mask highlighting
    load_and_display_frames(mask_info, video, image_root, axes, highlight_mask=binary_mask)
    
    # Get masks from cleaned data for the three frames
    annotation = cleaned_data['annotations'][mask_info['annotation_idx']]
    
    # Previous frame mask
    if prev_frame_idx < len(annotation['segmentations']):
        prev_segmentation = annotation['segmentations'][prev_frame_idx]
        prev_mask = decode_rle(prev_segmentation, video['height'], video['width'])
        prev_binary_mask = (prev_mask > 0).astype(np.uint8)
        
        # Load previous frame image
        prev_frame_path = os.path.join(image_root, video['file_names'][prev_frame_idx])
        if os.path.exists(prev_frame_path):
            prev_frame = cv2.imread(prev_frame_path)
            prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
            
            # Overlay mask with transparency
            axes[1, 0].imshow(prev_frame_rgb)
            axes[1, 0].imshow(prev_binary_mask, alpha=0.3, cmap='Reds')
            axes[1, 0].set_title(f'Frame {prev_frame_idx} Mask Overlay')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Frame not found', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title(f'Frame {prev_frame_idx} Mask Overlay')
            axes[1, 0].axis('off')
    else:
        axes[1, 0].text(0.5, 0.5, 'No mask data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title(f'Frame {prev_frame_idx} Mask Overlay')
        axes[1, 0].axis('off')
    
    # Current frame mask
    current_segmentation = annotation['segmentations'][frame_idx]
    current_mask = decode_rle(current_segmentation, video['height'], video['width'])
    current_binary_mask = (current_mask > 0).astype(np.uint8)
    
    # Load current frame image
    frame_path = os.path.join(image_root, video['file_names'][frame_idx])
    if os.path.exists(frame_path):
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Overlay mask with transparency
        axes[1, 1].imshow(frame_rgb)
        axes[1, 1].imshow(current_binary_mask, alpha=0.3, cmap='Reds')
        axes[1, 1].set_title(f'Frame {frame_idx} Mask Overlay')
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, 'Frame not found', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title(f'Frame {frame_idx} Mask Overlay')
        axes[1, 1].axis('off')
    
    # Next frame mask
    if next_frame_idx < len(annotation['segmentations']):
        next_segmentation = annotation['segmentations'][next_frame_idx]
        next_mask = decode_rle(next_segmentation, video['height'], video['width'])
        next_binary_mask = (next_mask > 0).astype(np.uint8)
        
        # Load next frame image
        next_frame_path = os.path.join(image_root, video['file_names'][next_frame_idx])
        if os.path.exists(next_frame_path):
            next_frame = cv2.imread(next_frame_path)
            next_frame_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
            
            # Overlay mask with transparency
            axes[1, 2].imshow(next_frame_rgb)
            axes[1, 2].imshow(next_binary_mask, alpha=0.3, cmap='Reds')
            axes[1, 2].set_title(f'Frame {next_frame_idx} Mask Overlay')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, 'Frame not found', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title(f'Frame {next_frame_idx} Mask Overlay')
            axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.5, 0.5, 'No mask data', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title(f'Frame {next_frame_idx} Mask Overlay')
        axes[1, 2].axis('off')
    
    # Add description of all issues
    all_issues = get_all_issues_for_mask(mask_info, data, video)
    description_text = "Issues detected:\n"
    for issue in all_issues:
        description_text += f"• {issue}\n"
    description_text += "\nOptions:\n1: Keep original\ns: Skip (return later), e: Manual edit, p: Progress, q: Quit"
    
    fig.text(0.02, 0.02, description_text, fontsize=10, transform=fig.transFigure, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save and display the image
    temp_image_path = save_and_display_image(fig, output_dir, mask_info, issue_idx)
    
    return temp_image_path, mask_options

def get_all_issues_for_mask(mask_info, data, video):
    """Get all issues present for a given mask"""
    issues = []
    
    # Get the mask data
    annotation = data['annotations'][mask_info['annotation_idx']]
    segmentation = annotation['segmentations'][mask_info['frame_idx']]
    mask = decode_rle(segmentation, video['height'], video['width'])
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Check each issue type
    if mask_info['issue_type'] == 'multi_part':
        issues.append(f"Multi-part: {mask_info['num_components']} disconnected parts")
        issues.append(f"  Component areas: {mask_info['component_sizes']}")
        issues.append(f"  Largest component: {mask_info['largest_component_size']} pixels")
    
    elif mask_info['issue_type'] == 'skinny':
        issues.append(f"Skinny parts: {mask_info['num_boundary_skinny_parts']} skinny boundary parts")
        for part in mask_info['boundary_skinny_parts']:
            issues.append(f"  {part[0]} boundary: {part[2]}×{part[3]} pixels")
    
    elif mask_info['issue_type'] == 'smoothness':
        issues.append(f"Smoothness: Rough boundary (ratio: {mask_info['smoothness_ratio']:.3f})")
        issues.append(f"  Perimeter: {mask_info['perimeter']}, Area: {mask_info['area']}")
    
    elif mask_info['issue_type'] == 'internal_holes':
        issues.append(f"Internal holes: {mask_info['hole_area']} pixels in holes")
        issues.append(f"  {mask_info['num_components']} components detected")
    
    elif mask_info['issue_type'] == 'size':
        if mask_info.get('is_too_small', False):
            issues.append(f"Size: Too small ({mask_info['area']} < {mask_info['min_threshold']})")
        elif mask_info.get('is_too_large', False):
            issues.append(f"Size: Too large ({mask_info['area']} > {mask_info['max_threshold']})")
    
    elif mask_info['issue_type'] == 'convexity':
        issues.append(f"Convexity: Low ratio ({mask_info['convexity_ratio']:.3f} < 0.7)")
        issues.append(f"  Contour area: {mask_info['contour_area']}, Hull area: {mask_info['hull_area']}")
    
    elif mask_info['issue_type'] == 'motion':
        issues.append(f"Motion: Large displacement ({mask_info['distance']:.1f} > {mask_info['max_threshold']})")
        issues.append(f"  From frame {mask_info['prev_frame_idx']} to {mask_info['frame_idx']}")
    
    elif mask_info['issue_type'] == 'boundary':
        issues.append(f"Boundary: High overlap ({mask_info['boundary_ratio']:.3f} > 0.3)")
        issues.append(f"  Boundary overlap: {mask_info['boundary_overlap']} pixels")
    
    elif mask_info['issue_type'] == 'temporal_boundary':
        issues.append(f"Temporal Boundary: {mask_info['inconsistency_type']}")
        issues.append(f"  Boundary ratio: {mask_info['boundary_ratio']:.3f}")
    
    elif mask_info['issue_type'] == 'inconsistent':
        issues.append(f"Inconsistent: Low IoU ({mask_info['iou']:.3f} < 0.5)")
        issues.append(f"  Area change: {mask_info['area_change_ratio']:.3f}, Hausdorff: {mask_info['hausdorff_distance']:.1f}")
    
    elif mask_info['issue_type'] == 'complexity':
        if mask_info.get('is_too_simple', False):
            issues.append(f"Complexity: Too simple ({mask_info['complexity_score']:.1f} < 20)")
        elif mask_info.get('is_too_complex', False):
            issues.append(f"Complexity: Too complex ({mask_info['complexity_score']:.1f} > 200)")
        issues.append(f"  Perimeter: {mask_info['perimeter']}, Area: {mask_info['area']}")
    
    elif mask_info['issue_type'] == 'overlap':
        issues.append(f"Overlap: {mask_info['overlap_pixels']} pixels overlap with annotation {mask_info['annotation2_id']}")
        issues.append(f"  Overlap ratios: {mask_info['overlap_ratio1']:.3f} vs {mask_info['overlap_ratio2']:.3f}")
        issues.append(f"  Areas: {mask_info['area1']} vs {mask_info['area2']} pixels")
    
    # Add a note about editability
    editable_types = ['multi_part', 'skinny', 'smoothness', 'internal', 'internal_holes']
    non_editable_types = ['size', 'convexity', 'motion', 'boundary', 'temporal_boundary', 'inconsistent', 'complexity', 'overlap']
    
    if mask_info['issue_type'] in editable_types:
        issues.append("🔧 Editable issue - automatic fixes available")
    elif mask_info['issue_type'] in non_editable_types:
        issues.append("🔧 Non-editable issue - manual review required")
    
    return issues

def create_component_mask(labels, component_id):
    """Create a binary mask for a specific component"""
    return (labels == component_id).astype(np.uint8)

def create_smoothed_connected_mask(labels, largest_component_id, num_labels):
    """Create a smoothed mask that connects all components to the largest one"""
    # Start with the largest component
    smoothed_mask = (labels == largest_component_id).astype(np.uint8)
    
    # For each other component, find the shortest path to the largest component
    for label in range(1, num_labels):
        if label != largest_component_id:
            # Get the current component
            component_mask = (labels == label).astype(np.uint8)
            
            # Find the closest points between this component and the largest component
            # Use morphological operations to create a bridge
            kernel = np.ones((5, 5), np.uint8)
            dilated_largest = cv2.dilate(smoothed_mask, kernel, iterations=3)
            dilated_component = cv2.dilate(component_mask, kernel, iterations=3)
            
            # Find intersection (bridge region)
            bridge = dilated_largest & dilated_component
            
            # If there's a bridge, connect them
            if np.any(bridge):
                # Create a smooth connection by dilating the bridge
                bridge_kernel = np.ones((3, 3), np.uint8)
                smooth_bridge = cv2.dilate(bridge.astype(np.uint8), bridge_kernel, iterations=2)
                
                # Add the component and the bridge to the smoothed mask
                smoothed_mask = np.logical_or(smoothed_mask, component_mask).astype(np.uint8)
                smoothed_mask = np.logical_or(smoothed_mask, smooth_bridge).astype(np.uint8)
            else:
                # If no bridge found, just add the component as is
                smoothed_mask = np.logical_or(smoothed_mask, component_mask).astype(np.uint8)
    
    return smoothed_mask

def print_issue_summary(mask_info, video, data):
    """Print detailed summary of the issue based on its type"""
    print(f"\n📊 ISSUE SUMMARY:")
    
    if mask_info['issue_type'] == 'multi_part':
        print(f"   • Multi-part mask detected")
        print(f"   • Number of disconnected parts: {mask_info['num_components']}")
        print(f"   • Component areas: {mask_info['component_sizes']}")
        print(f"   • Largest component area: {mask_info['largest_component_size']}")
        print(f"   • Total mask area: {mask_info['total_mask_size']}")
        
        # Calculate area ratios
        largest_ratio = mask_info['largest_component_size'] / mask_info['total_mask_size'] * 100
        print(f"   • Largest component is {largest_ratio:.1f}% of total area")
        
        # Check if largest component dominates
        if largest_ratio > 80:
            print(f"   • ⚠️  Largest component dominates (>80%) - likely good candidate for merge")
        elif largest_ratio < 50:
            print(f"   • ⚠️  No clear dominant component (<50%) - may need manual review")
    
    elif mask_info['issue_type'] == 'boundary_skinny':
        print(f"   • Boundary skinny parts detected")
        print(f"   • Number of boundary skinny parts: {mask_info['num_boundary_skinny_parts']}")
        print(f"   • Boundary skinny parts details: {mask_info['boundary_skinny_parts']}")
        
        # Analyze the skinny parts
        for i, part in enumerate(mask_info['boundary_skinny_parts']):
            boundary, pos, width, height = part
            print(f"     - Part {i+1}: {boundary} boundary at position {pos}, size {width}x{height}")
    
    elif mask_info['issue_type'] == 'size':
        print(f"   • Size issue detected")
        print(f"   • Current area: {mask_info['area']} pixels")
        print(f"   • Min threshold: {mask_info['min_threshold']}, Max threshold: {mask_info['max_threshold']}")
        
        if mask_info['is_too_small']:
            print(f"   • ⚠️  Mask is too small ({mask_info['area']} < {mask_info['min_threshold']})")
            print(f"   •   - Could be noise or partial fish")
        elif mask_info['is_too_large']:
            print(f"   • ⚠️  Mask is too large ({mask_info['area']} > {mask_info['max_threshold']})")
            print(f"   •   - Could be multiple fish or annotation error")
    
    elif mask_info['issue_type'] == 'convexity':
        print(f"   • Convexity issue detected")
        print(f"   • Convexity ratio: {mask_info['convexity_ratio']:.3f} (should be >0.7)")
        print(f"   • Contour area: {mask_info['contour_area']}")
        print(f"   • Convex hull area: {mask_info['hull_area']}")
        
        if mask_info['convexity_ratio'] < 0.5:
            print(f"   • ⚠️  Highly concave (<0.5) - likely annotation error")
        elif mask_info['convexity_ratio'] < 0.7:
            print(f"   • ⚠️  Moderately concave (<0.7) - may need review")
    
    elif mask_info['issue_type'] == 'motion':
        print(f"   • Motion tracking issue detected")
        print(f"   • Distance moved: {mask_info['distance']:.1f} pixels")
        print(f"   • Max allowed distance: {mask_info['max_threshold']}")
        print(f"   • Previous center: {mask_info['center1']}")
        print(f"   • Current center: {mask_info['center2']}")
        
        if mask_info['distance'] > mask_info['max_threshold'] * 2:
            print(f"   • ⚠️  Extreme teleportation (>2x threshold) - likely tracking error")
        elif mask_info['distance'] > mask_info['max_threshold']:
            print(f"   • ⚠️  Moderate teleportation - may be legitimate fast movement")
    
    elif mask_info['issue_type'] == 'boundary':
        print(f"   • Boundary issue detected")
        print(f"   • Boundary ratio: {mask_info['boundary_ratio']:.3f} (should be <0.3)")
        print(f"   • Touches boundary: {mask_info['touches_boundary']}")
        print(f"   • Boundary overlap area: {mask_info['boundary_overlap']}")
        print(f"   • Total mask area: {mask_info['total_area']}")
        
        if mask_info['boundary_ratio'] > 0.5:
            print(f"   • ⚠️  High boundary overlap (>50%) - likely fish entering/exiting frame")
        elif mask_info['boundary_ratio'] > 0.3:
            print(f"   • ⚠️  Moderate boundary overlap (>30%) - may need review")
    
    elif mask_info['issue_type'] == 'temporal_boundary':
        print(f"   • Temporal boundary issue detected")
        print(f"   • Inconsistency type: {mask_info['inconsistency_type']}")
        print(f"   • Current frame touches boundary: {mask_info['current_touches_boundary']}")
        print(f"   • Previous frame touches boundary: {mask_info['prev_touches_boundary']}")
        print(f"   • Next frame touches boundary: {mask_info['next_touches_boundary']}")
        print(f"   • Boundary ratio: {mask_info['boundary_ratio']:.3f}")
        print(f"   • Boundary overlap area: {mask_info['boundary_overlap']}")
        
        # Provide context based on inconsistency type
        if mask_info['inconsistency_type'] == 'isolated_boundary_contact':
            print(f"   • ⚠️  Fish touches boundary only in this frame - likely annotation error")
        elif mask_info['inconsistency_type'] == 'sudden_boundary_appearance':
            print(f"   • ⚠️  Fish suddenly appears at boundary - may be annotation error")
        elif mask_info['inconsistency_type'] == 'sudden_boundary_disappearance':
            print(f"   • ⚠️  Fish suddenly disappears from boundary - may be annotation error")
    
    elif mask_info['issue_type'] == 'smoothness':
        print(f"   • Smoothness issue detected")
        print(f"   • Smoothness ratio: {mask_info['smoothness_ratio']:.6f} (should be >0.01)")
        print(f"   • Perimeter: {mask_info['perimeter']:.1f}")
        print(f"   • Area: {mask_info['area']}")
        
        if mask_info['smoothness_ratio'] < 0.005:
            print(f"   • ⚠️  Very rough boundary (<0.005) - likely pixelated annotation")
        elif mask_info['smoothness_ratio'] < 0.01:
            print(f"   • ⚠️  Rough boundary (<0.01) - may need smoothing")
    
    elif mask_info['issue_type'] == 'internal_holes':
        print(f"   • Internal holes detected")
        print(f"   • Hole area: {mask_info['hole_area']}")
        print(f"   • Number of components: {mask_info['num_components']}")
        
        # Get more detailed hole information
        annotation = data['annotations'][mask_info['annotation_idx']]
        segmentation = annotation['segmentations'][mask_info['frame_idx']]
        mask = decode_rle(segmentation, video['height'], video['width'])
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Find holes using flood fill
        h, w = binary_mask.shape
        filled = binary_mask.copy()
        cv2.floodFill(filled, None, (0, 0), 1)
        holes = binary_mask - filled
        
        # Count individual holes
        hole_labels, hole_count = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
        print(f"   • Number of distinct holes: {hole_count - 1}")  # Subtract 1 for background
        
        if hole_count > 1:
            print(f"   • ⚠️  Multiple holes detected - may need manual review")
        elif mask_info['hole_area'] > mask_info['total_area'] * 0.1:
            print(f"   • ⚠️  Large holes (>10% of total area) - may need review")
    
    elif mask_info['issue_type'] == 'complexity':
        print(f"   • Shape complexity issue detected")
        print(f"   • Complexity score: {mask_info['complexity_score']:.1f}")
        print(f"   • Perimeter: {mask_info['perimeter']:.1f}")
        print(f"   • Area: {mask_info['area']}")
        print(f"   • Too simple: {mask_info['is_too_simple']}")
        print(f"   • Too complex: {mask_info['is_too_complex']}")
        
        if mask_info['is_too_simple']:
            print(f"   • ⚠️  Shape is too simple (<20) - may be noise or partial fish")
        elif mask_info['is_too_complex']:
            print(f"   • ⚠️  Shape is too complex (>200) - may be multiple fish or annotation error")
    
    elif mask_info['issue_type'] == 'inconsistent':
        print(f"   • Temporal inconsistency detected")
        print(f"   • IoU with previous frame: {mask_info['iou']:.3f} (should be >0.5)")
        print(f"   • Area change ratio: {mask_info['area_change_ratio']:.3f} (should be <0.5)")
        print(f"   • Hausdorff distance: {mask_info['hausdorff_distance']:.1f} (should be <100)")
        print(f"   • Previous frame area: {mask_info['area1']}")
        print(f"   • Current frame area: {mask_info['area2']}")
        
        if mask_info['iou'] < 0.3:
            print(f"   • ⚠️  Very low IoU (<0.3) - likely tracking error")
        elif mask_info['area_change_ratio'] > 0.8:
            print(f"   • ⚠️  Large area change (>80%) - may be legitimate or error")
        elif mask_info['hausdorff_distance'] > 200:
            print(f"   • ⚠️  Large shape change (>200px) - may need review")
    
    # Check if this is a non-editable issue type
    non_editable_types = ['size', 'convexity', 'motion', 'boundary', 'temporal_boundary', 'inconsistent', 'complexity']
    if mask_info['issue_type'] in non_editable_types:
        print(f"   • 🔧 Non-editable issue - no automatic fixes available")
        print(f"   •   - Manual review required")
        print(f"   •   - Use 'e' to mark for manual editing, 's' to skip temporarily")
    
    print(f"   • Mask hash: {mask_info['mask_hash'][:8]}...")
    print()

def show_review_progress(current_index, total_issues, review_results):
    """Show current review progress and statistics"""
    print(f"\n📈 REVIEW PROGRESS:")
    print(f"   • Current position: {current_index + 1}/{total_issues}")
    print(f"   • Progress: {(current_index + 1)/total_issues*100:.1f}%")
    print(f"   • Remaining: {total_issues - current_index - 1} issues")
    
    print(f"\n📊 DECISIONS SO FAR:")
    print(f"   • Edited masks: {len(review_results['edited'])}")
    print(f"   • Manual edit required: {len(review_results['skipped'])}")
    
    if review_results['edited'] or review_results['skipped']:
        total_decided = len(review_results['edited']) + len(review_results['skipped'])
        print(f"   • Total decided: {total_decided}")
        print(f"   • Decision rate: {total_decided/(current_index + 1)*100:.1f}%")
    
    print()

def get_component_bbox(component_mask):
    """Get bounding box for a component"""
    rows = np.any(component_mask, axis=1)
    cols = np.any(component_mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return x1, y1, x2, y2

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

def capture_mask_details(mask_info, data, video, binary_mask, cleaned_data=None):
    """Capture detailed information about a mask for tracking purposes"""
    
    annotation = data['annotations'][mask_info['annotation_idx']]
    frame_idx = mask_info['frame_idx']
    video_id = mask_info['video_id']
    
    # Get video file name
    video_file_name = video['file_names'][frame_idx] if frame_idx < len(video['file_names']) else f"frame_{frame_idx}"
    
    # Get original mask data
    original_segmentation = annotation['segmentations'][frame_idx]
    original_mask_hash = hashlib.md5(binary_mask.tobytes()).hexdigest()
    
    # Calculate original mask statistics
    original_area = np.sum(binary_mask)
    original_bbox = None
    if original_area > 0:
        # Find bounding box
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            original_bbox = [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]
    
    mask_details = {
        'video_id': video_id,
        'annotation_id': mask_info['annotation_id'],
        'frame_idx': frame_idx,
        'video_file_name': video_file_name,
        'category_id': mask_info['category_id'],
        'category_name': mask_info['category_name'],
        'original_segmentation': original_segmentation,
        'original_mask_hash': original_mask_hash,
        'original_area': int(original_area),
        'original_bbox': original_bbox,
        'video_height': video['height'],
        'video_width': video['width'],
        'issue_types': [issue['issue_type'] for issue in mask_info.get('issues', [mask_info])]
    }
    
    # If we have cleaned data, get the edited mask information
    if cleaned_data is not None:
        try:
            cleaned_annotation = cleaned_data['annotations'][mask_info['annotation_idx']]
            cleaned_segmentation = cleaned_annotation['segmentations'][frame_idx]
            
            if cleaned_segmentation is not None:
                cleaned_mask = decode_rle(cleaned_segmentation, video['height'], video['width'])
                cleaned_binary_mask = (cleaned_mask > 0).astype(np.uint8)
                cleaned_mask_hash = hashlib.md5(cleaned_binary_mask.tobytes()).hexdigest()
                cleaned_area = np.sum(cleaned_binary_mask)
                
                # Calculate edited mask statistics
                cleaned_bbox = None
                if cleaned_area > 0:
                    rows = np.any(cleaned_binary_mask, axis=1)
                    cols = np.any(cleaned_binary_mask, axis=0)
                    if np.any(rows) and np.any(cols):
                        rmin, rmax = np.where(rows)[0][[0, -1]]
                        cmin, cmax = np.where(cols)[0][[0, -1]]
                        cleaned_bbox = [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]
                
                mask_details.update({
                    'edited_segmentation': cleaned_segmentation,
                    'edited_mask_hash': cleaned_mask_hash,
                    'edited_area': int(cleaned_area),
                    'edited_bbox': cleaned_bbox,
                    'area_change': int(max(-original_area, cleaned_area - original_area)),  # Prevent overflow
                    'area_change_percentage': float((cleaned_area - original_area) / original_area * 100) if original_area > 0 else 0.0,
                    'mask_changed': cleaned_mask_hash != original_mask_hash
                })
            else:
                mask_details.update({
                    'edited_segmentation': None,
                    'edited_mask_hash': None,
                    'edited_area': 0,
                    'edited_bbox': None,
                    'area_change': -int(original_area),  # This is safe as it's just the negative of original_area
                    'area_change_percentage': -100.0 if original_area > 0 else 0.0,
                    'mask_changed': True
                })
        except (IndexError, KeyError) as e:
            print(f"Warning: Could not get cleaned mask data for {mask_info['annotation_id']}, frame {frame_idx}: {e}")
            mask_details.update({
                'edited_segmentation': None,
                'edited_mask_hash': None,
                'edited_area': 0,
                'edited_bbox': None,
                'area_change': -int(original_area),  # This is safe as it's just the negative of original_area
                'area_change_percentage': -100.0 if original_area > 0 else 0.0,
                'mask_changed': True
            })
    
    return mask_details

def save_detailed_tracking_data(tracking_data, review_results, data, cleaned_data, tracking_file):
    """Save detailed tracking data including original and edited mask information"""
    
    # Get videos dictionary
    videos = {v['id']: v for v in data['videos']}
    
    # Initialize detailed tracking structure
    if 'detailed_edits' not in tracking_data:
        tracking_data['detailed_edits'] = []
    
    # Process edited masks
    for edit_info in review_results['edited']:
        mask_group = edit_info['mask_group']
        primary_issue = mask_group['primary_issue']
        
        # Get original mask data
        annotation = data['annotations'][primary_issue['annotation_idx']]
        segmentation = annotation['segmentations'][primary_issue['frame_idx']]
        original_mask = decode_rle(segmentation, primary_issue.get('video_height', 960), primary_issue.get('video_width', 1280))
        original_binary_mask = (original_mask > 0).astype(np.uint8)
        
        # Capture detailed mask information
        mask_details = capture_mask_details(primary_issue, data, videos[primary_issue['video_id']], original_binary_mask, cleaned_data)
        
        # Add edit information
        mask_details.update({
            'edit_type': 'immediate_edit',
            'edit_date': datetime.now().isoformat(),
            'option_selected': edit_info['option_selected'],
            'option_id': edit_info['option_id'],
            'all_issues': [issue['issue_type'] for issue in mask_group['issues']]
        })
        
        # Add information about the changes made (for immediate edits)
        if edit_info['option_selected'] != 'keep_original':
            # Get the updated annotation from cleaned data
            cleaned_annotation = cleaned_data['annotations'][primary_issue['annotation_idx']]
            updated_segmentation = cleaned_annotation['segmentations'][primary_issue['frame_idx']]
            
            if updated_segmentation is not None:
                updated_mask = decode_rle(updated_segmentation, videos[primary_issue['video_id']]['height'], videos[primary_issue['video_id']]['width'])
                updated_binary_mask = (updated_mask > 0).astype(np.uint8)
                
                # Calculate updated statistics
                updated_area = np.sum(updated_binary_mask)
                updated_bbox = calculate_bbox_from_mask(updated_binary_mask)
                
                mask_details.update({
                    'updated_area': int(updated_area),
                    'updated_bbox': updated_bbox,
                    'updated_segmentation': updated_segmentation,  # Include the actual RLE segmentation
                    'area_change_from_edit': int(max(-mask_details['original_area'], updated_area - mask_details['original_area'])),  # Prevent overflow
                    'bbox_change_from_edit': {
                        'original': mask_details['original_bbox'],
                        'updated': updated_bbox
                    }
                })
        
        tracking_data['detailed_edits'].append(mask_details)
    
    # Process manually edited masks (marked with 'e')
    for skip_info in review_results['skipped']:
        mask_group = skip_info
        primary_issue = mask_group['primary_issue']
        
        # Get original mask data
        annotation = data['annotations'][primary_issue['annotation_idx']]
        segmentation = annotation['segmentations'][primary_issue['frame_idx']]
        original_mask = decode_rle(segmentation, primary_issue.get('video_height', 960), primary_issue.get('video_width', 1280))
        original_binary_mask = (original_mask > 0).astype(np.uint8)
        
        # Capture detailed mask information
        mask_details = capture_mask_details(primary_issue, data, videos[primary_issue['video_id']], original_binary_mask, cleaned_data)
        
        # Add edit information
        mask_details.update({
            'edit_type': 'manual_edit_required',
            'edit_date': datetime.now().isoformat(),
            'option_selected': 'manual_edit',
            'option_id': 'e',
            'all_issues': [issue['issue_type'] for issue in mask_group['issues']]
        })
        
        tracking_data['detailed_edits'].append(mask_details)
    
    # Save the detailed tracking data
    save_review_tracking(tracking_data, tracking_file)
    
    return tracking_data

def analyze_tracking_data(tracking_file, output_dir=None):
    """Analyze the detailed tracking data and create a summary report"""
    
    if not os.path.exists(tracking_file):
        print(f"Tracking file not found: {tracking_file}")
        return
    
    # Load tracking data
    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)
    
    if 'detailed_edits' not in tracking_data:
        print("No detailed edits found in tracking data")
        return
    
    detailed_edits = tracking_data['detailed_edits']
    
    print(f"\n=== DETAILED TRACKING ANALYSIS ===")
    print(f"Total edits tracked: {len(detailed_edits)}")
    
    # Statistics
    immediate_edits = [edit for edit in detailed_edits if edit['edit_type'] == 'immediate_edit']
    manual_edits = [edit for edit in detailed_edits if edit['edit_type'] == 'manual_edit_required']
    
    print(f"Immediate edits: {len(immediate_edits)}")
    print(f"Manual edits required: {len(manual_edits)}")
    
    # Issue type statistics
    all_issues = []
    for edit in detailed_edits:
        all_issues.extend(edit['all_issues'])
    
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    print(f"\nIssue types found:")
    for issue_type, count in sorted(issue_counts.items()):
        print(f"  {issue_type}: {count}")
    
    # Area change statistics
    area_changes = [edit['area_change'] for edit in detailed_edits if 'area_change' in edit]
    area_percentages = [edit['area_change_percentage'] for edit in detailed_edits if 'area_change_percentage' in edit]
    
    if area_changes:
        print(f"\nArea change statistics:")
        print(f"  Average area change: {np.mean(area_changes):.1f} pixels")
        print(f"  Median area change: {np.median(area_changes):.1f} pixels")
        print(f"  Min area change: {min(area_changes)} pixels")
        print(f"  Max area change: {max(area_changes)} pixels")
        print(f"  Average percentage change: {np.mean(area_percentages):.1f}%")
    
    # Masks that changed vs didn't change
    changed_masks = [edit for edit in detailed_edits if edit.get('mask_changed', False)]
    unchanged_masks = [edit for edit in detailed_edits if not edit.get('mask_changed', False)]
    
    print(f"\nMask change statistics:")
    print(f"  Masks that changed: {len(changed_masks)}")
    print(f"  Masks that didn't change: {len(unchanged_masks)}")
    
    # Create detailed report
    if output_dir:
        report_file = os.path.join(output_dir, "detailed_tracking_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_edits': len(detailed_edits),
                    'immediate_edits': len(immediate_edits),
                    'manual_edits': len(manual_edits),
                    'changed_masks': len(changed_masks),
                    'unchanged_masks': len(unchanged_masks),
                    'issue_type_counts': issue_counts,
                    'area_change_stats': {
                        'mean': float(np.mean(area_changes)) if area_changes else 0,
                        'median': float(np.median(area_changes)) if area_changes else 0,
                        'min': min(area_changes) if area_changes else 0,
                        'max': max(area_changes) if area_changes else 0,
                        'mean_percentage': float(np.mean(area_percentages)) if area_percentages else 0
                    }
                },
                'detailed_edits': detailed_edits
            }, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")
    
    return detailed_edits

def create_mask_comparison_report(tracking_file, original_json_file, cleaned_json_file, output_dir):
    """Create a detailed report comparing original and cleaned masks"""
    
    if not os.path.exists(tracking_file):
        print(f"Tracking file not found: {tracking_file}")
        return
    
    # Load tracking data
    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)
    
    if 'detailed_edits' not in tracking_data:
        print("No detailed edits found in tracking data")
        return
    
    detailed_edits = tracking_data['detailed_edits']
    
    # Load original and cleaned data
    with open(original_json_file, 'r') as f:
        original_data = json.load(f)
    
    with open(cleaned_json_file, 'r') as f:
        cleaned_data = json.load(f)
    
    # Create comparison report
    comparison_report = {
        'metadata': {
            'original_file': original_json_file,
            'cleaned_file': cleaned_json_file,
            'tracking_file': tracking_file,
            'total_edits': len(detailed_edits),
            'report_date': datetime.now().isoformat()
        },
        'mask_comparisons': []
    }
    
    for edit in detailed_edits:
        video_id = edit['video_id']
        annotation_id = edit['annotation_id']
        frame_idx = edit['frame_idx']
        
        # Find corresponding annotations
        original_annotation = None
        cleaned_annotation = None
        
        for ann in original_data['annotations']:
            if ann['id'] == annotation_id:
                original_annotation = ann
                break
        
        for ann in cleaned_data['annotations']:
            if ann['id'] == annotation_id:
                cleaned_annotation = ann
                break
        
        comparison = {
            'video_id': video_id,
            'annotation_id': annotation_id,
            'frame_idx': frame_idx,
            'video_file_name': edit['video_file_name'],
            'category_name': edit['category_name'],
            'edit_type': edit['edit_type'],
            'option_selected': edit['option_selected'],
            'all_issues': edit['all_issues'],
            'original_mask': {
                'segmentation': edit['original_segmentation'],
                'area': edit['original_area'],
                'bbox': edit['original_bbox'],
                'mask_hash': edit['original_mask_hash']
            },
            'edited_mask': {
                'segmentation': edit.get('edited_segmentation'),
                'area': edit.get('edited_area', 0),
                'bbox': edit.get('edited_bbox'),
                'mask_hash': edit.get('edited_mask_hash')
            },
            'changes': {
                'area_change': edit.get('area_change', 0),
                'area_change_percentage': edit.get('area_change_percentage', 0),
                'mask_changed': edit.get('mask_changed', False)
            }
        }
        
        comparison_report['mask_comparisons'].append(comparison)
    
    # Save comparison report
    comparison_file = os.path.join(output_dir, "mask_comparison_report.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    print(f"\nMask comparison report saved to: {comparison_file}")
    print(f"Total masks compared: {len(comparison_report['mask_comparisons'])}")
    
    return comparison_report

def create_unified_visualization(mask_info, video, binary_mask, image_root, issue_idx, output_dir, issue_type):
    """Create unified visualization for all issue types with numbered options
    
    Args:
        mask_info: Information about the mask being reviewed
        video: Video metadata
        binary_mask: The binary mask being reviewed
        image_root: Path to image directory
        issue_idx: Index for saving images
        output_dir: Output directory for saved images
        issue_type: Type of issue ('multi_part', 'skinny', 'smoothness', 'internal', 'internal_holes', 'inconsistent')
    """
    # Get mask options for the specific issue type
    mask_options = get_mask_options(binary_mask, issue_type)
    
    # Create visualization with options
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Set title based on issue type
    issue_titles = {
        'multi_part': f'Multi-part Mask: {mask_info["category_name"]}',
        'skinny': f'Skinny Mask: {mask_info["category_name"]}',
        'smoothness': f'Smoothness Issue: {mask_info["category_name"]}',
        'internal': f'Internal Holes: {mask_info["category_name"]}',
        'internal_holes': f'Internal Holes: {mask_info["category_name"]}',
        'inconsistent': f'Inconsistent Mask: {mask_info["category_name"]}'
    }
    
    title = issue_titles.get(issue_type, f'{issue_type.title()} Issue: {mask_info["category_name"]}')
    title += f' (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})'
    
    # Add special handling for inconsistent masks (shows frame range)
    if issue_type == 'inconsistent':
        title = f'Inconsistent Mask: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frames {mask_info["prev_frame_idx"]}-{mask_info["frame_idx"]})'
    
    fig.suptitle(title, fontsize=14)
    
    # Load and display frames using helper function (no mask overlay)
    load_and_display_frames(mask_info, video, image_root, axes)
    
    # Display mask options with numbers and highlighting
    for i, option in enumerate(mask_options['options']):
        if i < 3:  # Show up to 3 options in the bottom row
            row, col = 1, i
            
            # Create colored mask visualization
            mask = option['mask']
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            
            # Highlight the fish being reviewed in red
            colored_mask[mask > 0] = [255, 0, 0]  # Red for the fish being reviewed
            
            axes[row, col].imshow(colored_mask)
            axes[row, col].set_title(f'{option["id"]}: {option["name"]}')
            axes[row, col].axis('off')
    
    # Add option descriptions
    description_text = "Options:\n"
    for i, desc in enumerate(mask_options['descriptions']):
        description_text += f"{i+1}: {desc}\n"
    
    fig.text(0.02, 0.02, description_text, fontsize=10, transform=fig.transFigure, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save and display the image
    temp_image_path = save_and_display_image(fig, output_dir, mask_info, issue_idx)
    
    return temp_image_path, mask_options

def detect_overlapping_annotations(data, video_id, frame_idx, video_height, video_width):
    """Detect overlapping annotations within the same frame
    
    Args:
        data: The dataset dictionary
        video_id: ID of the video
        frame_idx: Frame index to check
        video_height: Height of the video
        video_width: Width of the video
        
    Returns:
        List of overlap issues found
    """
    overlap_issues = []
    
    # Get all annotations for this video and frame
    frame_annotations = []
    for annotation in data['annotations']:
        if (annotation['video_id'] == video_id and 
            len(annotation['segmentations']) > frame_idx and 
            annotation['segmentations'][frame_idx] is not None):
            frame_annotations.append(annotation)
    
    # If there's only one annotation, no overlap possible
    if len(frame_annotations) <= 1:
        return overlap_issues
    
    # Check for overlaps between all pairs of annotations
    for i, ann1 in enumerate(frame_annotations):
        for j, ann2 in enumerate(frame_annotations[i+1:], i+1):
            try:
                # Decode both masks
                mask1 = decode_rle(ann1['segmentations'][frame_idx], video_height, video_width)
                mask2 = decode_rle(ann2['segmentations'][frame_idx], video_height, video_width)
                
                binary_mask1 = (mask1 > 0).astype(np.uint8)
                binary_mask2 = (mask2 > 0).astype(np.uint8)
                
                # Calculate overlap
                overlap = np.sum(binary_mask1 & binary_mask2)
                area1 = np.sum(binary_mask1)
                area2 = np.sum(binary_mask2)
                
                # Calculate overlap ratios
                overlap_ratio1 = overlap / area1 if area1 > 0 else 0
                overlap_ratio2 = overlap / area2 if area2 > 0 else 0
                
                # Flag as issue if there's significant overlap (>10% of either annotation)
                if overlap > 0 and (overlap_ratio1 > 0.1 or overlap_ratio2 > 0.1):
                    overlap_issues.append({
                        'video_id': video_id,
                        'frame_idx': frame_idx,
                        'annotation1_id': ann1['id'],
                        'annotation2_id': ann2['id'],
                        'overlap_pixels': int(overlap),
                        'area1': int(area1),
                        'area2': int(area2),
                        'overlap_ratio1': float(overlap_ratio1),
                        'overlap_ratio2': float(overlap_ratio2),
                        'issue_type': 'overlap',
                        'category_id': ann1['category_id']
                    })
                    
            except Exception as e:
                print(f"Warning: Could not check overlap between annotations {ann1['id']} and {ann2['id']}: {e}")
                continue
    
    return overlap_issues

def add_overlap_issues_to_mask_issues(mask_issues, data):
    """Add overlap issues to the existing mask issues list"""
    print("Checking for overlapping annotations...")
    
    # Get unique video-frame combinations from existing issues
    video_frames = set()
    for issue in mask_issues:
        video_frames.add((issue['video_id'], issue['frame_idx']))
    
    # Check for overlaps in each video-frame combination
    overlap_issues = []
    for video_id, frame_idx in video_frames:
        # Get video info
        video = None
        for v in data['videos']:
            if v['id'] == video_id:
                video = v
                break
        
        if video is None:
            continue
            
        # Check for overlaps in this frame
        frame_overlaps = detect_overlapping_annotations(data, video_id, frame_idx, video['height'], video['width'])
        
        # Add annotation indices to overlap issues
        for overlap in frame_overlaps:
            # Find the annotation indices
            ann1_idx = None
            ann2_idx = None
            for i, annotation in enumerate(data['annotations']):
                if annotation['id'] == overlap['annotation1_id']:
                    ann1_idx = i
                elif annotation['id'] == overlap['annotation2_id']:
                    ann2_idx = i
                if ann1_idx is not None and ann2_idx is not None:
                    break
            
            if ann1_idx is not None and ann2_idx is not None:
                # Add both annotations as overlap issues
                for ann_idx in [ann1_idx, ann2_idx]:
                    overlap_issue = overlap.copy()
                    overlap_issue['annotation_idx'] = ann_idx
                    overlap_issue['annotation_id'] = data['annotations'][ann_idx]['id']
                    overlap_issue['category_id'] = data['annotations'][ann_idx]['category_id']
                    overlap_issue['category_name'] = next((cat['name'] for cat in data['categories'] if cat['id'] == overlap_issue['category_id']), 'unknown')
                    overlap_issues.append(overlap_issue)
    
    # Add overlap issues to the main list
    mask_issues.extend(overlap_issues)
    print(f"Added {len(overlap_issues)} overlap issues")
    
    return mask_issues

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive mask review for fish dataset')
    parser.add_argument('--json_file', default="/data/fishway_ytvis/all_videos.json", 
                       help='Path to the JSON dataset file')
    parser.add_argument('--cleaned_json_file', 
                       help='Path to the cleaned JSON file (if not provided, will use output_dir/all_videos_cleaned.json)')
    parser.add_argument('--image_root', default="/data/fishway_ytvis/all_videos", 
                       help='Path to the image root directory')
    parser.add_argument('--output_dir', default="/data/fishway_ytvis/mask_validation_results", 
                       help='Path to the output directory')
    parser.add_argument('--mask_issues_file', 
                       help='Path to the mask issues JSON file (if not provided, will look in output_dir)')
    parser.add_argument('--force', action='store_true', 
                       help='Force review even if already reviewed')
    parser.add_argument('--analyze_tracking', action='store_true',
                       help='Analyze existing tracking data and create summary report')
    parser.add_argument('--create_comparison_report', action='store_true',
                       help='Create detailed comparison report between original and cleaned datasets')
    parser.add_argument('--check_overlaps', action='store_true',
                       help='Check for overlapping annotations and add them to the review list')
    
    args = parser.parse_args()
    
    # Configuration
    json_file = args.json_file
    image_root = args.image_root
    output_dir = args.output_dir
    
    # Determine cleaned JSON file
    if args.cleaned_json_file:
        cleaned_json_file = args.cleaned_json_file
    else:
        cleaned_json_file = os.path.join(output_dir, "all_videos_cleaned.json")
    
    # Determine mask issues file
    if args.mask_issues_file:
        mask_issues_file = args.mask_issues_file
    else:
        mask_issues_file = os.path.join(output_dir, "mask_issues.json")
    
    # Handle analysis options
    if args.analyze_tracking:
        tracking_file = os.path.join(output_dir, "mask_review_tracking.json")
        print("=== ANALYZING TRACKING DATA ===")
        analyze_tracking_data(tracking_file, output_dir)
        exit(0)
    
    if args.create_comparison_report:
        tracking_file = os.path.join(output_dir, "mask_review_tracking.json")
        print("=== CREATING COMPARISON REPORT ===")
        create_mask_comparison_report(tracking_file, json_file, cleaned_json_file, output_dir)
        exit(0)
    
    # Check if mask issues file exists
    if not os.path.exists(mask_issues_file):
        print(f"Error: Mask issues file not found: {mask_issues_file}")
        print("Please run 02_mask_outlier_check.py first to generate the mask issues file.")
        exit(1)
    
    # Load mask issues
    print("=== INTERACTIVE MASK REVIEW ===")
    print(f"Loading mask issues from: {mask_issues_file}")
    
    with open(mask_issues_file, 'r') as f:
        mask_issues = json.load(f)
    
    print(f"Loaded {len(mask_issues)} mask issues")
    
    # Add overlap issues to the mask issues list if requested
    if args.check_overlaps:
        mask_issues = add_overlap_issues_to_mask_issues(mask_issues, data)
    
    # Start interactive review
    tracking_file = os.path.join(output_dir, "mask_review_tracking.json")
    review_results = interactive_mask_review_with_tracking(
        json_file, cleaned_json_file, image_root, mask_issues, output_dir, tracking_file
    )
    
    print("\n=== REVIEW COMPLETE ===")
    print(f"Review results saved to: {output_dir}")