import json
import os
import cv2
import numpy as np
from pycocotools import mask as mask_util
import copy

def decode_rle(rle_obj, height, width):
    """Decode RLE mask to binary array"""
    if isinstance(rle_obj['counts'], list):
        rle = mask_util.frPyObjects(rle_obj, height, width)
    else:
        rle = rle_obj
    return mask_util.decode(rle)

def encode_rle(mask):
    """Encode binary mask to RLE format"""
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

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

def detect_holes(mask):
    """Detect holes in a mask by finding enclosed regions (0s that don't touch the boundary)"""
    binary_mask = (mask > 0).astype(np.uint8)
    h, w = binary_mask.shape
    
    # Create a mask of background (inverse of the binary mask)
    background = 1 - binary_mask
    
    # Find connected components of the background
    num_labels, labels = cv2.connectedComponents(background, connectivity=8)
    
    # Check which background components touch the boundary
    boundary_touching = set()
    
    # Check top and bottom rows
    for x in range(w):
        if labels[0, x] > 0:
            boundary_touching.add(labels[0, x])
        if labels[h-1, x] > 0:
            boundary_touching.add(labels[h-1, x])
    
    # Check left and right columns
    for y in range(h):
        if labels[y, 0] > 0:
            boundary_touching.add(labels[y, 0])
        if labels[y, w-1] > 0:
            boundary_touching.add(labels[y, w-1])
    
    # Holes are background components that don't touch the boundary
    holes = []
    for label in range(1, num_labels):
        if label not in boundary_touching:
            hole_mask = (labels == label).astype(np.uint8)
            hole_area = np.sum(hole_mask)
            if hole_area > 0:
                holes.append({
                    'label': label,
                    'mask': hole_mask,
                    'area': hole_area
                })
    
    return holes

def fill_holes(mask):
    """Fill internal holes in a mask"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Detect holes
    holes = detect_holes(binary_mask)
    
    # Fill all holes
    filled_mask = binary_mask.copy()
    for hole in holes:
        filled_mask = filled_mask | hole['mask']
    
    return filled_mask

def fill_holes_except_largest(mask):
    """Fill all holes except the largest one"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Detect holes
    holes = detect_holes(binary_mask)
    
    if not holes:
        return binary_mask
    
    # Find the largest hole
    largest_hole = max(holes, key=lambda h: h['area'])
    
    # Fill all holes except the largest
    filled_mask = binary_mask.copy()
    for hole in holes:
        if hole['label'] != largest_hole['label']:
            filled_mask = filled_mask | hole['mask']
    
    return filled_mask

def remove_small_holes(mask, max_hole_area=5):
    """Remove holes that are smaller than or equal to max_hole_area"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Detect holes
    holes = detect_holes(binary_mask)
    
    # Fill only small holes
    cleaned_mask = binary_mask.copy()
    for hole in holes:
        if hole['area'] <= max_hole_area:
            cleaned_mask = cleaned_mask | hole['mask']
    
    return cleaned_mask

def remove_small_components(mask, max_component_area=5):
    """Remove disconnected components that are smaller than or equal to max_component_area"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    if num_labels <= 1:  # Only background
        return binary_mask
    
    # Analyze each component
    component_sizes = []
    component_masks = []
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        size = np.sum(component_mask)
        component_sizes.append(size)
        component_masks.append(component_mask)
    
    # Keep only components larger than max_component_area
    cleaned_mask = np.zeros_like(binary_mask)
    for i, (size, component_mask) in enumerate(zip(component_sizes, component_masks)):
        if size > max_component_area:
            cleaned_mask = cleaned_mask | component_mask
    
    return cleaned_mask

def clean_mask(mask, max_hole_area=5, max_component_area=5):
    """Remove both small holes and small disconnected components from a mask"""
    # First remove small holes
    mask_no_holes = remove_small_holes(mask, max_hole_area)
    
    # Then remove small components
    cleaned_mask = remove_small_components(mask_no_holes, max_component_area)
    
    return cleaned_mask

def remove_skinny_parts(mask, min_width=10):
    """Remove skinny parts from a mask"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Use morphological operations to remove skinny parts
    kernel = np.ones((min_width, min_width), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    return dilated

def smooth_boundary(mask, iterations=1):
    """Smooth the boundary of a mask"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Use morphological operations to smooth the boundary
    kernel = np.ones((3, 3), np.uint8)
    smoothed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    return smoothed

def get_multi_part_options(mask):
    """Generate different options for multi-part mask editing
    
    Note: This operates on a single fish annotation's mask.
    If there are multiple fish in the frame, each fish has its own annotation
    and this function only affects the fish that was flagged for issues.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    if num_labels <= 1:  # Only background
        return {
            'options': [{'id': 1, 'name': 'Original', 'mask': binary_mask}],
            'descriptions': ['Original mask (no changes)']
        }
    
    # Analyze each component
    component_sizes = []
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        size = np.sum(component_mask)
        component_sizes.append(size)
    
    largest_component_id = np.argmax(component_sizes) + 1
    
    options = []
    descriptions = []
    
    # Option 1: Original mask (all components)
    options.append({'id': 1, 'name': 'Original', 'mask': binary_mask})
    descriptions.append(f'Original mask ({len(component_sizes)} components)')
    
    # Option 2: Smoothed connected mask
    if len(component_sizes) > 1:
        smoothed_mask = create_smoothed_connected_mask(labels, largest_component_id, num_labels)
        options.append({'id': 2, 'name': 'Connected', 'mask': smoothed_mask})
        descriptions.append(f'All components connected (smoothed)')
    
    # Option 3: Largest component only
    largest_mask = create_component_mask(labels, largest_component_id)
    options.append({'id': 3, 'name': 'Largest Only', 'mask': largest_mask})
    descriptions.append(f'Largest component only (size: {component_sizes[largest_component_id-1]})')
    
    return {
        'options': options,
        'descriptions': descriptions
    }

def get_hole_filling_options(mask):
    """Generate different options for hole filling"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Detect holes
    holes = detect_holes(binary_mask)
    has_holes = len(holes) > 0
    
    options = []
    descriptions = []
    
    # Option 1: Original mask
    options.append({'id': 1, 'name': 'Original', 'mask': binary_mask})
    descriptions.append('Original mask (with holes)' if has_holes else 'Original mask')
    
    # Option 2: All holes filled
    if has_holes:
        filled_mask = fill_holes(binary_mask)
        options.append({'id': 2, 'name': 'All Holes Filled', 'mask': filled_mask})
        descriptions.append('All holes filled')
    
    # Option 3: All holes filled except largest
    if has_holes and len(holes) > 1:
        partial_filled_mask = fill_holes_except_largest(binary_mask)
        options.append({'id': 3, 'name': 'Fill Except Largest', 'mask': partial_filled_mask})
        descriptions.append('All holes filled except the largest hole')
    elif has_holes and len(holes) == 1:
        # Only one hole, so this option is same as option 2
        if len(options) == 2:  # Only add if we have option 2
            partial_filled_mask = fill_holes_except_largest(binary_mask)
            options.append({'id': 3, 'name': 'Fill Except Largest', 'mask': partial_filled_mask})
            descriptions.append('All holes filled except the largest hole')
    
    return {
        'options': options,
        'descriptions': descriptions
    }

def get_skinny_removal_options(mask):
    """Generate different options for skinny part removal"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Detect skinny parts
    kernel = np.ones((10, 10), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    skinny_parts = binary_mask - dilated
    has_skinny_parts = np.any(skinny_parts)
    
    options = []
    descriptions = []
    
    # Option 1: Original mask
    options.append({'id': 1, 'name': 'Original', 'mask': binary_mask})
    descriptions.append('Original mask (with skinny parts)' if has_skinny_parts else 'Original mask')
    
    # Option 2: Cleaned mask
    if has_skinny_parts:
        cleaned_mask = remove_skinny_parts(binary_mask)
        options.append({'id': 2, 'name': 'Cleaned', 'mask': cleaned_mask})
        descriptions.append('Skinny parts removed')
    
    return {
        'options': options,
        'descriptions': descriptions
    }

def get_smoothing_options(mask):
    """Generate different options for boundary smoothing"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    options = []
    descriptions = []
    
    # Option 1: Original mask
    options.append({'id': 1, 'name': 'Original', 'mask': binary_mask})
    descriptions.append('Original mask')
    
    # Option 2: Light smoothing
    light_smoothed = smooth_boundary(binary_mask, iterations=1)
    options.append({'id': 2, 'name': 'Light Smooth', 'mask': light_smoothed})
    descriptions.append('Light boundary smoothing')
    
    # Option 3: Heavy smoothing
    heavy_smoothed = smooth_boundary(binary_mask, iterations=2)
    options.append({'id': 3, 'name': 'Heavy Smooth', 'mask': heavy_smoothed})
    descriptions.append('Heavy boundary smoothing')
    
    return {
        'options': options,
        'descriptions': descriptions
    }

def get_generic_options(mask):
    """Generate generic options for other issue types"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    options = []
    descriptions = []
    
    # Option 1: Original mask
    options.append({'id': 1, 'name': 'Original', 'mask': binary_mask})
    descriptions.append('Original mask (no changes)')
    
    return {
        'options': options,
        'descriptions': descriptions
    }

def get_non_editable_options(mask):
    """Generate options for non-editable issues (size, convexity, motion, boundary, inconsistent, complexity)"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    options = []
    descriptions = []
    
    # Option 1: Keep original (no changes)
    options.append({'id': 1, 'name': 'Keep Original', 'mask': binary_mask})
    descriptions.append('Keep original mask (no automatic editing possible)')
    
    return {
        'options': options,
        'descriptions': descriptions
    }

def get_mask_options(mask, issue_type):
    """Get appropriate mask editing options based on issue type"""
    if issue_type == 'multi_part':
        return get_multi_part_options(mask)
    elif issue_type == 'internal' or issue_type == 'internal_holes':
        return get_hole_filling_options(mask)
    elif issue_type == 'skinny':
        return get_skinny_removal_options(mask)
    elif issue_type == 'smoothness':
        return get_smoothing_options(mask)
    elif issue_type in ['size', 'convexity', 'motion', 'boundary', 'inconsistent', 'complexity']:
        return get_non_editable_options(mask)
    else:
        return get_generic_options(mask)

def calculate_bbox_from_mask(mask):
    """Calculate bounding box from a binary mask"""
    binary_mask = (mask > 0).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return [0, 0, 0, 0]  # Empty mask
    
    # Find bounding box
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]
    else:
        return [0, 0, 0, 0]

def calculate_area_from_mask(mask):
    """Calculate area from a binary mask"""
    binary_mask = (mask > 0).astype(np.uint8)
    return int(np.sum(binary_mask))

def apply_mask_edit(data, annotation_idx, frame_idx, new_mask, video_height, video_width):
    """Apply a mask edit to the dataset, updating segmentation, bbox, and area"""
    # Encode the new mask to RLE format
    new_rle = encode_rle(new_mask)
    
    # Calculate new bounding box and area
    new_bbox = calculate_bbox_from_mask(new_mask)
    new_area = calculate_area_from_mask(new_mask)
    
    # Update the annotation
    annotation = data['annotations'][annotation_idx]
    annotation['segmentations'][frame_idx] = new_rle
    
    # Update bounding box if it exists
    if 'bboxes' in annotation and frame_idx < len(annotation['bboxes']):
        annotation['bboxes'][frame_idx] = new_bbox
    
    # Update area if it exists
    if 'areas' in annotation and frame_idx < len(annotation['areas']):
        annotation['areas'][frame_idx] = new_area
    
    return data

def ensure_cleaned_json_exists(original_json_path, cleaned_json_path):
    """Ensure the cleaned JSON file exists and is up to date"""
    if not os.path.exists(cleaned_json_path):
        # Create a copy of the original
        print(f"Creating cleaned JSON file: {cleaned_json_path}")
        with open(original_json_path, 'r') as f:
            data = json.load(f)
        with open(cleaned_json_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    
    # Check if files are compatible
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    with open(cleaned_json_path, 'r') as f:
        cleaned_data = json.load(f)
    
    original_videos = {v['id'] for v in original_data['videos']}
    cleaned_videos = {v['id'] for v in cleaned_data['videos']}
    
    # Check for videos in cleaned that are not in original
    extra_videos = cleaned_videos - original_videos
    if extra_videos:
        print(f"Warning: The following videos in {cleaned_json_path} are not in {original_json_path}:")
        for vid in extra_videos:
            print(f"  - Video {vid}")
    
    # Check for videos in original that are not in cleaned
    missing_videos = original_videos - cleaned_videos
    if missing_videos:
        print(f"Adding {len(missing_videos)} missing videos to {cleaned_json_path}")
        
        # Add missing videos to cleaned data
        for video in original_data['videos']:
            if video['id'] in missing_videos:
                cleaned_data['videos'].append(video)
        
        # Add missing annotations
        for annotation in original_data['annotations']:
            if annotation['video_id'] in missing_videos:
                cleaned_data['annotations'].append(annotation)
        
        # Save updated cleaned data
        with open(cleaned_json_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        return True
    
    return False

def load_cleaned_data(cleaned_json_path):
    """Load the cleaned dataset"""
    with open(cleaned_json_path, 'r') as f:
        return json.load(f)

def save_cleaned_data(data, cleaned_json_path):
    """Save the cleaned dataset"""
    with open(cleaned_json_path, 'w') as f:
        json.dump(data, f, indent=2) 