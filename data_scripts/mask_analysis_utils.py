import json
import os
import cv2
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_util
from collections import defaultdict
import hashlib
from datetime import datetime
from scipy.spatial.distance import directed_hausdorff
import math
from mask_editor import detect_holes, fill_holes, remove_small_components

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

def calculate_mask_hash(mask):
    """Calculate a hash of the mask for change detection"""
    return hashlib.md5(mask.tobytes()).hexdigest()

def validate_fish_size(mask, min_area=100, max_area=50000):
    """Validate fish mask size - not too small (noise) or too large (multiple fish)"""
    total_area = np.sum(mask > 0)
    
    is_too_small = total_area < min_area
    is_too_large = total_area > max_area
    
    return {
        'area': int(total_area),
        'is_too_small': is_too_small,
        'is_too_large': is_too_large,
        'min_threshold': min_area,
        'max_threshold': max_area,
        'has_size_issue': is_too_small or is_too_large
    }

def analyze_convexity(mask, convexity_threshold=0.75):
    """Analyze mask convexity - fish should generally be convex"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'convexity_ratio': 0.0,
            'is_highly_concave': False,
            'contour_area': 0,
            'hull_area': 0
        }
    
    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    
    # Calculate convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    # Convexity ratio (area of contour / area of convex hull)
    convexity_ratio = contour_area / hull_area if hull_area > 0 else 0
    
    # Flag if highly concave (ratio < convexity_threshold)
    is_highly_concave = convexity_ratio < convexity_threshold
    
    return {
        'convexity_ratio': float(convexity_ratio),
        'is_highly_concave': is_highly_concave,
        'contour_area': int(contour_area),
        'hull_area': int(hull_area)
    }

def calculate_mask_center(mask):
    """Calculate the center of mass of a mask"""
    binary_mask = (mask > 0).astype(np.uint8)
    moments = cv2.moments(binary_mask)
    
    if moments['m00'] == 0:
        return None
    
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return (cx, cy)

def validate_motion_tracking(mask1, mask2, max_distance=100):
    """Validate motion between consecutive frames - fish shouldn't teleport"""
    center1 = calculate_mask_center(mask1)
    center2 = calculate_mask_center(mask2)
    
    if center1 is None or center2 is None:
        return {
            'distance': float('inf'),
            'is_teleport': True,
            'center1': center1,
            'center2': center2
        }
    
    distance = math.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
    is_teleport = distance > max_distance
    
    return {
        'distance': float(distance),
        'is_teleport': is_teleport,
        'center1': center1,
        'center2': center2,
        'max_threshold': max_distance
    }

def check_boundary_violations(mask, frame_width, frame_height, margin=10, boundary_ratio_threshold=0.3):
    """Check if fish appears/disappears at frame edges without reason"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Check if mask touches any edge
    touches_left = np.any(binary_mask[:, :margin])
    touches_right = np.any(binary_mask[:, -margin:])
    touches_top = np.any(binary_mask[:margin, :])
    touches_bottom = np.any(binary_mask[-margin:, :])
    
    touches_boundary = touches_left or touches_right or touches_top or touches_bottom
    
    # Calculate how much of the mask is near boundaries
    boundary_region = np.zeros_like(binary_mask)
    boundary_region[:margin, :] = 1  # top
    boundary_region[-margin:, :] = 1  # bottom
    boundary_region[:, :margin] = 1  # left
    boundary_region[:, -margin:] = 1  # right
    
    boundary_overlap = np.sum(binary_mask & boundary_region)
    total_mask_area = np.sum(binary_mask)
    boundary_ratio = boundary_overlap / total_mask_area if total_mask_area > 0 else 0
    
    # Flag if significant portion is near boundary
    has_boundary_issue = boundary_ratio > boundary_ratio_threshold
    
    return {
        'touches_boundary': touches_boundary,
        'boundary_ratio': float(boundary_ratio),
        'has_boundary_issue': has_boundary_issue,
        'boundary_overlap': int(boundary_overlap),
        'total_area': int(total_mask_area)
    }

def analyze_boundary_smoothness(mask, smoothness_threshold=0.01):
    """Analyze mask boundary smoothness - should be relatively smooth, not pixelated"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'perimeter': 0,
            'area': 0,
            'smoothness_ratio': 0.0,
            'is_rough': False
        }
    
    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate perimeter and area
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    # Smoothness ratio (area / perimeter^2) - higher is smoother
    smoothness_ratio = area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Flag if boundary is too rough (low smoothness ratio)
    is_rough = smoothness_ratio < smoothness_threshold
    
    return {
        'perimeter': float(perimeter),
        'area': int(area),
        'smoothness_ratio': float(smoothness_ratio),
        'is_rough': is_rough
    }

def analyze_shape_complexity(mask, min_complexity_threshold=18, max_complexity_threshold=50):
    """Analyze shape complexity - shouldn't be too simple or too complex"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'complexity_score': 0.0,
            'is_too_simple': False,
            'is_too_complex': False
        }
    
    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate complexity using perimeter and area
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    # Complexity score (perimeter^2 / area) - higher is more complex
    complexity_score = (perimeter ** 2) / area if area > 0 else 0
    
    # Flag if too simple or too complex
    is_too_simple = complexity_score < min_complexity_threshold  # Very simple shapes
    is_too_complex = complexity_score > max_complexity_threshold  # Very complex shapes
    
    return {
        'complexity_score': float(complexity_score),
        'is_too_simple': is_too_simple,
        'is_too_complex': is_too_complex,
        'perimeter': float(perimeter),
        'area': int(area)
    }

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

def calculate_mask_consistency(mask1, mask2):
    """Calculate consistency metrics between two consecutive masks"""
    binary_mask1 = (mask1 > 0).astype(np.uint8)
    binary_mask2 = (mask2 > 0).astype(np.uint8)
    
    # Calculate areas as floats to avoid overflow
    area1 = float(np.sum(binary_mask1))
    area2 = float(np.sum(binary_mask2))
    
    # Calculate intersection and union
    intersection = float(np.sum(binary_mask1 & binary_mask2))
    union = float(np.sum(binary_mask1 | binary_mask2))
    
    # Calculate metrics
    iou = intersection / union if union > 0 else 0
    area_change_ratio = abs(area2 - area1) / max(area1, area2) if max(area1, area2) > 0 else 0
    
    # Calculate Hausdorff distance for shape similarity
    try:
        # Find contours for Hausdorff distance
        contours1, _ = cv2.findContours(binary_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(binary_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours1 and contours2:
            # Use the largest contour from each mask
            contour1 = max(contours1, key=cv2.contourArea)
            contour2 = max(contours2, key=cv2.contourArea)
            
            # Convert contours to point arrays
            points1 = contour1.reshape(-1, 2)
            points2 = contour2.reshape(-1, 2)
            
            # Calculate Hausdorff distance
            hausdorff_dist = max(
                directed_hausdorff(points1, points2)[0],
                directed_hausdorff(points2, points1)[0]
            )
        else:
            hausdorff_dist = float('inf')
    except:
        hausdorff_dist = float('inf')
    
    return {
        'iou': float(iou),
        'area_change_ratio': float(area_change_ratio),
        'hausdorff_distance': float(hausdorff_dist),
        'area1': int(area1),
        'area2': int(area2),
        'intersection': int(intersection),
        'union': int(union)
    }

def detect_boundary_skinny_parts(mask, min_width=3, min_height=10):
    """Detect skinny parts against image boundaries that are artifacts of boundary drawing"""
    binary_mask = (mask > 0).astype(np.uint8)
    h, w = binary_mask.shape
    
    # Check each boundary for skinny parts
    boundary_skinny_parts = []
    
    # Check left boundary (x=0)
    for y in range(h):
        width = 0
        while width < w and binary_mask[y, width] == 1:
            width += 1
        if 0 < width <= min_width:  # Narrow strip against left boundary
            # Check if it's tall enough
            height = 1
            # Check up
            for dy in range(1, min_height):
                if y - dy >= 0 and binary_mask[y - dy, 0] == 1:
                    height += 1
                else:
                    break
            # Check down
            for dy in range(1, min_height):
                if y + dy < h and binary_mask[y + dy, 0] == 1:
                    height += 1
                else:
                    break
            if height >= min_height:
                boundary_skinny_parts.append(('left', y, width, height))
    
    # Check right boundary (x=w-1)
    for y in range(h):
        width = 0
        while width < w and binary_mask[y, w-1-width] == 1:
            width += 1
        if 0 < width <= min_width:  # Narrow strip against right boundary
            # Check if it's tall enough
            height = 1
            # Check up
            for dy in range(1, min_height):
                if y - dy >= 0 and binary_mask[y - dy, w-1] == 1:
                    height += 1
                else:
                    break
            # Check down
            for dy in range(1, min_height):
                if y + dy < h and binary_mask[y + dy, w-1] == 1:
                    height += 1
                else:
                    break
            if height >= min_height:
                boundary_skinny_parts.append(('right', y, width, height))
    
    # Check top boundary (y=0)
    for x in range(w):
        height = 0
        while height < h and binary_mask[height, x] == 1:
            height += 1
        if 0 < height <= min_width:  # Narrow strip against top boundary
            # Check if it's wide enough
            width = 1
            # Check left
            for dx in range(1, min_height):
                if x - dx >= 0 and binary_mask[0, x - dx] == 1:
                    width += 1
                else:
                    break
            # Check right
            for dx in range(1, min_height):
                if x + dx < w and binary_mask[0, x + dx] == 1:
                    width += 1
                else:
                    break
            if width >= min_height:
                boundary_skinny_parts.append(('top', x, height, width))
    
    # Check bottom boundary (y=h-1)
    for x in range(w):
        height = 0
        while height < h and binary_mask[h-1-height, x] == 1:
            height += 1
        if 0 < height <= min_width:  # Narrow strip against bottom boundary
            # Check if it's wide enough
            width = 1
            # Check left
            for dx in range(1, min_height):
                if x - dx >= 0 and binary_mask[h-1, x - dx] == 1:
                    width += 1
                else:
                    break
            # Check right
            for dx in range(1, min_height):
                if x + dx < w and binary_mask[h-1, x + dx] == 1:
                    width += 1
                else:
                    break
            if width >= min_height:
                boundary_skinny_parts.append(('bottom', x, height, width))
    
    has_boundary_skinny_parts = len(boundary_skinny_parts) > 0
    
    return {
        'has_boundary_skinny_parts': has_boundary_skinny_parts,
        'boundary_skinny_parts': boundary_skinny_parts,
        'num_boundary_skinny_parts': len(boundary_skinny_parts)
    }

def check_internal_holes(mask):
    """Check if mask has any internal holes using the new detection method"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Use the new hole detection method
    holes = detect_holes(binary_mask)
    hole_area = sum(hole['area'] for hole in holes)
    
    # Check if there are any holes
    has_internal_holes = len(holes) > 0
    
    return {
        'has_internal_holes': has_internal_holes,
        'hole_area': int(hole_area),
        'num_components': len(holes)
    }

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

def clean_mask(mask, max_hole_area=5, max_component_area=5):
    """Remove both small holes and small disconnected components from a mask"""
    # First remove small holes
    mask_no_holes = remove_small_holes(mask, max_hole_area)
    
    # Then remove small components
    cleaned_mask = remove_small_components(mask_no_holes, max_component_area)
    
    return cleaned_mask

def check_temporal_boundary_consistency(mask, prev_mask, next_mask, frame_width, frame_height, margin=10):
    """Check if fish touches boundary only for one frame (temporal inconsistency)"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Check if current mask touches boundaries
    touches_top = np.any(binary_mask[:margin, :])
    touches_bottom = np.any(binary_mask[frame_height-margin:, :])
    touches_left = np.any(binary_mask[:, :margin])
    touches_right = np.any(binary_mask[:, frame_width-margin:])
    
    current_touches_boundary = touches_top or touches_bottom or touches_left or touches_right
    
    # Check previous frame if available
    prev_touches_boundary = False
    if prev_mask is not None:
        prev_binary_mask = (prev_mask > 0).astype(np.uint8)
        prev_touches_top = np.any(prev_binary_mask[:margin, :])
        prev_touches_bottom = np.any(prev_binary_mask[frame_height-margin:, :])
        prev_touches_left = np.any(prev_binary_mask[:, :margin])
        prev_touches_right = np.any(prev_binary_mask[:, frame_width-margin:])
        prev_touches_boundary = prev_touches_top or prev_touches_bottom or prev_touches_left or prev_touches_right
    
    # Check next frame if available
    next_touches_boundary = False
    if next_mask is not None:
        next_binary_mask = (next_mask > 0).astype(np.uint8)
        next_touches_top = np.any(next_binary_mask[:margin, :])
        next_touches_bottom = np.any(next_binary_mask[frame_height-margin:, :])
        next_touches_left = np.any(next_binary_mask[:, :margin])
        next_touches_right = np.any(next_binary_mask[:, frame_width-margin:])
        next_touches_boundary = next_touches_top or next_touches_bottom or next_touches_left or next_touches_right
    
    # Detect temporal inconsistency
    has_temporal_inconsistency = False
    inconsistency_type = None
    
    if current_touches_boundary:
        # Current frame touches boundary
        if prev_mask is not None and next_mask is not None:
            # Both previous and next frames available
            if not prev_touches_boundary and not next_touches_boundary:
                has_temporal_inconsistency = True
                inconsistency_type = "isolated_boundary_contact"
        elif prev_mask is not None:
            # Only previous frame available
            if not prev_touches_boundary:
                has_temporal_inconsistency = True
                inconsistency_type = "sudden_boundary_appearance"
        elif next_mask is not None:
            # Only next frame available
            if not next_touches_boundary:
                has_temporal_inconsistency = True
                inconsistency_type = "sudden_boundary_disappearance"
    
    # Calculate boundary ratio for context
    boundary_region = np.zeros_like(binary_mask)
    boundary_region[:margin, :] = 1  # top
    boundary_region[-margin:, :] = 1  # bottom
    boundary_region[:, :margin] = 1  # left
    boundary_region[:, -margin:] = 1  # right
    
    boundary_overlap = np.sum(binary_mask & boundary_region)
    total_mask_area = np.sum(binary_mask)
    boundary_ratio = boundary_overlap / total_mask_area if total_mask_area > 0 else 0
    
    return {
        'has_temporal_inconsistency': has_temporal_inconsistency,
        'inconsistency_type': inconsistency_type,
        'current_touches_boundary': current_touches_boundary,
        'prev_touches_boundary': prev_touches_boundary,
        'next_touches_boundary': next_touches_boundary,
        'boundary_ratio': float(boundary_ratio),
        'boundary_overlap': int(boundary_overlap),
        'total_area': int(total_mask_area)
    } 