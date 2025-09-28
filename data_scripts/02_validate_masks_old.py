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
import hashlib
from datetime import datetime
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import ConvexHull
from scipy.stats import zscore
import math

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

def analyze_convexity(mask):
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
    
    # Flag if highly concave (ratio < 0.7)
    is_highly_concave = convexity_ratio < 0.7
    
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

def check_boundary_violations(mask, frame_width, frame_height, margin=10):
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
    has_boundary_issue = boundary_ratio > 0.3
    
    return {
        'touches_boundary': touches_boundary,
        'boundary_ratio': float(boundary_ratio),
        'has_boundary_issue': has_boundary_issue,
        'boundary_overlap': int(boundary_overlap),
        'total_area': int(total_mask_area)
    }

def analyze_boundary_smoothness(mask):
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
    is_rough = smoothness_ratio < 0.01
    
    return {
        'perimeter': float(perimeter),
        'area': int(area),
        'smoothness_ratio': float(smoothness_ratio),
        'is_rough': is_rough
    }

def check_internal_consistency(mask):
    """Check for holes or disconnected regions within fish masks"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    # Check for holes using a more robust method
    # Create a padded version for flood fill
    h, w = binary_mask.shape
    padded_mask = np.zeros((h+2, w+2), dtype=np.uint8)
    padded_mask[1:h+1, 1:w+1] = binary_mask
    
    # Flood fill from outside
    filled = padded_mask.copy()
    cv2.floodFill(filled, None, (0, 0), 1)
    
    # Extract the original region
    filled_region = filled[1:h+1, 1:w+1]
    
    # Holes are areas that are 1 in original but 0 in filled
    holes = binary_mask & (~filled_region.astype(bool))
    hole_area = np.sum(holes)
    
    # Check if there are multiple components (disconnected regions)
    has_multiple_components = num_labels > 2  # background + mask components
    
    # Flag if significant holes or disconnected regions
    total_area = np.sum(binary_mask)
    hole_ratio = hole_area / total_area if total_area > 0 else 0
    
    # Make detection less sensitive - only flag if holes are significant
    has_internal_issues = (hole_ratio > 0.3 and hole_area > 50) or has_multiple_components  # Increased threshold and added minimum hole area
    
    return {
        'hole_area': int(hole_area),
        'hole_ratio': float(hole_ratio),
        'num_components': num_labels - 1,
        'has_multiple_components': has_multiple_components,
        'has_internal_issues': has_internal_issues
    }

def analyze_shape_complexity(mask):
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
    is_too_simple = complexity_score < 20  # Very simple shapes
    is_too_complex = complexity_score > 200  # Very complex shapes
    
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

def calculate_data_hash(json_file):
    """Calculate hash of the dataset file to detect changes"""
    if not os.path.exists(json_file):
        return None
    
    # Get file modification time and size as a simple hash
    stat = os.stat(json_file)
    hash_data = f"{json_file}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(hash_data.encode()).hexdigest()

def calculate_script_hash():
    """Calculate hash of the current script to detect changes"""
    script_path = __file__
    if not os.path.exists(script_path):
        return None
    
    # Get file modification time and size
    stat = os.stat(script_path)
    hash_data = f"{script_path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(hash_data.encode()).hexdigest()

def save_review_tracking(tracking_data, tracking_file):
    """Save review tracking data"""
    tracking_data['last_updated'] = datetime.now().isoformat()
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f, indent=2)

def check_video_needs_review(video_id, video_folder, tracking_data, review_criteria_version):
    """Check if a video needs to be reviewed based on tracking data"""
    if video_id not in tracking_data['reviewed_videos']:
        return True, "Video not previously reviewed"
    
    video_tracking = tracking_data['reviewed_videos'][video_id]
    
    # Check if review criteria have changed
    if video_tracking.get('review_criteria_version') != review_criteria_version:
        return True, f"Review criteria changed from {video_tracking.get('review_criteria_version')} to {review_criteria_version}"
    
    # Check if video folder has changed (indicating new data)
    if video_tracking.get('video_folder') != video_folder:
        return True, f"Video folder changed from {video_tracking.get('video_folder')} to {video_folder}"
    
    return False, "Video already reviewed with current criteria"

def can_use_cached_results(tracking_data, json_file, force_review=False):
    """Check if we can use cached validation results"""
    if force_review:
        return False, "Force review requested"
    
    # Check if we have cached results
    if 'validation_results' not in tracking_data:
        return False, "No cached validation results found"
    
    # Check if data file has changed
    current_data_hash = calculate_data_hash(json_file)
    cached_data_hash = tracking_data.get('data_hash')
    
    if current_data_hash != cached_data_hash:
        return False, f"Data file changed (hash: {cached_data_hash[:8]}... -> {current_data_hash[:8]}...)"
    
    # Check if script has changed
    current_script_hash = calculate_script_hash()
    cached_script_hash = tracking_data.get('script_hash')
    
    if current_script_hash != cached_script_hash:
        return False, f"Script changed (hash: {cached_script_hash[:8]}... -> {current_script_hash[:8]}...)"
    
    return True, "Using cached results"

def convert_bools_to_ints(obj):
    """Convert boolean values to integers for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_bools_to_ints(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_bools_to_ints(item) for item in obj]
    elif isinstance(obj, bool):
        return int(obj)
    elif isinstance(obj, (int, float, str)) or obj is None:
        return obj
    else:
        # For any other type, try to convert to string
        return str(obj)

def validate_dataset_masks_with_tracking(json_file, image_root, output_dir=None, tracking_file=None, force_review=False):
    """Validate all masks in the dataset with enhanced tracking functionality"""
    
    if tracking_file is None:
        tracking_file = os.path.join(output_dir, "mask_review_tracking.json") if output_dir else "mask_review_tracking.json"
    
    # Load tracking data
    tracking_data = load_review_tracking(tracking_file)
    review_criteria_version = tracking_data.get('review_criteria_version', '1.0')
    
    # Check if we can use cached results
    can_use_cache, cache_reason = can_use_cached_results(tracking_data, json_file, force_review)
    
    if can_use_cache:
        print(f"✓ {cache_reason}")
        print(f"Loading cached validation results...")
        
        # Load cached results
        cached_results = tracking_data['validation_results']
        mask_issues = cached_results.get('mask_issues', [])
        component_stats = cached_results.get('component_stats', {})
        videos_to_review = cached_results.get('videos_to_review', [])
        
        # Convert back to proper types if needed
        if mask_issues and isinstance(mask_issues[0].get('is_too_small', None), int):
            # Convert integer booleans back to actual booleans
            for issue in mask_issues:
                for key, value in issue.items():
                    if isinstance(value, int) and key.startswith('is_'):
                        issue[key] = bool(value)
        
        print(f"✓ Loaded {len(mask_issues)} cached mask issues")
        print(f"✓ Loaded cached component statistics")
        print(f"✓ Loaded {len(videos_to_review)} videos to review")
        
        return mask_issues, component_stats, videos_to_review
    
    print(f"⚠️  {cache_reason}")
    print(f"Recalculating validation results...")
    
    print(f"Loading dataset from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract video and annotation information
    videos = {v['id']: v for v in data['videos']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Statistics
    total_annotations = 0
    multi_part_masks = []
    boundary_skinny_masks = []
    inconsistent_masks = []
    size_issues = []
    convexity_issues = []
    motion_issues = []
    boundary_issues = []
    smoothness_issues = []
    internal_holes_issues = []
    complexity_issues = []
    component_stats = defaultdict(list)
    videos_to_review = []
    skipped_videos = []
    
    # Collect all mask areas for statistical analysis
    all_mask_areas = []
    
    print(f"Found {len(data['videos'])} videos")
    print(f"Found {len(data['annotations'])} annotations")
    
    # Check which videos need review
    for video in data['videos']:
        video_id = video['id']
        video_folder = video['file_names'][0].split('/')[0] if video['file_names'] else f"video_{video_id}"
        
        needs_review, reason = check_video_needs_review(video_id, video_folder, tracking_data, review_criteria_version)
        
        if force_review or needs_review:
            videos_to_review.append(video_id)
            print(f"Video {video_id} ({video_folder}): {reason}")
        else:
            skipped_videos.append(video_id)
            print(f"Video {video_id} ({video_folder}): Skipping - already reviewed")
    
    print(f"\nVideos to review: {len(videos_to_review)}")
    print(f"Videos skipped: {len(skipped_videos)}")
    
    # First pass: collect all mask areas for statistical analysis
    print("Collecting mask statistics for outlier detection...")
    for ann_idx, annotation in enumerate(tqdm(data['annotations'], desc="Collecting statistics")):
        video_id = annotation['video_id']
        if video_id not in videos_to_review:
            continue
            
        video = videos[video_id]
        height, width = video['height'], video['width']
        
        for frame_idx, segmentation in enumerate(annotation['segmentations']):
            if segmentation is None:
                continue
                
            try:
                mask = decode_rle(segmentation, height, width)
                area = np.sum(mask > 0)
                if area > 0:
                    all_mask_areas.append(area)
            except Exception as e:
                continue
    
    # Calculate statistical thresholds for size outliers
    if all_mask_areas:
        areas_array = np.array(all_mask_areas)
        mean_area = np.mean(areas_array)
        std_area = np.std(areas_array)
        size_threshold_low = mean_area - 2 * std_area
        size_threshold_high = mean_area + 2 * std_area
    else:
        size_threshold_low = 100
        size_threshold_high = 50000
    
    print(f"Size thresholds: {size_threshold_low:.0f} - {size_threshold_high:.0f} pixels")
    
    # Process each annotation for videos that need review
    for ann_idx, annotation in enumerate(tqdm(data['annotations'], desc="Analyzing masks")):
        video_id = annotation['video_id']
        
        # Skip if video doesn't need review
        if video_id not in videos_to_review:
            continue
            
        video = videos[video_id]
        category_id = annotation['category_id']
        
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
                
                # Check for multi-part masks
                if analysis['is_multi_part']:
                    multi_part_masks.append({
                        'annotation_idx': ann_idx,
                        'annotation_id': annotation['id'],
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'num_components': analysis['num_components'],
                        'component_sizes': [int(size) for size in analysis['component_sizes']],
                        'largest_component_size': int(analysis['largest_component_size']),
                        'total_mask_size': int(np.sum(mask > 0)),
                        'mask_hash': calculate_mask_hash(mask),
                        'issue_type': 'multi_part'
                    })
                
                # Check for boundary skinny parts (replaces old skinny detection)
                boundary_skinny_analysis = detect_boundary_skinny_parts(mask, min_width=3, min_height=10)
                if boundary_skinny_analysis['has_boundary_skinny_parts']:
                    boundary_skinny_masks.append({
                        'annotation_idx': ann_idx,
                        'annotation_id': annotation['id'],
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'num_boundary_skinny_parts': boundary_skinny_analysis['num_boundary_skinny_parts'],
                        'boundary_skinny_parts': boundary_skinny_analysis['boundary_skinny_parts'],
                        'mask_hash': calculate_mask_hash(mask),
                        'issue_type': 'boundary_skinny'
                    })
                
                # Check for size issues
                size_analysis = validate_fish_size(mask, min_area=int(size_threshold_low), max_area=int(size_threshold_high))
                if size_analysis['has_size_issue']:
                    size_issues.append({
                        'annotation_idx': ann_idx,
                        'annotation_id': annotation['id'],
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'area': size_analysis['area'],
                        'is_too_small': size_analysis['is_too_small'],
                        'is_too_large': size_analysis['is_too_large'],
                        'min_threshold': size_analysis['min_threshold'],
                        'max_threshold': size_analysis['max_threshold'],
                        'mask_hash': calculate_mask_hash(mask),
                        'issue_type': 'size'
                    })
                
                # Check for convexity issues
                convexity_analysis = analyze_convexity(mask)
                if convexity_analysis['is_highly_concave']:
                    convexity_issues.append({
                        'annotation_idx': ann_idx,
                        'annotation_id': annotation['id'],
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'convexity_ratio': convexity_analysis['convexity_ratio'],
                        'contour_area': convexity_analysis['contour_area'],
                        'hull_area': convexity_analysis['hull_area'],
                        'mask_hash': calculate_mask_hash(mask),
                        'issue_type': 'convexity'
                    })
                
                # Check for boundary violations
                boundary_analysis = check_boundary_violations(mask, width, height, margin=10)
                if boundary_analysis['has_boundary_issue']:
                    boundary_issues.append({
                        'annotation_idx': ann_idx,
                        'annotation_id': annotation['id'],
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'boundary_ratio': boundary_analysis['boundary_ratio'],
                        'touches_boundary': boundary_analysis['touches_boundary'],
                        'boundary_overlap': boundary_analysis['boundary_overlap'],
                        'total_area': boundary_analysis['total_area'],
                        'mask_hash': calculate_mask_hash(mask),
                        'issue_type': 'boundary'
                    })
                
                # Check for boundary smoothness issues
                smoothness_analysis = analyze_boundary_smoothness(mask)
                if smoothness_analysis['is_rough']:
                    smoothness_issues.append({
                        'annotation_idx': ann_idx,
                        'annotation_id': annotation['id'],
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'smoothness_ratio': smoothness_analysis['smoothness_ratio'],
                        'perimeter': smoothness_analysis['perimeter'],
                        'area': smoothness_analysis['area'],
                        'mask_hash': calculate_mask_hash(mask),
                        'issue_type': 'smoothness'
                    })
                
                # Check for internal holes (simplified version)
                internal_holes_analysis = check_internal_holes(mask)
                if internal_holes_analysis['has_internal_holes']:
                    internal_holes_issues.append({
                        'annotation_idx': ann_idx,
                        'annotation_id': annotation['id'],
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'hole_area': internal_holes_analysis['hole_area'],
                        'num_components': internal_holes_analysis['num_components'],
                        'mask_hash': calculate_mask_hash(mask),
                        'issue_type': 'internal_holes'
                    })
                
                # Check for shape complexity issues
                complexity_analysis = analyze_shape_complexity(mask)
                if complexity_analysis['is_too_simple'] or complexity_analysis['is_too_complex']:
                    complexity_issues.append({
                        'annotation_idx': ann_idx,
                        'annotation_id': annotation['id'],
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'complexity_score': complexity_analysis['complexity_score'],
                        'is_too_simple': complexity_analysis['is_too_simple'],
                        'is_too_complex': complexity_analysis['is_too_complex'],
                        'perimeter': complexity_analysis['perimeter'],
                        'area': complexity_analysis['area'],
                        'mask_hash': calculate_mask_hash(mask),
                        'issue_type': 'complexity'
                    })
                
                # Check for consistency with previous frame
                if frame_idx > 0 and annotation['segmentations'][frame_idx - 1] is not None:
                    prev_mask = decode_rle(annotation['segmentations'][frame_idx - 1], height, width)
                    consistency = calculate_mask_consistency(prev_mask, mask)
                    
                    # Flag if significant changes detected
                    if (consistency['iou'] < 0.5 or  # Relaxed from 0.7 to 0.5
                        consistency['area_change_ratio'] > 0.5 or  # Relaxed from 0.3 to 0.5
                        consistency['hausdorff_distance'] > 100):  # Relaxed from 50 to 100
                        inconsistent_masks.append({
                            'annotation_idx': ann_idx,
                            'annotation_id': annotation['id'],
                            'video_id': video_id,
                            'category_id': category_id,
                            'category_name': categories[category_id],
                            'frame_idx': frame_idx,
                            'prev_frame_idx': frame_idx - 1,
                            'iou': consistency['iou'],
                            'area_change_ratio': consistency['area_change_ratio'],
                            'hausdorff_distance': consistency['hausdorff_distance'],
                            'area1': consistency['area1'],
                            'area2': consistency['area2'],
                            'mask_hash': calculate_mask_hash(mask),
                            'issue_type': 'inconsistent'
                        })
                    
                    # Check for motion tracking issues (teleportation)
                    motion_analysis = validate_motion_tracking(prev_mask, mask, max_distance=100)
                    if motion_analysis['is_teleport']:
                        motion_issues.append({
                            'annotation_idx': ann_idx,
                            'annotation_id': annotation['id'],
                            'video_id': video_id,
                            'category_id': category_id,
                            'category_name': categories[category_id],
                            'frame_idx': frame_idx,
                            'prev_frame_idx': frame_idx - 1,
                            'distance': motion_analysis['distance'],
                            'center1': motion_analysis['center1'],
                            'center2': motion_analysis['center2'],
                            'max_threshold': motion_analysis['max_threshold'],
                            'mask_hash': calculate_mask_hash(mask),
                            'issue_type': 'motion'
                        })
                    
                # Check for boundary skinny parts (replaces old skinny detection)
                boundary_skinny_analysis = detect_boundary_skinny_parts(mask, min_width=3, min_height=10)
                if boundary_skinny_analysis['has_boundary_skinny_parts']:
                    boundary_skinny_masks.append({
                        'annotation_idx': ann_idx,
                        'annotation_id': annotation['id'],
                        'video_id': video_id,
                        'category_id': category_id,
                        'category_name': categories[category_id],
                        'frame_idx': frame_idx,
                        'num_boundary_skinny_parts': boundary_skinny_analysis['num_boundary_skinny_parts'],
                        'boundary_skinny_parts': boundary_skinny_analysis['boundary_skinny_parts'],
                        'mask_hash': calculate_mask_hash(mask),
                        'issue_type': 'boundary_skinny'
                    })
                
            except Exception as e:
                print(f"Error processing annotation {ann_idx}, frame {frame_idx}: {e}")
    
    # Combine all issues
    all_issues = (multi_part_masks + boundary_skinny_masks + inconsistent_masks + size_issues + 
                  convexity_issues + motion_issues + boundary_issues + smoothness_issues + 
                  internal_holes_issues + complexity_issues)
    
    # Print summary statistics
    print(f"\n=== ENHANCED MASK VALIDATION SUMMARY ===")
    print(f"Total annotations processed: {total_annotations}")
    print(f"Multi-part masks found: {len(multi_part_masks)}")
    print(f"Skinny masks found: {len(boundary_skinny_masks)}")
    print(f"Inconsistent masks found: {len(inconsistent_masks)}")
    print(f"Size issues found: {len(size_issues)}")
    print(f"Convexity issues found: {len(convexity_issues)}")
    print(f"Motion issues found: {len(motion_issues)}")
    print(f"Boundary issues found: {len(boundary_issues)}")
    print(f"Smoothness issues found: {len(smoothness_issues)}")
    print(f"Internal issues found: {len(internal_holes_issues)}")
    print(f"Complexity issues found: {len(complexity_issues)}")
    print(f"Total issues found: {len(all_issues)}")
    
    if total_annotations > 0:
        print(f"Percentage of multi-part masks: {len(multi_part_masks)/total_annotations*100:.2f}%")
        print(f"Percentage of skinny masks: {len(boundary_skinny_masks)/total_annotations*100:.2f}%")
        print(f"Percentage of inconsistent masks: {len(inconsistent_masks)/total_annotations*100:.2f}%")
        print(f"Percentage of size issues: {len(size_issues)/total_annotations*100:.2f}%")
        print(f"Percentage of convexity issues: {len(convexity_issues)/total_annotations*100:.2f}%")
        print(f"Percentage of motion issues: {len(motion_issues)/total_annotations*100:.2f}%")
        print(f"Percentage of boundary issues: {len(boundary_issues)/total_annotations*100:.2f}%")
        print(f"Percentage of smoothness issues: {len(smoothness_issues)/total_annotations*100:.2f}%")
        print(f"Percentage of internal issues: {len(internal_holes_issues)/total_annotations*100:.2f}%")
        print(f"Percentage of complexity issues: {len(complexity_issues)/total_annotations*100:.2f}%")
    
    if component_stats['num_components']:
        print(f"Average components per mask: {np.mean(component_stats['num_components']):.2f}")
        print(f"Max components in a single mask: {max(component_stats['num_components'])}")
    
    # Save detailed results to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all issues
        issues_file = os.path.join(output_dir, "mask_issues.json")
        # Convert boolean values to integers for JSON serialization
        serializable_issues = convert_bools_to_ints(all_issues)
        with open(issues_file, 'w') as f:
            json.dump(serializable_issues, f, indent=2)
        print(f"All mask issues saved to: {issues_file}")
        
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
            'skinny_masks_count': len(boundary_skinny_masks),
            'inconsistent_masks_count': len(inconsistent_masks),
            'size_issues_count': len(size_issues),
            'convexity_issues_count': len(convexity_issues),
            'motion_issues_count': len(motion_issues),
            'boundary_issues_count': len(boundary_issues),
            'smoothness_issues_count': len(smoothness_issues),
            'internal_issues_count': len(internal_holes_issues),
            'complexity_issues_count': len(complexity_issues),
            'total_issues_count': len(all_issues),
            'multi_part_percentage': len(multi_part_masks)/total_annotations*100 if total_annotations > 0 else 0,
            'skinny_percentage': len(boundary_skinny_masks)/total_annotations*100 if total_annotations > 0 else 0,
            'inconsistent_percentage': len(inconsistent_masks)/total_annotations*100 if total_annotations > 0 else 0,
            'size_percentage': len(size_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'convexity_percentage': len(convexity_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'motion_percentage': len(motion_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'boundary_percentage': len(boundary_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'smoothness_percentage': len(smoothness_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'internal_percentage': len(internal_holes_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'complexity_percentage': len(complexity_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'component_stats': serializable_component_stats,
            'videos_reviewed': len(videos_to_review),
            'videos_skipped': len(skipped_videos),
            'size_thresholds': {
                'low': float(size_threshold_low),
                'high': float(size_threshold_high)
            }
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to: {summary_file}")
    
    # Save validation results and hashes for caching
    # Convert all data to JSON-serializable format
    serializable_mask_issues = convert_bools_to_ints(all_issues)
    serializable_videos_to_review = [int(vid) for vid in videos_to_review]
    
    tracking_data['validation_results'] = {
        'mask_issues': serializable_mask_issues,
        'component_stats': serializable_component_stats,
        'videos_to_review': serializable_videos_to_review,
        'summary': convert_bools_to_ints(summary)
    }
    tracking_data['data_hash'] = calculate_data_hash(json_file)
    tracking_data['script_hash'] = calculate_script_hash()
    save_review_tracking(tracking_data, tracking_file)
    print(f"✓ Validation results cached for future runs")
    
    return all_issues, component_stats, videos_to_review

def interactive_mask_review_with_tracking(json_file, image_root, mask_issues, output_dir, tracking_file=None):
    """Interactive review of mask issues with enhanced tracking functionality"""
    
    if tracking_file is None:
        tracking_file = os.path.join(output_dir, "mask_review_tracking.json") if output_dir else "mask_review_tracking.json"
    
    # Load tracking data
    tracking_data = load_review_tracking(tracking_file)
    
    print(f"\n=== INTERACTIVE MASK REVIEW ===")
    print(f"Found {len(mask_issues)} mask issues to review")
    print(f"Press 'k' to keep, 'm' to merge to largest, 's' to skip, 'e' for manual editing, 'p' for progress, 'S' to save, 'q' to quit")
    
    # Check for existing review progress
    if 'review_progress' in tracking_data:
        last_reviewed = tracking_data['review_progress'].get('last_reviewed_index', -1)
        if last_reviewed >= 0:
            resume_choice = input(f"\nFound previous review progress. Last reviewed: issue {last_reviewed + 1}/{len(mask_issues)}. Resume from here? (y/n): ").lower().strip()
            if resume_choice == 'y':
                start_index = last_reviewed + 1
                print(f"Resuming from issue {start_index + 1}/{len(mask_issues)}")
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
        'keep_all': [],
        'merge_to_largest': [],
        'skipped': [],
        'requires_editing': [],
        'manual_edits': []
    }
    
    # Track which videos are being reviewed
    reviewed_videos = set()
    
    for i, mask_info in enumerate(mask_issues[start_index:], start=start_index):
        video_id = mask_info['video_id']
        video = videos[video_id]
        video_folder = video['file_names'][0].split('/')[0] if video['file_names'] else f"video_{video_id}"
        reviewed_videos.add(video_id)
        
        print(f"\n--- Reviewing issue {i+1}/{len(mask_issues)} ---")
        print(f"Video ID: {video_id}, Category: {mask_info['category_name']}")
        print(f"Frame: {mask_info['frame_idx']}, Issue type: {mask_info['issue_type']}")
        
        # Print detailed issue summary
        print_issue_summary(mask_info, video, data)
        
        # Get the original frame and mask
        annotation = data['annotations'][mask_info['annotation_idx']]
        segmentation = annotation['segmentations'][mask_info['frame_idx']]
        
        # Decode mask
        mask = decode_rle(segmentation, video['height'], video['width'])
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Create visualization based on issue type
        if mask_info['issue_type'] == 'multi_part':
            create_multi_part_visualization(mask_info, video, binary_mask, image_root, i)
        elif mask_info['issue_type'] == 'skinny':
            create_skinny_mask_visualization(mask_info, video, binary_mask, image_root, i)
        elif mask_info['issue_type'] == 'inconsistent':
            create_inconsistent_mask_visualization(mask_info, video, binary_mask, image_root, data, i)
        elif mask_info['issue_type'] == 'size':
            create_size_issue_visualization(mask_info, video, binary_mask, image_root, i)
        elif mask_info['issue_type'] == 'convexity':
            create_convexity_issue_visualization(mask_info, video, binary_mask, image_root, i)
        elif mask_info['issue_type'] == 'motion':
            create_motion_issue_visualization(mask_info, video, binary_mask, image_root, data, i)
        elif mask_info['issue_type'] == 'boundary':
            create_boundary_issue_visualization(mask_info, video, binary_mask, image_root, i)
        elif mask_info['issue_type'] == 'smoothness':
            create_smoothness_issue_visualization(mask_info, video, binary_mask, image_root, i)
        elif mask_info['issue_type'] == 'internal':
            create_internal_issue_visualization(mask_info, video, binary_mask, image_root, i)
        elif mask_info['issue_type'] == 'complexity':
            create_complexity_issue_visualization(mask_info, video, binary_mask, image_root, i)
        
        # Get user input
        while True:
            key = input(f"\nIssue {i+1}/{len(mask_issues)} - Action (k/m/s/e/p/S/q): ").strip()
            
            if key.lower() == 'k':  # Keep
                review_results['keep_all'].append(mask_info)
                print("✓ Keeping as is")
                break
            elif key.lower() == 'm':  # Merge to largest (only for multi-part)
                if mask_info['issue_type'] == 'multi_part':
                    review_results['merge_to_largest'].append(mask_info)
                    print("✓ Merging to largest component")
                else:
                    print("Merge option only available for multi-part masks")
                break
            elif key.lower() == 's':  # Skip
                review_results['skipped'].append(mask_info)
                print("✓ Skipped")
                break
            elif key.lower() == 'e':  # Requires editing
                review_results['requires_editing'].append(mask_info)
                print("✓ Marked for manual editing")
                break
            elif key.lower() == 'p':  # Show progress
                show_review_progress(i, len(mask_issues), review_results)
                continue
            elif key == 'S':  # Save progress (capital S)
                tracking_data['review_progress']['last_reviewed_index'] = i
                tracking_data['review_progress']['total_issues'] = len(mask_issues)
                tracking_data['review_progress']['reviewed_count'] = i + 1
                tracking_data['review_progress']['last_updated'] = datetime.now().isoformat()
                save_review_tracking(tracking_data, tracking_file)
                print(f"✓ Progress saved at issue {i + 1}/{len(mask_issues)}")
                continue
            elif key.lower() == 'q':  # Quit
                print("Quitting review...")
                break
            else:
                print("Invalid input. Use 'k' (keep), 'm' (merge), 's' (skip), 'e' (edit), 'p' (progress), 'S' (save), or 'q' (quit)")
        
        if key == 'q':
            # Save progress before quitting
            tracking_data['review_progress']['last_reviewed_index'] = i
            tracking_data['review_progress']['total_issues'] = len(mask_issues)
            tracking_data['review_progress']['reviewed_count'] = i + 1
            tracking_data['review_progress']['last_updated'] = datetime.now().isoformat()
            save_review_tracking(tracking_data, tracking_file)
            print(f"\nProgress saved. Reviewed {i + 1}/{len(mask_issues)} issues.")
            print("You can resume later by running the script again.")
            break
        elif key == 'p':  # Show progress
            show_review_progress(i, len(mask_issues), review_results)
            continue
        elif key == 's':  # Save progress
            tracking_data['review_progress']['last_reviewed_index'] = i
            tracking_data['review_progress']['total_issues'] = len(mask_issues)
            tracking_data['review_progress']['reviewed_count'] = i + 1
            tracking_data['review_progress']['last_updated'] = datetime.now().isoformat()
            save_review_tracking(tracking_data, tracking_file)
            print(f"✓ Progress saved at issue {i + 1}/{len(mask_issues)}")
            continue
    
    # Update tracking data
    for video_id in reviewed_videos:
        video = videos[video_id]
        video_folder = video['file_names'][0].split('/')[0] if video['file_names'] else f"video_{video_id}"
        
        tracking_data['reviewed_videos'][video_id] = {
            'video_folder': video_folder,
            'review_date': datetime.now().isoformat(),
            'review_criteria_version': tracking_data.get('review_criteria_version', '1.0'),
            'review_status': 'completed'
        }
    
    # Track edited frames
    for mask_info in review_results['merge_to_largest']:
        video_id = mask_info['video_id']
        frame_idx = mask_info['frame_idx']
        annotation_id = mask_info['annotation_id']
        
        if video_id not in tracking_data['edited_frames']:
            tracking_data['edited_frames'][video_id] = {}
        
        if frame_idx not in tracking_data['edited_frames'][video_id]:
            tracking_data['edited_frames'][video_id][frame_idx] = []
        
        tracking_data['edited_frames'][video_id][frame_idx].append({
            'annotation_id': annotation_id,
            'edit_type': 'merge_to_largest',
            'edit_date': datetime.now().isoformat(),
            'original_components': mask_info['num_components'],
            'component_sizes': mask_info['component_sizes']
        })
    
    # Track frames requiring manual editing
    for mask_info in review_results['requires_editing']:
        video_id = mask_info['video_id']
        frame_idx = mask_info['frame_idx']
        annotation_id = mask_info['annotation_id']
        
        if video_id not in tracking_data['edited_frames']:
            tracking_data['edited_frames'][video_id] = {}
        
        if frame_idx not in tracking_data['edited_frames'][video_id]:
            tracking_data['edited_frames'][video_id][frame_idx] = []
        
        tracking_data['edited_frames'][video_id][frame_idx].append({
            'annotation_id': annotation_id,
            'edit_type': 'requires_manual_editing',
            'edit_date': datetime.now().isoformat(),
            'issue_type': mask_info['issue_type'],
            'issue_details': {k: v for k, v in mask_info.items() if k not in ['annotation_idx', 'annotation_id', 'video_id', 'category_id', 'category_name', 'frame_idx', 'mask_hash', 'issue_type']}
        })
    
    # Save tracking data
    save_review_tracking(tracking_data, tracking_file)
    
    # Save review results
    review_file = os.path.join(output_dir, "mask_review_results.json")
    with open(review_file, 'w') as f:
        json.dump(review_results, f, indent=2)
    
    print(f"\n=== REVIEW COMPLETE ===")
    print(f"Kept as is: {len(review_results['keep_all'])}")
    print(f"Merged to largest: {len(review_results['merge_to_largest'])}")
    print(f"Skipped: {len(review_results['skipped'])}")
    print(f"Requires manual editing: {len(review_results['requires_editing'])}")
    print(f"Results saved to: {review_file}")
    print(f"Tracking data saved to: {tracking_file}")
    
    return review_results

def create_multi_part_visualization(mask_info, video, binary_mask, image_root, issue_idx):
    """Create visualization for multi-part mask issues"""
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Multi-part Mask: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
    
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
    
    # Connected components (different colors)
    colored_components = np.zeros((*labels.shape, 3), dtype=np.uint8)
    colors = plt.cm.Set3(np.linspace(0, 1, num_labels))
    for label in range(1, num_labels):
        colored_components[labels == label] = (colors[label][:3] * 255).astype(np.uint8)
    
    axes[1, 0].imshow(colored_components)
    axes[1, 0].set_title(f'Components ({mask_info["num_components"]} parts, colored)')
    axes[1, 0].axis('off')
    
    # Show what it would look like if all parts were connected (smoothed into largest)
    component_sizes = mask_info['component_sizes']
    largest_component_id = np.argmax(component_sizes) + 1
    
    # Create a smoothed version that connects all components to the largest
    smoothed_mask = create_smoothed_connected_mask(labels, largest_component_id, num_labels)
    axes[1, 1].imshow(smoothed_mask, cmap='gray')
    axes[1, 1].set_title('Smoothed Connected (all parts merged)')
    axes[1, 1].axis('off')
    
    # Show mask without smaller parts (largest component only)
    largest_mask = create_component_mask(labels, largest_component_id)
    axes[1, 2].imshow(largest_mask, cmap='gray')
    axes[1, 2].set_title(f'Largest Only (size: {component_sizes[largest_component_id-1]})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def create_skinny_mask_visualization(mask_info, video, binary_mask, image_root, issue_idx):
    """Create visualization for skinny mask issues"""
    # Detect skinny parts
    skinny_analysis = detect_skinny_masks(binary_mask, min_width=10)
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Skinny Mask: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
    
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
    
    # Original mask
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')
    
    # Skinny parts highlighted
    skinny_parts = skinny_analysis['skinny_mask']
    axes[1, 1].imshow(skinny_parts, cmap='hot')
    axes[1, 1].set_title(f'Skinny Parts (ratio: {skinny_analysis["skinny_ratio"]:.2f})')
    axes[1, 1].axis('off')
    
    # Mask without skinny parts
    kernel = np.ones((10, 10), np.uint8)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    cleaned_mask = dilated
    axes[1, 2].imshow(cleaned_mask, cmap='gray')
    axes[1, 2].set_title('Cleaned Mask (no skinny parts)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def create_inconsistent_mask_visualization(mask_info, video, binary_mask, image_root, data, issue_idx):
    """Create visualization for inconsistent mask issues"""
    # Get previous frame mask
    annotation = data['annotations'][mask_info['annotation_idx']]
    prev_segmentation = annotation['segmentations'][mask_info['prev_frame_idx']]
    prev_mask = decode_rle(prev_segmentation, video['height'], video['width'])
    prev_binary_mask = (prev_mask > 0).astype(np.uint8)
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Inconsistent Mask: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frames {mask_info["prev_frame_idx"]}-{mask_info["frame_idx"]})', fontsize=14)
    
    # Get frame indices
    frame_idx = mask_info['frame_idx']
    prev_frame_idx = mask_info['prev_frame_idx']
    next_frame_idx = min(len(video['file_names']) - 1, frame_idx + 1)
    
    # Previous frame
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
    
    # Current frame
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
    
    # Next frame
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
    
    # Previous mask
    axes[1, 0].imshow(prev_binary_mask, cmap='gray')
    axes[1, 0].set_title(f'Previous Mask (area: {mask_info["area1"]})')
    axes[1, 0].axis('off')
    
    # Current mask
    axes[1, 1].imshow(binary_mask, cmap='gray')
    axes[1, 1].set_title(f'Current Mask (area: {mask_info["area2"]})')
    axes[1, 1].axis('off')
    
    # Difference
    difference = binary_mask.astype(int) - prev_binary_mask.astype(int)
    axes[1, 2].imshow(difference, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 2].set_title('Difference (red=added, blue=removed)')
    axes[1, 2].axis('off')
    
    # Add metrics
    info_text = f"IoU: {mask_info['iou']:.3f}\n"
    info_text += f"Area change: {mask_info['area_change_ratio']:.3f}\n"
    info_text += f"Hausdorff dist: {mask_info['hausdorff_distance']:.1f}"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, transform=fig.transFigure, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def create_size_issue_visualization(mask_info, video, binary_mask, image_root, issue_idx):
    """Create visualization for size issue masks"""
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Size Issue: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
    
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
    
    # Original mask
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title(f'Original Mask (area: {mask_info["area"]})')
    axes[1, 0].axis('off')
    
    # Size comparison
    axes[1, 1].bar(['Too Small', 'Normal', 'Too Large'], 
                   [mask_info["is_too_small"], not (mask_info["is_too_small"] or mask_info["is_too_large"]), mask_info["is_too_large"]],
                   color=['red', 'green', 'red'])
    axes[1, 1].set_title('Size Classification')
    axes[1, 1].set_ylim(0, 1)
    
    # Threshold information
    info_text = f"Area: {mask_info['area']} pixels\n"
    info_text += f"Min threshold: {mask_info['min_threshold']}\n"
    info_text += f"Max threshold: {mask_info['max_threshold']}\n"
    info_text += f"Too small: {mask_info['is_too_small']}\n"
    info_text += f"Too large: {mask_info['is_too_large']}"
    
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    axes[1, 2].set_title('Size Analysis')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def create_convexity_issue_visualization(mask_info, video, binary_mask, image_root, issue_idx):
    """Create visualization for convexity issue masks"""
    # Find contours and convex hull
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    hull = cv2.convexHull(contour) if contour is not None else None
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Convexity Issue: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
    
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
    
    # Original mask with contour
    axes[1, 0].imshow(binary_mask, cmap='gray')
    if contour is not None:
        contour_points = contour.reshape(-1, 2)
        axes[1, 0].plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
    axes[1, 0].set_title(f'Original Mask (convexity: {mask_info["convexity_ratio"]:.3f})')
    axes[1, 0].axis('off')
    
    # Convex hull overlay
    axes[1, 1].imshow(binary_mask, cmap='gray')
    if hull is not None:
        hull_points = hull.reshape(-1, 2)
        axes[1, 1].plot(hull_points[:, 0], hull_points[:, 1], 'g-', linewidth=2, label='Convex Hull')
    axes[1, 1].set_title('Convex Hull Overlay')
    axes[1, 1].axis('off')
    
    # Convexity analysis
    info_text = f"Convexity ratio: {mask_info['convexity_ratio']:.3f}\n"
    info_text += f"Contour area: {mask_info['contour_area']}\n"
    info_text += f"Hull area: {mask_info['hull_area']}\n"
    info_text += f"Highly concave: {mask_info['convexity_ratio'] < 0.7}"
    
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    axes[1, 2].set_title('Convexity Analysis')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def create_motion_issue_visualization(mask_info, video, binary_mask, image_root, data, issue_idx):
    """Create visualization for motion issue masks"""
    # Get previous frame mask
    annotation = data['annotations'][mask_info['annotation_idx']]
    prev_segmentation = annotation['segmentations'][mask_info['prev_frame_idx']]
    prev_mask = decode_rle(prev_segmentation, video['height'], video['width'])
    prev_binary_mask = (prev_mask > 0).astype(np.uint8)
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Motion Issue: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frames {mask_info["prev_frame_idx"]}-{mask_info["frame_idx"]})', fontsize=14)
    
    # Get frame indices
    frame_idx = mask_info['frame_idx']
    prev_frame_idx = mask_info['prev_frame_idx']
    next_frame_idx = min(len(video['file_names']) - 1, frame_idx + 1)
    
    # Previous frame
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
    
    # Current frame
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
    
    # Next frame
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
    
    # Previous mask
    axes[1, 0].imshow(prev_binary_mask, cmap='gray')
    if mask_info['center1']:
        axes[1, 0].plot(mask_info['center1'][0], mask_info['center1'][1], 'ro', markersize=10)
    axes[1, 0].set_title('Previous Mask')
    axes[1, 0].axis('off')
    
    # Current mask
    axes[1, 1].imshow(binary_mask, cmap='gray')
    if mask_info['center2']:
        axes[1, 1].plot(mask_info['center2'][0], mask_info['center2'][1], 'ro', markersize=10)
    axes[1, 1].set_title('Current Mask')
    axes[1, 1].axis('off')
    
    # Motion visualization
    if mask_info['center1'] and mask_info['center2']:
        axes[1, 2].plot([mask_info['center1'][0], mask_info['center2'][0]], 
                        [mask_info['center1'][1], mask_info['center2'][1]], 'r-', linewidth=3)
        axes[1, 2].plot(mask_info['center1'][0], mask_info['center1'][1], 'bo', markersize=10, label='Start')
        axes[1, 2].plot(mask_info['center2'][0], mask_info['center2'][1], 'ro', markersize=10, label='End')
        axes[1, 2].set_xlim(0, video['width'])
        axes[1, 2].set_ylim(video['height'], 0)  # Invert y-axis for image coordinates
        axes[1, 2].set_title('Motion Path')
        axes[1, 2].legend()
    else:
        axes[1, 2].text(0.5, 0.5, 'No center data', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Motion Path')
    axes[1, 2].axis('off')
    
    # Add motion analysis info
    info_text = f"Distance: {mask_info['distance']:.1f} pixels\n"
    info_text += f"Max threshold: {mask_info['max_threshold']}\n"
    info_text += f"Teleportation: {mask_info['distance'] > mask_info['max_threshold']}"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, transform=fig.transFigure, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def create_boundary_issue_visualization(mask_info, video, binary_mask, image_root, issue_idx):
    """Create visualization for boundary issue masks"""
    # Create boundary region mask
    h, w = binary_mask.shape
    margin = 10
    boundary_region = np.zeros_like(binary_mask)
    boundary_region[:margin, :] = 1  # top
    boundary_region[-margin:, :] = 1  # bottom
    boundary_region[:, :margin] = 1  # left
    boundary_region[:, -margin:] = 1  # right
    
    boundary_overlap = binary_mask & boundary_region
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Boundary Issue: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
    
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
    
    # Original mask
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')
    
    # Boundary region
    axes[1, 1].imshow(boundary_region, cmap='hot')
    axes[1, 1].set_title('Boundary Region')
    axes[1, 1].axis('off')
    
    # Boundary overlap
    axes[1, 2].imshow(boundary_overlap, cmap='hot')
    axes[1, 2].set_title(f'Boundary Overlap (ratio: {mask_info["boundary_ratio"]:.3f})')
    axes[1, 2].axis('off')
    
    # Add boundary analysis info
    info_text = f"Boundary ratio: {mask_info['boundary_ratio']:.3f}\n"
    info_text += f"Touches boundary: {mask_info['touches_boundary']}\n"
    info_text += f"Boundary overlap: {mask_info['boundary_overlap']}\n"
    info_text += f"Total area: {mask_info['total_area']}"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, transform=fig.transFigure, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def create_smoothness_issue_visualization(mask_info, video, binary_mask, image_root, issue_idx):
    """Create visualization for smoothness issue masks"""
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Smoothness Issue: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
    
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
    
    # Original mask with contour
    axes[1, 0].imshow(binary_mask, cmap='gray')
    if contour is not None:
        contour_points = contour.reshape(-1, 2)
        axes[1, 0].plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
    axes[1, 0].set_title(f'Original Mask (smoothness: {mask_info["smoothness_ratio"]:.6f})')
    axes[1, 0].axis('off')
    
    # Perimeter visualization
    axes[1, 1].imshow(binary_mask, cmap='gray')
    if contour is not None:
        # Highlight perimeter points
        for point in contour:
            axes[1, 1].plot(point[0][0], point[0][1], 'r.', markersize=1)
    axes[1, 1].set_title(f'Perimeter Points (length: {mask_info["perimeter"]:.1f})')
    axes[1, 1].axis('off')
    
    # Smoothness analysis
    info_text = f"Smoothness ratio: {mask_info['smoothness_ratio']:.6f}\n"
    info_text += f"Perimeter: {mask_info['perimeter']:.1f}\n"
    info_text += f"Area: {mask_info['area']}\n"
    info_text += f"Is rough: {mask_info['smoothness_ratio'] < 0.01}"
    
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    axes[1, 2].set_title('Smoothness Analysis')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def create_internal_issue_visualization(mask_info, video, binary_mask, image_root, issue_idx):
    """Create visualization for internal consistency issue masks"""
    # Find holes using flood fill
    h, w = binary_mask.shape
    filled = binary_mask.copy()
    cv2.floodFill(filled, None, (0, 0), 1)
    holes = binary_mask - filled
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Internal Issue: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
    
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
    
    # Original mask
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')
    
    # Holes highlighted
    axes[1, 1].imshow(holes, cmap='hot')
    axes[1, 1].set_title(f'Holes (area: {mask_info["hole_area"]})')
    axes[1, 1].axis('off')
    
    # Filled mask
    axes[1, 2].imshow(filled, cmap='gray')
    axes[1, 2].set_title('Filled Mask')
    axes[1, 2].axis('off')
    
    # Add internal analysis info
    info_text = f"Hole ratio: {mask_info['hole_ratio']:.3f}\n"
    info_text += f"Hole area: {mask_info['hole_area']}\n"
    info_text += f"Components: {mask_info['num_components']}\n"
    info_text += f"Multiple components: {mask_info['has_multiple_components']}"
    
    fig.text(0.02, 0.02, info_text, fontsize=10, transform=fig.transFigure, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def create_complexity_issue_visualization(mask_info, video, binary_mask, image_root, issue_idx):
    """Create visualization for complexity issue masks"""
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    
    # Create visualization with 6 plots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Complexity Issue: {mask_info["category_name"]} (Video {mask_info["video_id"]}, Frame {mask_info["frame_idx"]})', fontsize=14)
    
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
    
    # Original mask with contour
    axes[1, 0].imshow(binary_mask, cmap='gray')
    if contour is not None:
        contour_points = contour.reshape(-1, 2)
        axes[1, 0].plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
    axes[1, 0].set_title(f'Original Mask (complexity: {mask_info["complexity_score"]:.1f})')
    axes[1, 0].axis('off')
    
    # Complexity classification
    complexity_status = "Too Simple" if mask_info['is_too_simple'] else "Too Complex" if mask_info['is_too_complex'] else "Normal"
    colors = ['red' if mask_info['is_too_simple'] or mask_info['is_too_complex'] else 'green']
    
    axes[1, 1].bar([complexity_status], [1], color=colors)
    axes[1, 1].set_title('Complexity Classification')
    axes[1, 1].set_ylim(0, 1)
    
    # Complexity analysis
    info_text = f"Complexity score: {mask_info['complexity_score']:.1f}\n"
    info_text += f"Perimeter: {mask_info['perimeter']:.1f}\n"
    info_text += f"Area: {mask_info['area']}\n"
    info_text += f"Too simple: {mask_info['is_too_simple']}\n"
    info_text += f"Too complex: {mask_info['is_too_complex']}"
    
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=10, transform=axes[1, 2].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    axes[1, 2].set_title('Complexity Analysis')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

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
    
    print(f"   • Mask hash: {mask_info['mask_hash'][:8]}...")
    print()

def show_review_progress(current_index, total_issues, review_results):
    """Show current review progress and statistics"""
    print(f"\n📈 REVIEW PROGRESS:")
    print(f"   • Current position: {current_index + 1}/{total_issues}")
    print(f"   • Progress: {(current_index + 1)/total_issues*100:.1f}%")
    print(f"   • Remaining: {total_issues - current_index - 1} issues")
    
    print(f"\n📊 DECISIONS SO FAR:")
    print(f"   • Keep as is: {len(review_results['keep_all'])}")
    print(f"   • Merge to largest: {len(review_results['merge_to_largest'])}")
    print(f"   • Skip: {len(review_results['skipped'])}")
    print(f"   • Requires editing: {len(review_results['requires_editing'])}")
    
    if review_results['keep_all'] or review_results['merge_to_largest'] or review_results['skipped'] or review_results['requires_editing']:
        total_decided = len(review_results['keep_all']) + len(review_results['merge_to_largest']) + len(review_results['skipped']) + len(review_results['requires_editing'])
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
    """Check if mask has any internal holes (simplified version)"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    # Check for holes using flood fill
    h, w = binary_mask.shape
    padded_mask = np.zeros((h+2, w+2), dtype=np.uint8)
    padded_mask[1:h+1, 1:w+1] = binary_mask
    
    # Flood fill from outside
    filled = padded_mask.copy()
    cv2.floodFill(filled, None, (0, 0), 1)
    
    # Extract the original region
    filled_region = filled[1:h+1, 1:w+1]
    
    # Holes are areas that are 1 in original but 0 in filled
    holes = binary_mask & (~filled_region.astype(bool))
    hole_area = np.sum(holes)
    
    # Simply check if there are any holes
    has_internal_holes = hole_area > 0
    
    return {
        'has_internal_holes': has_internal_holes,
        'hole_area': int(hole_area),
        'num_components': num_labels - 1
    }

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate fish mask dataset')
    parser.add_argument('--json_file', default="/data/fishway_ytvis/all_videos.json", 
                       help='Path to the JSON dataset file')
    parser.add_argument('--image_root', default="/data/fishway_ytvis/all_videos", 
                       help='Path to the image root directory')
    parser.add_argument('--output_dir', default="/data/fishway_ytvis/mask_validation_results", 
                       help='Path to the output directory')
    parser.add_argument('--force', action='store_true', 
                       help='Force recalculation even if data has not changed')
    parser.add_argument('--no-cache', action='store_true', 
                       help='Disable caching and always recalculate')
    
    args = parser.parse_args()
    
    # Configuration
    json_file = args.json_file
    image_root = args.image_root
    output_dir = args.output_dir
    tracking_file = os.path.join(output_dir, "mask_review_tracking.json")
    
    # Run validation with tracking
    mask_issues, stats, videos_to_review = validate_dataset_masks_with_tracking(
        json_file, image_root, output_dir, tracking_file, force_review=args.force
    )
    
    # Visualize first few issues if any found
    if mask_issues:
        print(f"\nGenerating visualizations for first 5 mask issues...")
        for i, mask_info in enumerate(mask_issues[:5]):
            viz_path = os.path.join(output_dir, f"mask_issue_{i+1}.png")
            visualize_multi_part_mask(json_file, image_root, mask_info, viz_path)
        
        print(f"\n=== INTERACTIVE REVIEW OPTION ===")
        print(f"Found {len(mask_issues)} mask issues")
        print(f"To review these masks interactively, run:")
        print(f"python interactive_mask_review02.py --json_file {json_file} --image_root {image_root} --output_dir {output_dir} --mask_issues_file {os.path.join(output_dir, 'mask_issues.json')}")
    else:
        print("No mask issues found to review.") 