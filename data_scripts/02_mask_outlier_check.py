import json
import os
import cv2
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_util
from collections import defaultdict
from tqdm import tqdm
import hashlib
from datetime import datetime
from scipy.spatial.distance import directed_hausdorff
import math
from mask_editor import detect_holes, fill_holes, remove_small_components
from mask_analysis_utils import (
    decode_rle, analyze_mask_components, validate_fish_size, analyze_convexity,
    calculate_mask_center, validate_motion_tracking, check_boundary_violations,
    analyze_boundary_smoothness, analyze_shape_complexity, detect_skinny_masks,
    calculate_mask_consistency, detect_boundary_skinny_parts, check_internal_holes,
    remove_small_holes, clean_mask, check_temporal_boundary_consistency,
    calculate_mask_hash
)

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

def validate_dataset_masks_with_tracking(json_file, image_root, output_dir=None, tracking_file=None, force_review=False,
                                       # Size thresholds
                                       min_area=100, max_area=50000,
                                       # Convexity threshold
                                       convexity_threshold=0.75,
                                       # Motion tracking threshold
                                       max_distance=100,
                                       # Boundary thresholds
                                       boundary_margin=10, boundary_ratio_threshold=0.3,
                                       # Smoothness threshold
                                       smoothness_threshold=0.01,
                                       # Complexity thresholds
                                       min_complexity_threshold=18, max_complexity_threshold=50,
                                       # Consistency thresholds
                                       iou_threshold=0.5, area_change_threshold=0.5, hausdorff_threshold=100,
                                       # Boundary skinny thresholds
                                       boundary_skinny_min_width=3, boundary_skinny_min_height=10,
                                       # Cleaning thresholds
                                       max_hole_area=5, max_component_area=5):
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
    overlap_issues = []
    component_stats = defaultdict(list)
    videos_to_review = []
    skipped_videos = []
    
    # Track unique masks with issues
    unique_masks_with_issues = set()
    
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
    
    print(f"Size thresholds: {min_area} - {max_area} pixels")
    print(f"Using thresholds:")
    print(f"  Convexity: {convexity_threshold}")
    print(f"  Motion tracking: {max_distance} pixels")
    print(f"  Boundary margin: {boundary_margin} pixels, ratio: {boundary_ratio_threshold}")
    print(f"  Smoothness: {smoothness_threshold}")
    print(f"  Complexity: {min_complexity_threshold} - {max_complexity_threshold}")
    print(f"  Consistency: IoU={iou_threshold}, area_change={area_change_threshold}, hausdorff={hausdorff_threshold}")
    print(f"  Boundary skinny: width={boundary_skinny_min_width}, height={boundary_skinny_min_height}")
    print(f"  Cleaning: hole_area={max_hole_area}, component_area={max_component_area}")
    
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
                
                # Clean the mask first (remove small holes and components)
                cleaned_mask = clean_mask(mask, max_hole_area=max_hole_area, max_component_area=max_component_area)
                
                # Analyze components on cleaned mask
                analysis = analyze_mask_components(cleaned_mask)
                
                # Store statistics
                component_stats['num_components'].append(analysis['num_components'])
                component_stats['largest_component_size'].append(analysis['largest_component_size'])
                
                # Create a unique identifier for this mask
                mask_id = f"{video_id}_{annotation['id']}_{frame_idx}"
                
                # Check for multi-part masks (on cleaned mask)
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
                        'total_mask_size': int(np.sum(cleaned_mask > 0)),
                        'mask_hash': calculate_mask_hash(cleaned_mask),
                        'issue_type': 'multi_part'
                    })
                    unique_masks_with_issues.add(mask_id)
                
                # Check for boundary skinny parts (on cleaned mask)
                boundary_skinny_analysis = detect_boundary_skinny_parts(cleaned_mask, min_width=boundary_skinny_min_width, min_height=boundary_skinny_min_height)
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
                        'mask_hash': calculate_mask_hash(cleaned_mask),
                        'issue_type': 'boundary_skinny'
                    })
                    unique_masks_with_issues.add(mask_id)
                
                # Check for size issues (on cleaned mask)
                size_analysis = validate_fish_size(cleaned_mask, min_area=min_area, max_area=max_area)
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
                        'mask_hash': calculate_mask_hash(cleaned_mask),
                        'issue_type': 'size'
                    })
                    unique_masks_with_issues.add(mask_id)
                
                # Check for convexity issues (on cleaned mask)
                convexity_analysis = analyze_convexity(cleaned_mask, convexity_threshold=convexity_threshold)
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
                        'mask_hash': calculate_mask_hash(cleaned_mask),
                        'issue_type': 'convexity'
                    })
                    unique_masks_with_issues.add(mask_id)
                
                # Check for boundary violations (on cleaned mask)
                boundary_analysis = check_boundary_violations(cleaned_mask, width, height, margin=boundary_margin, boundary_ratio_threshold=boundary_ratio_threshold)
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
                        'mask_hash': calculate_mask_hash(cleaned_mask),
                        'issue_type': 'boundary'
                    })
                    unique_masks_with_issues.add(mask_id)
                
                # Check for boundary smoothness issues (on cleaned mask)
                smoothness_analysis = analyze_boundary_smoothness(cleaned_mask, smoothness_threshold=smoothness_threshold)
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
                        'mask_hash': calculate_mask_hash(cleaned_mask),
                        'issue_type': 'smoothness'
                    })
                    unique_masks_with_issues.add(mask_id)
                
                # Check for internal holes (on cleaned mask)
                internal_holes_analysis = check_internal_holes(cleaned_mask)
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
                        'mask_hash': calculate_mask_hash(cleaned_mask),
                        'issue_type': 'internal_holes'
                    })
                    unique_masks_with_issues.add(mask_id)
                
                # Check for shape complexity issues (on cleaned mask)
                complexity_analysis = analyze_shape_complexity(cleaned_mask, min_complexity_threshold=min_complexity_threshold, max_complexity_threshold=max_complexity_threshold)
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
                        'mask_hash': calculate_mask_hash(cleaned_mask),
                        'issue_type': 'complexity'
                    })
                    unique_masks_with_issues.add(mask_id)
                
                # Check for consistency with previous frame (on cleaned masks)
                if frame_idx > 0 and annotation['segmentations'][frame_idx - 1] is not None:
                    prev_mask = decode_rle(annotation['segmentations'][frame_idx - 1], height, width)
                    prev_cleaned_mask = clean_mask(prev_mask, max_hole_area=max_hole_area, max_component_area=max_component_area)
                    consistency = calculate_mask_consistency(prev_cleaned_mask, cleaned_mask)
                    
                    # Flag if significant changes detected
                    if (consistency['iou'] < iou_threshold or
                        consistency['area_change_ratio'] > area_change_threshold or
                        consistency['hausdorff_distance'] > hausdorff_threshold):
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
                            'mask_hash': calculate_mask_hash(cleaned_mask),
                            'issue_type': 'inconsistent'
                        })
                        unique_masks_with_issues.add(mask_id)
                    
                    # Check for motion tracking issues (teleportation)
                    motion_analysis = validate_motion_tracking(prev_cleaned_mask, cleaned_mask, max_distance=max_distance)
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
                            'mask_hash': calculate_mask_hash(cleaned_mask),
                            'issue_type': 'motion'
                        })
                        unique_masks_with_issues.add(mask_id)
                    
                    # Check for temporal boundary consistency
                    # Get next frame if available
                    next_cleaned_mask = None
                    if frame_idx + 1 < len(annotation['segmentations']) and annotation['segmentations'][frame_idx + 1] is not None:
                        next_mask = decode_rle(annotation['segmentations'][frame_idx + 1], height, width)
                        next_cleaned_mask = clean_mask(next_mask, max_hole_area=max_hole_area, max_component_area=max_component_area)
                    
                    temporal_boundary_analysis = check_temporal_boundary_consistency(cleaned_mask, prev_cleaned_mask, next_cleaned_mask, width, height, margin=boundary_margin)
                    if temporal_boundary_analysis['has_temporal_inconsistency']:
                        boundary_issues.append({
                            'annotation_idx': ann_idx,
                            'annotation_id': annotation['id'],
                            'video_id': video_id,
                            'category_id': category_id,
                            'category_name': categories[category_id],
                            'frame_idx': frame_idx,
                            'inconsistency_type': temporal_boundary_analysis['inconsistency_type'],
                            'boundary_ratio': temporal_boundary_analysis['boundary_ratio'],
                            'boundary_overlap': temporal_boundary_analysis['boundary_overlap'],
                            'total_area': temporal_boundary_analysis['total_area'],
                            'mask_hash': calculate_mask_hash(cleaned_mask),
                            'issue_type': 'temporal_boundary'
                        })
                        unique_masks_with_issues.add(mask_id)
                    
            except Exception as e:
                print(f"Error processing annotation {ann_idx}, frame {frame_idx}: {e}")
    
    # Check for overlapping annotations within each frame
    print("Checking for overlapping annotations...")
    processed_frames = set()  # Track which frames we've already checked
    
    for ann_idx, annotation in enumerate(tqdm(data['annotations'], desc="Checking overlaps")):
        video_id = annotation['video_id']
        
        # Skip if video doesn't need review
        if video_id not in videos_to_review:
            continue
            
        video = videos[video_id]
        height, width = video['height'], video['width']
        
        # Check each frame for overlaps
        for frame_idx, segmentation in enumerate(annotation['segmentations']):
            if segmentation is None:
                continue
                
            # Create a unique frame identifier to avoid duplicate checks
            frame_key = f"{video_id}_{frame_idx}"
            if frame_key in processed_frames:
                continue
                
            processed_frames.add(frame_key)
            
            # Check for overlaps in this frame
            frame_overlaps = detect_overlapping_annotations(data, video_id, frame_idx, height, width)
            overlap_issues.extend(frame_overlaps)
    
    # Combine all issues
    all_issues = (multi_part_masks + boundary_skinny_masks + inconsistent_masks + size_issues + 
                  convexity_issues + motion_issues + boundary_issues + smoothness_issues + 
                  internal_holes_issues + complexity_issues + overlap_issues)
    
    # Print summary statistics
    print(f"\n=== ENHANCED MASK VALIDATION SUMMARY (AFTER CLEANING) ===")
    print(f"Total annotations processed: {total_annotations}")
    print(f"Unique masks with issues: {len(unique_masks_with_issues)}")
    print(f"Total issues found: {len(all_issues)}")
    print(f"\nIssues by type:")
    print(f"  Multi-part masks: {len(multi_part_masks)} issues")
    print(f"  Skinny masks: {len(boundary_skinny_masks)} issues")
    print(f"  Inconsistent masks: {len(inconsistent_masks)} issues")
    print(f"  Size issues: {len(size_issues)} issues")
    print(f"  Convexity issues: {len(convexity_issues)} issues")
    print(f"  Motion issues: {len(motion_issues)} issues")
    print(f"  Boundary issues: {len(boundary_issues)} issues")
    print(f"  Temporal boundary issues: {len([i for i in boundary_issues if i['issue_type'] == 'temporal_boundary'])} issues")
    print(f"  Smoothness issues: {len(smoothness_issues)} issues")
    print(f"  Internal holes: {len(internal_holes_issues)} issues")
    print(f"  Complexity issues: {len(complexity_issues)} issues")
    print(f"  Overlap issues: {len(overlap_issues)} issues")
    
    if total_annotations > 0:
        print(f"\nPercentages (of total annotations):")
        print(f"  Masks with any issue: {len(unique_masks_with_issues)/total_annotations*100:.2f}%")
        print(f"  Multi-part masks: {len(multi_part_masks)/total_annotations*100:.2f}%")
        print(f"  Skinny masks: {len(boundary_skinny_masks)/total_annotations*100:.2f}%")
        print(f"  Inconsistent masks: {len(inconsistent_masks)/total_annotations*100:.2f}%")
        print(f"  Size issues: {len(size_issues)/total_annotations*100:.2f}%")
        print(f"  Convexity issues: {len(convexity_issues)/total_annotations*100:.2f}%")
        print(f"  Motion issues: {len(motion_issues)/total_annotations*100:.2f}%")
        print(f"  Boundary issues: {len(boundary_issues)/total_annotations*100:.2f}%")
        print(f"  Temporal boundary issues: {len([i for i in boundary_issues if i['issue_type'] == 'temporal_boundary'])/total_annotations*100:.2f}%")
        print(f"  Smoothness issues: {len(smoothness_issues)/total_annotations*100:.2f}%")
        print(f"  Internal holes: {len(internal_holes_issues)/total_annotations*100:.2f}%")
        print(f"  Complexity issues: {len(complexity_issues)/total_annotations*100:.2f}%")
        print(f"  Overlap issues: {len(overlap_issues)/total_annotations*100:.2f}%")
    
    if component_stats['num_components']:
        print(f"\nComponent statistics (after cleaning):")
        print(f"  Average components per mask: {np.mean(component_stats['num_components']):.2f}")
        print(f"  Max components in a single mask: {max(component_stats['num_components'])}")
    
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
            'unique_masks_with_issues': len(unique_masks_with_issues),
            'total_issues_count': len(all_issues),
            'multi_part_masks_count': len(multi_part_masks),
            'skinny_masks_count': len(boundary_skinny_masks),
            'inconsistent_masks_count': len(inconsistent_masks),
            'size_issues_count': len(size_issues),
            'convexity_issues_count': len(convexity_issues),
            'motion_issues_count': len(motion_issues),
            'boundary_issues_count': len(boundary_issues),
            'temporal_boundary_issues_count': len([i for i in boundary_issues if i['issue_type'] == 'temporal_boundary']),
            'smoothness_issues_count': len(smoothness_issues),
            'internal_issues_count': len(internal_holes_issues),
            'complexity_issues_count': len(complexity_issues),
            'overlap_issues_count': len(overlap_issues),
            'masks_with_issues_percentage': len(unique_masks_with_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'multi_part_percentage': len(multi_part_masks)/total_annotations*100 if total_annotations > 0 else 0,
            'skinny_percentage': len(boundary_skinny_masks)/total_annotations*100 if total_annotations > 0 else 0,
            'inconsistent_percentage': len(inconsistent_masks)/total_annotations*100 if total_annotations > 0 else 0,
            'size_percentage': len(size_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'convexity_percentage': len(convexity_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'motion_percentage': len(motion_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'boundary_percentage': len(boundary_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'temporal_boundary_percentage': len([i for i in boundary_issues if i['issue_type'] == 'temporal_boundary'])/total_annotations*100 if total_annotations > 0 else 0,
            'smoothness_percentage': len(smoothness_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'internal_percentage': len(internal_holes_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'complexity_percentage': len(complexity_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'overlap_percentage': len(overlap_issues)/total_annotations*100 if total_annotations > 0 else 0,
            'component_stats': serializable_component_stats,
            'videos_reviewed': len(videos_to_review),
            'videos_skipped': len(skipped_videos),
            'thresholds_used': {
                'min_area': min_area,
                'max_area': max_area,
                'convexity_threshold': convexity_threshold,
                'max_distance': max_distance,
                'boundary_margin': boundary_margin,
                'boundary_ratio_threshold': boundary_ratio_threshold,
                'smoothness_threshold': smoothness_threshold,
                'min_complexity_threshold': min_complexity_threshold,
                'max_complexity_threshold': max_complexity_threshold,
                'iou_threshold': iou_threshold,
                'area_change_threshold': area_change_threshold,
                'hausdorff_threshold': hausdorff_threshold,
                'boundary_skinny_min_width': boundary_skinny_min_width,
                'boundary_skinny_min_height': boundary_skinny_min_height,
                'max_hole_area': max_hole_area,
                'max_component_area': max_component_area
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



def create_cleaned_dataset(json_file, cleaned_json_file, max_hole_area=5, max_component_area=5):
    """Create a cleaned version of the dataset with small holes and small components automatically removed"""
    print(f"Creating cleaned dataset: {cleaned_json_file}")
    print(f"Removing holes with area <= {max_hole_area} pixels")
    print(f"Removing components with area <= {max_component_area} pixels")
    
    # Load original dataset
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create a copy for cleaning
    cleaned_data = data.copy()
    
    # Track statistics
    total_masks = 0
    masks_with_small_holes = 0
    masks_with_small_components = 0
    total_small_holes_removed = 0
    total_small_components_removed = 0
    
    # Process each annotation
    for ann_idx, annotation in enumerate(tqdm(cleaned_data['annotations'], desc="Cleaning masks")):
        video_id = annotation['video_id']
        
        # Get video dimensions
        video = next(v for v in cleaned_data['videos'] if v['id'] == video_id)
        height, width = video['height'], video['width']
        
        # Process each frame's segmentation
        for frame_idx, segmentation in enumerate(annotation['segmentations']):
            if segmentation is None:
                continue
                
            total_masks += 1
            
            try:
                # Decode the mask
                mask = decode_rle(segmentation, height, width)
                
                # Check for small holes
                holes = detect_holes(mask)
                small_holes = [hole for hole in holes if hole['area'] <= max_hole_area]
                
                # Check for small components
                binary_mask = (mask > 0).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
                small_components = []
                if num_labels > 1:
                    for label in range(1, num_labels):
                        component_mask = (labels == label).astype(np.uint8)
                        size = np.sum(component_mask)
                        if size <= max_component_area:
                            small_components.append({'area': size, 'mask': component_mask})
                
                # Clean the mask if needed
                if small_holes or small_components:
                    cleaned_mask = clean_mask(mask, max_hole_area, max_component_area)
                    
                    # Update statistics
                    if small_holes:
                        masks_with_small_holes += 1
                        total_small_holes_removed += len(small_holes)
                    
                    if small_components:
                        masks_with_small_components += 1
                        total_small_components_removed += len(small_components)
                    
                    # Encode back to RLE
                    new_rle = mask_util.encode(np.asfortranarray(cleaned_mask))
                    new_rle['counts'] = new_rle['counts'].decode('utf-8')
                    
                    # Update the segmentation
                    cleaned_data['annotations'][ann_idx]['segmentations'][frame_idx] = new_rle
                    
            except Exception as e:
                print(f"Error processing annotation {ann_idx}, frame {frame_idx}: {e}")
    
    # Save cleaned dataset
    with open(cleaned_json_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"\n=== CLEANING SUMMARY ===")
    print(f"Total masks processed: {total_masks}")
    print(f"Masks with small holes removed: {masks_with_small_holes}")
    print(f"Total small holes removed: {total_small_holes_removed}")
    print(f"Masks with small components removed: {masks_with_small_components}")
    print(f"Total small components removed: {total_small_components_removed}")
    print(f"Cleaned dataset saved to: {cleaned_json_file}")
    
    return cleaned_data



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
                        'category_id': ann1['category_id'],
                        'category_name': 'unknown'
                    })
                    
            except Exception as e:
                print(f"Warning: Could not check overlap between annotations {ann1['id']} and {ann2['id']}: {e}")
                continue
    
    return overlap_issues


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Check for mask outliers and issues in fish dataset')
    parser.add_argument('--json_file', default="/data/fishway_ytvis/all_videos.json", 
                       help='Path to the JSON dataset file')
    parser.add_argument('--cleaned_json_file', 
                       help='Path to the cleaned JSON file (if not provided, will use output_dir/all_videos_cleaned.json)')
    parser.add_argument('--image_root', default="/data/fishway_ytvis/all_videos", 
                       help='Path to the image root directory')
    parser.add_argument('--output_dir', default="/data/fishway_ytvis/mask_validation_results", 
                       help='Path to the output directory')
    parser.add_argument('--force', action='store_true', 
                       help='Force recalculation even if data has not changed')
    parser.add_argument('--no-cache', action='store_true', 
                       help='Disable caching and always recalculate')
    parser.add_argument('--max_hole_area', type=int, default=15,
                       help='Maximum hole area to automatically remove (default: 15)')
    parser.add_argument('--max_component_area', type=int, default=20,
                       help='Maximum component area to automatically remove (default: 20)')
    
    # Size thresholds
    parser.add_argument('--min_area', type=int, default=2000,
                       help='Minimum area threshold for size validation (default: 2000)')
    parser.add_argument('--max_area', type=int, default=390000,
                       help='Maximum area threshold for size validation (default: 390000)')
    
    # Convexity threshold
    parser.add_argument('--convexity_threshold', type=float, default=0.6,
                       help='Convexity ratio threshold (default: 0.6)')
    
    # Motion tracking threshold
    parser.add_argument('--max_distance', type=int, default=150,
                       help='Maximum distance for motion tracking validation (default: 150)')
    
    # Boundary thresholds
    parser.add_argument('--boundary_margin', type=int, default=10,
                       help='Boundary margin for violation detection (default: 10)')
    parser.add_argument('--boundary_ratio_threshold', type=float, default=0.3,
                       help='Boundary ratio threshold for violation detection (default: 0.3)')
    
    # Smoothness threshold
    parser.add_argument('--smoothness_threshold', type=float, default=0.01,
                       help='Smoothness ratio threshold (default: 0.01)')
    
    # Complexity thresholds
    parser.add_argument('--min_complexity_threshold', type=float, default=5,
                       help='Minimum complexity threshold (default: 5)')
    parser.add_argument('--max_complexity_threshold', type=float, default=100,
                       help='Maximum complexity threshold (default: 100)')
    
    # Consistency thresholds
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                       help='IoU threshold for consistency validation (default: 0.3)')
    parser.add_argument('--area_change_threshold', type=float, default=0.8,
                       help='Area change ratio threshold for consistency validation (default: 0.8)')
    parser.add_argument('--hausdorff_threshold', type=float, default=250,
                       help='Hausdorff distance threshold for consistency validation (default: 250)')
    
    # Boundary skinny thresholds
    parser.add_argument('--boundary_skinny_min_width', type=int, default=3,
                       help='Minimum width for boundary skinny detection (default: 3)')
    parser.add_argument('--boundary_skinny_min_height', type=int, default=20,
                       help='Minimum height for boundary skinny detection (default: 20)')
    
    parser.add_argument('--create_cleaned', action='store_true',
                       help='Create a cleaned version of the dataset with small holes and components removed')
    
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
    
    tracking_file = os.path.join(output_dir, "mask_review_tracking.json")
    
    # Create cleaned dataset if requested
    if args.create_cleaned:
        print("=== CREATING CLEANED DATASET ===")
        create_cleaned_dataset(json_file, cleaned_json_file, args.max_hole_area, args.max_component_area)
        print(f"✓ Cleaned dataset created: {cleaned_json_file}")
        print(f"✓ Use this file for mask review: --cleaned_json_file {cleaned_json_file}")
    
    # Run validation with tracking
    print("\n=== MASK OUTLIER CHECK ===")
    mask_issues, stats, videos_to_review = validate_dataset_masks_with_tracking(
        json_file, image_root, output_dir, tracking_file, force_review=args.force,
        # Size thresholds
        min_area=args.min_area, max_area=args.max_area,
        # Convexity threshold
        convexity_threshold=args.convexity_threshold,
        # Motion tracking threshold
        max_distance=args.max_distance,
        # Boundary thresholds
        boundary_margin=args.boundary_margin, boundary_ratio_threshold=args.boundary_ratio_threshold,
        # Smoothness threshold
        smoothness_threshold=args.smoothness_threshold,
        # Complexity thresholds
        min_complexity_threshold=args.min_complexity_threshold, max_complexity_threshold=args.max_complexity_threshold,
        # Consistency thresholds
        iou_threshold=args.iou_threshold, area_change_threshold=args.area_change_threshold, hausdorff_threshold=args.hausdorff_threshold,
        # Boundary skinny thresholds
        boundary_skinny_min_width=args.boundary_skinny_min_width, boundary_skinny_min_height=args.boundary_skinny_min_height,
        # Cleaning thresholds
        max_hole_area=args.max_hole_area, max_component_area=args.max_component_area
    )
    
    # Print summary
    if mask_issues:
        print(f"\n=== SUMMARY ===")
        print(f"Found {len(mask_issues)} mask issues across {len(videos_to_review)} videos")
        print(f"Results saved to: {output_dir}")
        print(f"\nTo review these issues interactively, run:")
        print(f"python fish-dvis/data_scripts/02_review_masks.py --json_file {json_file} --image_root {image_root} --output_dir {output_dir}")
        if args.create_cleaned:
            print(f"\nOr use the cleaned dataset for review:")
            print(f"python fish-dvis/data_scripts/02_review_masks.py --json_file {json_file} --cleaned_json_file {cleaned_json_file} --image_root {image_root} --output_dir {output_dir}")
    else:
        print("✓ No mask issues found!") 