#!/usr/bin/env python3
"""
Script to analyze size distributions of mask components and internal holes,
plus all issue metrics for threshold determination
"""

import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pycocotools import mask as mask_util
from collections import defaultdict
from tqdm import tqdm
import argparse
from mask_editor import detect_holes, clean_mask
import math
from scipy.spatial.distance import directed_hausdorff
from mask_analysis_utils import (
    decode_rle, analyze_mask_components, validate_fish_size, analyze_convexity,
    calculate_mask_center, validate_motion_tracking, check_boundary_violations,
    analyze_boundary_smoothness, analyze_shape_complexity, detect_skinny_masks,
    calculate_mask_consistency, detect_boundary_skinny_parts, check_internal_holes,
    remove_small_holes, clean_mask, check_temporal_boundary_consistency
)

def analyze_mask_holes(mask):
    """Analyze mask holes and return sizes"""
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Detect holes
    holes = detect_holes(binary_mask)
    
    # Extract hole sizes
    hole_sizes = [hole['area'] for hole in holes]
    
    return hole_sizes

def analyze_dataset_distributions(json_file, output_dir):
    """Analyze component and hole size distributions for the entire dataset"""
    print(f"Loading dataset from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract video information
    videos = {v['id']: v for v in data['videos']}
    
    # Collect all metrics
    all_component_sizes = []
    all_hole_sizes = []
    all_convexity_ratios = []
    all_smoothness_ratios = []
    all_complexity_scores = []
    all_mask_areas = []
    all_boundary_ratios = []
    all_skinny_part_areas = []
    all_motion_distances = []
    all_iou_scores = []
    all_area_change_ratios = []
    all_hausdorff_distances = []
    all_internal_hole_areas = []
    all_multi_part_counts = []
    
    print(f"Found {len(data['videos'])} videos")
    print(f"Found {len(data['annotations'])} annotations")
    
    # Process each annotation
    for ann_idx, annotation in enumerate(tqdm(data['annotations'], desc="Analyzing masks")):
        video_id = annotation['video_id']
        video = videos[video_id]
        height, width = video['height'], video['width']
        
        # Process each frame's segmentation
        for frame_idx, segmentation in enumerate(annotation['segmentations']):
            if segmentation is None:
                continue
                
            try:
                # Decode the mask
                mask = decode_rle(segmentation, height, width)
                
                # Use the original mask without cleaning
                cleaned_mask = mask
                
                # Analyze components
                component_analysis = analyze_mask_components(cleaned_mask)
                all_component_sizes.extend(component_analysis['component_sizes'])
                
                # Analyze holes
                hole_sizes = analyze_mask_holes(cleaned_mask)
                all_hole_sizes.extend(hole_sizes)
                
                # Analyze mask area
                mask_area = np.sum(cleaned_mask > 0)
                all_mask_areas.append(mask_area)
                
                # Analyze convexity
                convexity_analysis = analyze_convexity(cleaned_mask)
                all_convexity_ratios.append(convexity_analysis['convexity_ratio'])
                
                # Analyze smoothness
                smoothness_analysis = analyze_boundary_smoothness(cleaned_mask)
                all_smoothness_ratios.append(smoothness_analysis['smoothness_ratio'])
                
                # Analyze complexity
                complexity_analysis = analyze_shape_complexity(cleaned_mask)
                all_complexity_scores.append(complexity_analysis['complexity_score'])
                
                # Analyze temporal boundary consistency
                # Get previous and next frames for temporal analysis
                prev_mask = None
                next_mask = None
                
                if frame_idx > 0 and annotation['segmentations'][frame_idx - 1] is not None:
                    prev_mask = decode_rle(annotation['segmentations'][frame_idx - 1], height, width)
                
                if frame_idx + 1 < len(annotation['segmentations']) and annotation['segmentations'][frame_idx + 1] is not None:
                    next_mask = decode_rle(annotation['segmentations'][frame_idx + 1], height, width)
                
                temporal_boundary_analysis = check_temporal_boundary_consistency(cleaned_mask, prev_mask, next_mask, width, height, margin=10)
                all_boundary_ratios.append(temporal_boundary_analysis['boundary_ratio'])
                
                # Analyze skinny parts
                skinny_analysis = detect_boundary_skinny_parts(cleaned_mask, min_width=3, min_height=10)
                if skinny_analysis['has_boundary_skinny_parts']:
                    for part in skinny_analysis['boundary_skinny_parts']:
                        # part is a tuple: (side, y, width, height)
                        # Calculate area as width * height
                        area = part[2] * part[3]  # width * height
                        all_skinny_part_areas.append(area)
                
                # Analyze internal holes
                internal_holes_analysis = check_internal_holes(cleaned_mask)
                if internal_holes_analysis['has_internal_holes']:
                    all_internal_hole_areas.append(internal_holes_analysis['hole_area'])
                
                # Analyze multi-part masks
                if component_analysis['is_multi_part']:
                    all_multi_part_counts.append(component_analysis['num_components'])
                
                # Analyze temporal consistency with previous frame
                if frame_idx > 0 and annotation['segmentations'][frame_idx - 1] is not None:
                    prev_mask = decode_rle(annotation['segmentations'][frame_idx - 1], height, width)
                    
                    # Motion tracking
                    motion_analysis = validate_motion_tracking(prev_mask, cleaned_mask, max_distance=100)
                    all_motion_distances.append(motion_analysis['distance'])
                    
                    # Consistency metrics
                    consistency_analysis = calculate_mask_consistency(prev_mask, cleaned_mask)
                    all_iou_scores.append(consistency_analysis['iou'])
                    all_area_change_ratios.append(consistency_analysis['area_change_ratio'])
                    all_hausdorff_distances.append(consistency_analysis['hausdorff_distance'])
                
            except Exception as e:
                print(f"Error processing annotation {ann_idx}, frame {frame_idx}: {e}")
    
    # Convert to numpy arrays for analysis
    component_sizes = np.array(all_component_sizes)
    hole_sizes = np.array(all_hole_sizes)
    convexity_ratios = np.array(all_convexity_ratios)
    smoothness_ratios = np.array(all_smoothness_ratios)
    complexity_scores = np.array(all_complexity_scores)
    mask_areas = np.array(all_mask_areas)
    boundary_ratios = np.array(all_boundary_ratios)
    skinny_part_areas = np.array(all_skinny_part_areas)
    motion_distances = np.array(all_motion_distances)
    iou_scores = np.array(all_iou_scores)
    area_change_ratios = np.array(all_area_change_ratios)
    hausdorff_distances = np.array(all_hausdorff_distances)
    internal_hole_areas = np.array(all_internal_hole_areas)
    multi_part_counts = np.array(all_multi_part_counts)
    
    # Print summary statistics
    print(f"\n=== DISTRIBUTION SUMMARY ===")
    print(f"Total masks analyzed: {len(mask_areas)}")
    print(f"Total components analyzed: {len(component_sizes)}")
    print(f"Total holes analyzed: {len(hole_sizes)}")
    print(f"Total skinny parts analyzed: {len(skinny_part_areas)}")
    print(f"Total motion distances analyzed: {len(motion_distances)}")
    print(f"Total consistency comparisons analyzed: {len(iou_scores)}")
    print(f"Total internal holes analyzed: {len(internal_hole_areas)}")
    print(f"Total multi-part masks analyzed: {len(multi_part_counts)}")
    
    # Create histograms
    create_issue_distribution_plots(
        component_sizes, hole_sizes, convexity_ratios, smoothness_ratios,
        complexity_scores, mask_areas, boundary_ratios, skinny_part_areas,
        motion_distances, iou_scores, area_change_ratios, hausdorff_distances,
        internal_hole_areas, multi_part_counts, output_dir
    )
    
    # Create three-panel histograms for each metric
    create_three_panel_histograms(
        component_sizes, hole_sizes, convexity_ratios, smoothness_ratios,
        complexity_scores, mask_areas, boundary_ratios, skinny_part_areas,
        motion_distances, iou_scores, area_change_ratios, hausdorff_distances,
        internal_hole_areas, multi_part_counts, output_dir
    )
    
    # Save detailed statistics
    save_issue_statistics(
        component_sizes, hole_sizes, convexity_ratios, smoothness_ratios,
        complexity_scores, mask_areas, boundary_ratios, skinny_part_areas,
        motion_distances, iou_scores, area_change_ratios, hausdorff_distances,
        internal_hole_areas, multi_part_counts, output_dir
    )
    
    return {
        'component_sizes': component_sizes,
        'hole_sizes': hole_sizes,
        'convexity_ratios': convexity_ratios,
        'smoothness_ratios': smoothness_ratios,
        'complexity_scores': complexity_scores,
        'mask_areas': mask_areas,
        'boundary_ratios': boundary_ratios,
        'skinny_part_areas': skinny_part_areas,
        'motion_distances': motion_distances,
        'iou_scores': iou_scores,
        'area_change_ratios': area_change_ratios,
        'hausdorff_distances': hausdorff_distances,
        'internal_hole_areas': internal_hole_areas,
        'multi_part_counts': multi_part_counts
    }

def create_issue_distribution_plots(component_sizes, hole_sizes, convexity_ratios, smoothness_ratios,
                                  complexity_scores, mask_areas, boundary_ratios, skinny_part_areas,
                                  motion_distances, iou_scores, area_change_ratios, hausdorff_distances,
                                  internal_hole_areas, multi_part_counts, output_dir):
    """Create histogram plots for all issue metrics"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a large figure with all metrics
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle('Mask Issue Metrics Distribution Analysis', fontsize=16)
    
    # 1. Component sizes
    if len(component_sizes) > 0:
        axes[0, 0].hist(component_sizes, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Component Size (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Component Size Distribution')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(component_sizes, p)
            axes[0, 0].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No components found', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Component Size Distribution')
    
    # 2. Hole sizes
    if len(hole_sizes) > 0:
        axes[0, 1].hist(hole_sizes, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_xlabel('Hole Size (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Hole Size Distribution')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(hole_sizes, p)
            axes[0, 1].axvline(value, color='blue', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No holes found', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Hole Size Distribution')
    
    # 3. Convexity ratios
    if len(convexity_ratios) > 0:
        axes[0, 2].hist(convexity_ratios, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].set_xlabel('Convexity Ratio')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Convexity Ratio Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            value = np.percentile(convexity_ratios, p)
            axes[0, 2].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[0, 2].legend()
    
    # 4. Smoothness ratios
    if len(smoothness_ratios) > 0:
        axes[1, 0].hist(smoothness_ratios, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Smoothness Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Smoothness Ratio Distribution')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(smoothness_ratios, p)
            axes[1, 0].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[1, 0].legend()
    
    # 5. Complexity scores
    if len(complexity_scores) > 0:
        axes[1, 1].hist(complexity_scores, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Complexity Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Complexity Score Distribution')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(complexity_scores, p)
            axes[1, 1].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[1, 1].legend()
    
    # 6. Mask areas
    if len(mask_areas) > 0:
        axes[1, 2].hist(mask_areas, bins=50, alpha=0.7, color='brown', edgecolor='black')
        axes[1, 2].set_xlabel('Mask Area (pixels)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Mask Area Distribution')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(mask_areas, p)
            axes[1, 2].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[1, 2].legend()
    
    # 7. Boundary ratios
    if len(boundary_ratios) > 0:
        axes[2, 0].hist(boundary_ratios, bins=50, alpha=0.7, color='pink', edgecolor='black')
        axes[2, 0].set_xlabel('Boundary Ratio')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Boundary Ratio Distribution')
        axes[2, 0].set_yscale('log')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(boundary_ratios, p)
            axes[2, 0].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[2, 0].legend()
    
    # 8. Skinny part areas
    if len(skinny_part_areas) > 0:
        axes[2, 1].hist(skinny_part_areas, bins=50, alpha=0.7, color='cyan', edgecolor='black')
        axes[2, 1].set_xlabel('Skinny Part Area (pixels)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Skinny Part Area Distribution')
        axes[2, 1].set_yscale('log')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(skinny_part_areas, p)
            axes[2, 1].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[2, 1].legend()
    else:
        axes[2, 1].text(0.5, 0.5, 'No skinny parts found', ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Skinny Part Area Distribution')
    
    # 9. Motion distances
    if len(motion_distances) > 0:
        axes[2, 2].hist(motion_distances, bins=50, alpha=0.7, color='teal', edgecolor='black')
        axes[2, 2].set_xlabel('Motion Distance (pixels)')
        axes[2, 2].set_ylabel('Frequency')
        axes[2, 2].set_title('Motion Distance Distribution')
        axes[2, 2].set_yscale('log')
        axes[2, 2].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(motion_distances, p)
            axes[2, 2].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[2, 2].legend()
    else:
        axes[2, 2].text(0.5, 0.5, 'No motion data found', ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('Motion Distance Distribution')
    
    # 10. IoU scores
    if len(iou_scores) > 0:
        axes[2, 3].hist(iou_scores, bins=50, alpha=0.7, color='indigo', edgecolor='black')
        axes[2, 3].set_xlabel('IoU Score')
        axes[2, 3].set_ylabel('Frequency')
        axes[2, 3].set_title('IoU Score Distribution')
        axes[2, 3].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            value = np.percentile(iou_scores, p)
            axes[2, 3].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[2, 3].legend()
    else:
        axes[2, 3].text(0.5, 0.5, 'No IoU data found', ha='center', va='center', transform=axes[2, 3].transAxes)
        axes[2, 3].set_title('IoU Score Distribution')
    
    # 11. Area change ratios
    if len(area_change_ratios) > 0:
        axes[3, 0].hist(area_change_ratios, bins=50, alpha=0.7, color='olive', edgecolor='black')
        axes[3, 0].set_xlabel('Area Change Ratio')
        axes[3, 0].set_ylabel('Frequency')
        axes[3, 0].set_title('Area Change Ratio Distribution')
        axes[3, 0].set_yscale('log')
        axes[3, 0].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(area_change_ratios, p)
            axes[3, 0].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[3, 0].legend()
    else:
        axes[3, 0].text(0.5, 0.5, 'No area change data found', ha='center', va='center', transform=axes[3, 0].transAxes)
        axes[3, 0].set_title('Area Change Ratio Distribution')
    
    # 12. Hausdorff distances
    if len(hausdorff_distances) > 0:
        axes[3, 1].hist(hausdorff_distances, bins=50, alpha=0.7, color='maroon', edgecolor='black')
        axes[3, 1].set_xlabel('Hausdorff Distance (pixels)')
        axes[3, 1].set_ylabel('Frequency')
        axes[3, 1].set_title('Hausdorff Distance Distribution')
        axes[3, 1].set_yscale('log')
        axes[3, 1].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(hausdorff_distances, p)
            axes[3, 1].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[3, 1].legend()
    else:
        axes[3, 1].text(0.5, 0.5, 'No Hausdorff data found', ha='center', va='center', transform=axes[3, 1].transAxes)
        axes[3, 1].set_title('Hausdorff Distance Distribution')
    
    # 13. Internal hole areas
    if len(internal_hole_areas) > 0:
        axes[3, 2].hist(internal_hole_areas, bins=50, alpha=0.7, color='navy', edgecolor='black')
        axes[3, 2].set_xlabel('Internal Hole Area (pixels)')
        axes[3, 2].set_ylabel('Frequency')
        axes[3, 2].set_title('Internal Hole Area Distribution')
        axes[3, 2].set_yscale('log')
        axes[3, 2].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(internal_hole_areas, p)
            axes[3, 2].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[3, 2].legend()
    else:
        axes[3, 2].text(0.5, 0.5, 'No internal holes found', ha='center', va='center', transform=axes[3, 2].transAxes)
        axes[3, 2].set_title('Internal Hole Area Distribution')
    
    # 14. Multi-part counts
    if len(multi_part_counts) > 0:
        axes[3, 3].hist(multi_part_counts, bins=20, alpha=0.7, color='darkgreen', edgecolor='black')
        axes[3, 3].set_xlabel('Number of Components')
        axes[3, 3].set_ylabel('Frequency')
        axes[3, 3].set_title('Multi-part Component Count Distribution')
        axes[3, 3].grid(True, alpha=0.3)
        
        # Add percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(multi_part_counts, p)
            axes[3, 3].axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{p}th percentile')
        axes[3, 3].legend()
    else:
        axes[3, 3].text(0.5, 0.5, 'No multi-part masks found', ha='center', va='center', transform=axes[3, 3].transAxes)
        axes[3, 3].set_title('Multi-part Component Count Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'issue_metrics_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"Issue metrics distributions saved to: {os.path.join(output_dir, 'issue_metrics_distributions.png')}")
    plt.close()
    
    # Create cumulative distribution plots
    create_cumulative_plots(
        component_sizes, hole_sizes, convexity_ratios, smoothness_ratios,
        complexity_scores, mask_areas, boundary_ratios, skinny_part_areas,
        motion_distances, iou_scores, area_change_ratios, hausdorff_distances,
        internal_hole_areas, multi_part_counts, output_dir
    )

def create_cumulative_plots(component_sizes, hole_sizes, convexity_ratios, smoothness_ratios,
                           complexity_scores, mask_areas, boundary_ratios, skinny_part_areas,
                           motion_distances, iou_scores, area_change_ratios, hausdorff_distances,
                           internal_hole_areas, multi_part_counts, output_dir):
    """Create cumulative distribution plots for all metrics"""
    
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle('Cumulative Distributions for Threshold Selection', fontsize=16)
    
    metrics = [
        (component_sizes, 'Component Sizes', 'blue', axes[0, 0]),
        (hole_sizes, 'Hole Sizes', 'red', axes[0, 1]),
        (convexity_ratios, 'Convexity Ratios', 'green', axes[0, 2]),
        (smoothness_ratios, 'Smoothness Ratios', 'orange', axes[0, 3]),
        (complexity_scores, 'Complexity Scores', 'purple', axes[1, 0]),
        (mask_areas, 'Mask Areas', 'brown', axes[1, 1]),
        (boundary_ratios, 'Boundary Ratios', 'pink', axes[1, 2]),
        (skinny_part_areas, 'Skinny Part Areas', 'cyan', axes[1, 3]),
        (motion_distances, 'Motion Distances', 'teal', axes[2, 0]),
        (iou_scores, 'IoU Scores', 'indigo', axes[2, 1]),
        (area_change_ratios, 'Area Change Ratios', 'olive', axes[2, 2]),
        (hausdorff_distances, 'Hausdorff Distances', 'maroon', axes[2, 3]),
        (internal_hole_areas, 'Internal Hole Areas', 'navy', axes[3, 0]),
        (multi_part_counts, 'Multi-part Counts', 'darkgreen', axes[3, 1])
    ]
    
    for data, title, color, ax in metrics:
        if len(data) > 0:
            sorted_data = np.sort(data)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cumulative, color=color, linewidth=2)
            ax.set_xlabel(title.split()[0])
            ax.set_ylabel('Cumulative Probability')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add reference lines for key percentiles
            for percentile in [25, 50, 75, 90, 95]:
                value = np.percentile(data, percentile)
                ax.axvline(value, color='red', linestyle='--', alpha=0.7, label=f'{percentile}th percentile')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'No {title.lower()} found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'issue_metrics_cumulative.png'), dpi=300, bbox_inches='tight')
    print(f"Cumulative distributions saved to: {os.path.join(output_dir, 'issue_metrics_cumulative.png')}")
    plt.close()

def create_three_panel_histograms(component_sizes, hole_sizes, convexity_ratios, smoothness_ratios,
                                 complexity_scores, mask_areas, boundary_ratios, skinny_part_areas,
                                 motion_distances, iou_scores, area_change_ratios, hausdorff_distances,
                                 internal_hole_areas, multi_part_counts, output_dir):
    """Create three-panel histograms for each metric: full distribution, 10th percentile and below, 90th percentile and above"""
    
    metrics = [
        (component_sizes, 'Component Sizes', 'blue'),
        (hole_sizes, 'Hole Sizes', 'red'),
        (convexity_ratios, 'Convexity Ratios', 'green'),
        (smoothness_ratios, 'Smoothness Ratios', 'orange'),
        (complexity_scores, 'Complexity Scores', 'purple'),
        (mask_areas, 'Mask Areas', 'brown'),
        (boundary_ratios, 'Boundary Ratios', 'pink'),
        (skinny_part_areas, 'Skinny Part Areas', 'cyan'),
        (motion_distances, 'Motion Distances', 'teal'),
        (iou_scores, 'IoU Scores', 'indigo'),
        (area_change_ratios, 'Area Change Ratios', 'olive'),
        (hausdorff_distances, 'Hausdorff Distances', 'maroon'),
        (internal_hole_areas, 'Internal Hole Areas', 'navy'),
        (multi_part_counts, 'Multi-part Counts', 'darkgreen')
    ]
    
    for data, title, color in metrics:
        if len(data) == 0:
            print(f"Skipping {title} - no data available")
            continue
            
        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{title} Distribution Analysis', fontsize=14)
        
        # Calculate percentiles
        p10 = np.percentile(data, 10)
        p90 = np.percentile(data, 90)
        
        # 1. Full distribution
        axes[0].hist(data, bins=50, alpha=0.7, color=color, edgecolor='black')
        axes[0].set_xlabel(title.split()[0])
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Full Distribution')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Add percentile lines
        axes[0].axvline(p10, color='red', linestyle='--', alpha=0.8, label=f'10th percentile: {p10:.2f}')
        axes[0].axvline(p90, color='orange', linestyle='--', alpha=0.8, label=f'90th percentile: {p90:.2f}')
        axes[0].legend()
        
        # 2. 10th percentile and below
        low_data = data[data <= p10]
        if len(low_data) > 0:
            axes[1].hist(low_data, bins=30, alpha=0.7, color='red', edgecolor='black')
            axes[1].set_xlabel(title.split()[0])
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'10th Percentile and Below\n({len(low_data)} samples)')
            axes[1].grid(True, alpha=0.3)
            
            # Add statistics
            mean_low = np.mean(low_data)
            median_low = np.median(low_data)
            axes[1].axvline(mean_low, color='blue', linestyle='-', alpha=0.8, label=f'Mean: {mean_low:.2f}')
            axes[1].axvline(median_low, color='green', linestyle='-', alpha=0.8, label=f'Median: {median_low:.2f}')
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No data in 10th percentile', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('10th Percentile and Below')
        
        # 3. 90th percentile and above
        high_data = data[data >= p90]
        if len(high_data) > 0:
            axes[2].hist(high_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[2].set_xlabel(title.split()[0])
            axes[2].set_ylabel('Frequency')
            axes[2].set_title(f'90th Percentile and Above\n({len(high_data)} samples)')
            axes[2].grid(True, alpha=0.3)
            
            # Add statistics
            mean_high = np.mean(high_data)
            median_high = np.median(high_data)
            axes[2].axvline(mean_high, color='blue', linestyle='-', alpha=0.8, label=f'Mean: {mean_high:.2f}')
            axes[2].axvline(median_high, color='green', linestyle='-', alpha=0.8, label=f'Median: {median_high:.2f}')
            axes[2].legend()
        else:
            axes[2].text(0.5, 0.5, 'No data in 90th percentile', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('90th Percentile and Above')
        
        plt.tight_layout()
        
        # Save individual metric plot
        safe_title = title.replace(' ', '_').lower()
        filename = f'{safe_title}_three_panel_histogram.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"{title} three-panel histogram saved to: {os.path.join(output_dir, filename)}")
        plt.close()
    
    # Create a summary plot with all metrics in a grid
    create_summary_three_panel_grid(metrics, output_dir)

def create_summary_three_panel_grid(metrics, output_dir):
    """Create a summary grid showing the three-panel view for all metrics"""
    
    # Filter out metrics with no data
    valid_metrics = [(data, title, color) for data, title, color in metrics if len(data) > 0]
    
    if not valid_metrics:
        print("No valid metrics to plot in summary grid")
        return
    
    # Calculate grid dimensions
    n_metrics = len(valid_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle('Summary: Full Distribution, 10th Percentile and Below, 90th Percentile and Above', fontsize=16)
    
    # Flatten axes if needed
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (data, title, color) in enumerate(valid_metrics):
        row = idx // n_cols
        col = idx % n_cols
        
        # Calculate percentiles
        p10 = np.percentile(data, 10)
        p90 = np.percentile(data, 90)
        
        # Create subplot with three panels
        ax = axes[row, col]
        
        # Plot full distribution
        ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black')
        ax.set_xlabel(title.split()[0])
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title}\nFull Distribution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add percentile lines
        ax.axvline(p10, color='red', linestyle='--', alpha=0.8, label=f'P10: {p10:.2f}')
        ax.axvline(p90, color='orange', linestyle='--', alpha=0.8, label=f'P90: {p90:.2f}')
        ax.legend(fontsize=8)
        
        # Add statistics text
        stats_text = f'Count: {len(data)}\nMean: {np.mean(data):.2f}\nP10: {p10:.2f}\nP90: {p90:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(valid_metrics), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_three_panel_grid.png'), dpi=300, bbox_inches='tight')
    print(f"Summary three-panel grid saved to: {os.path.join(output_dir, 'summary_three_panel_grid.png')}")
    plt.close()

def save_issue_statistics(component_sizes, hole_sizes, convexity_ratios, smoothness_ratios,
                         complexity_scores, mask_areas, boundary_ratios, skinny_part_areas,
                         motion_distances, iou_scores, area_change_ratios, hausdorff_distances,
                         internal_hole_areas, multi_part_counts, output_dir):
    """Save detailed statistics to JSON file"""
    
    def get_statistics(data, name):
        if len(data) == 0:
            return {
                'count': 0,
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'percentiles': {str(p): 0 for p in [10, 25, 50, 75, 90, 95, 99]}
            }
        
        return {
            'count': len(data),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'std': float(np.std(data)),
            'percentiles': {str(p): float(np.percentile(data, p)) for p in [10, 25, 50, 75, 90, 95, 99]}
        }
    
    stats = {
        'component_sizes': get_statistics(component_sizes, 'component_sizes'),
        'hole_sizes': get_statistics(hole_sizes, 'hole_sizes'),
        'convexity_ratios': get_statistics(convexity_ratios, 'convexity_ratios'),
        'smoothness_ratios': get_statistics(smoothness_ratios, 'smoothness_ratios'),
        'complexity_scores': get_statistics(complexity_scores, 'complexity_scores'),
        'mask_areas': get_statistics(mask_areas, 'mask_areas'),
        'boundary_ratios': get_statistics(boundary_ratios, 'boundary_ratios'),
        'skinny_part_areas': get_statistics(skinny_part_areas, 'skinny_part_areas'),
        'motion_distances': get_statistics(motion_distances, 'motion_distances'),
        'iou_scores': get_statistics(iou_scores, 'iou_scores'),
        'area_change_ratios': get_statistics(area_change_ratios, 'area_change_ratios'),
        'hausdorff_distances': get_statistics(hausdorff_distances, 'hausdorff_distances'),
        'internal_hole_areas': get_statistics(internal_hole_areas, 'internal_hole_areas'),
        'multi_part_counts': get_statistics(multi_part_counts, 'multi_part_counts')
    }
    
    stats_file = os.path.join(output_dir, 'issue_metrics_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Detailed issue metrics statistics saved to: {stats_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze mask component and hole size distributions')
    parser.add_argument('--json_file', default="/data/fishway_ytvis/all_videos.json", 
                       help='Path to the JSON dataset file')
    parser.add_argument('--output_dir', default="/data/fishway_ytvis/mask_analysis", 
                       help='Path to the output directory')
    
    args = parser.parse_args()
    
    # Analyze distributions
    results = analyze_dataset_distributions(args.json_file, args.output_dir)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Results saved to: {args.output_dir}")
    print(f"Generated files:")
    print(f"  - issue_metrics_distributions.png: Histogram plots for all metrics")
    print(f"  - issue_metrics_cumulative.png: Cumulative distribution plots")
    print(f"  - summary_three_panel_grid.png: Summary grid of all metrics")
    print(f"  - [metric]_three_panel_histogram.png: Individual three-panel histograms for each metric")
    print(f"  - issue_metrics_statistics.json: Detailed statistics") 