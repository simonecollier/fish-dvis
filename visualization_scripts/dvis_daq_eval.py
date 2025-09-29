"""
Standalone DVIS-DAQ evaluation functions
This module provides the exact same evaluation methodology as DVIS-DAQ
without requiring relative imports.
"""

import numpy as np
import datetime
import time
from collections import defaultdict
import copy
import math
import json
import sys
import os
import subprocess
import tempfile
import shutil

# Use installed pycocotools for mask utilities
import pycocotools.mask as maskUtils

def extract_per_category_metrics(coco_eval, coco_gt):
    """
    Extract comprehensive per-category metrics including AP, AP50, AP75, APs, APm, APl, AR1, AR10.
    """
    per_category_metrics = {}
    
    # Get class names from ground truth
    cat_ids = coco_gt.getCatIds()
    cat_names = {}
    for cat in coco_gt.loadCats(cat_ids):
        cat_names[cat['id']] = cat['name']
    
    # Get precision and recall arrays
    precisions = coco_eval.eval["precision"]  # Shape: (T,R,K,A,O,M)
    recalls = coco_eval.eval["recall"]        # Shape: (T,K,A,O,M)
    
    # Get parameters
    params = coco_eval.params
    iou_thresholds = params.iouThrs
    area_ranges = params.areaRng
    max_dets = params.maxDets
    
    # Find indices for different IoU thresholds
    iou_50_idx = np.where(np.array(iou_thresholds) == 0.5)[0][0] if 0.5 in iou_thresholds else 0
    iou_75_idx = np.where(np.array(iou_thresholds) == 0.75)[0][0] if 0.75 in iou_thresholds else 2
    
    # Find indices for area ranges (small, medium, large)
    area_indices = {}
    for i, area_range in enumerate(area_ranges):
        if area_range[0] == 0 and area_range[1] < 32**2:  # small
            area_indices['small'] = i
        elif area_range[0] >= 32**2 and area_range[1] < 96**2:  # medium
            area_indices['medium'] = i
        elif area_range[0] >= 96**2:  # large
            area_indices['large'] = i
    
    # Find maxDets indices
    max_dets_array = np.array(max_dets)
    max_dets_1_idx = np.where(max_dets_array == 1)[0][0] if 1 in max_dets else 0
    max_dets_10_idx = np.where(max_dets_array == 10)[0][0] if 10 in max_dets else 1
    max_dets_100_idx = np.where(max_dets_array == 100)[0][0] if 100 in max_dets else 2
    
    for idx, cat_id in enumerate(cat_ids):
        cat_name = cat_names[cat_id]
        cat_metrics = {}
        
        # Calculate AP (mean over all IoU thresholds)
        precision_all = precisions[:, :, idx, 0, 0, max_dets_100_idx]  # all areas, all occ
        all_precision = precision_all.flatten()
        all_precision = all_precision[all_precision > -1]
        ap = np.mean(all_precision) if all_precision.size > 0 else float("nan")
        cat_metrics['AP'] = float(ap * 100)
        
        # Calculate AP50
        precision_50 = precisions[iou_50_idx, :, idx, 0, 0, max_dets_100_idx]
        precision_50 = precision_50[precision_50 > -1]
        ap50 = np.mean(precision_50) if precision_50.size > 0 else float("nan")
        cat_metrics['AP50'] = float(ap50 * 100)
        
        # Calculate AP75
        precision_75 = precisions[iou_75_idx, :, idx, 0, 0, max_dets_100_idx]
        precision_75 = precision_75[precision_75 > -1]
        ap75 = np.mean(precision_75) if precision_75.size > 0 else float("nan")
        cat_metrics['AP75'] = float(ap75 * 100)
        
        # Calculate APs, APm, APl (area-specific)
        for area_name, area_idx in area_indices.items():
            if area_name in ['small', 'medium', 'large']:
                precision_area = precisions[iou_50_idx, :, idx, area_idx, 0, max_dets_100_idx]
                precision_area = precision_area[precision_area > -1]
                ap_area = np.mean(precision_area) if precision_area.size > 0 else float("nan")
                cat_metrics[f'AP{area_name[0]}'] = float(ap_area * 100)
        
        # Calculate AR1, AR10
        recall_1 = recalls[:, idx, 0, 0, max_dets_1_idx]
        recall_1 = recall_1[recall_1 > -1]
        ar1 = np.mean(recall_1) if recall_1.size > 0 else float("nan")
        cat_metrics['AR1'] = float(ar1 * 100)
        
        recall_10 = recalls[:, idx, 0, 0, max_dets_10_idx]
        recall_10 = recall_10[recall_10 > -1]
        ar10 = np.mean(recall_10) if recall_10.size > 0 else float("nan")
        cat_metrics['AR10'] = float(ar10 * 100)
        
        per_category_metrics[cat_name] = cat_metrics
    
    return per_category_metrics

def run_dvis_daq_evaluation_with_renamed_pycocotools(predictions, ground_truth_json_path):
    """
    Run DVIS-DAQ evaluation by temporarily renaming the pycocotools folder to avoid import conflicts.
    """
    dvis_daq_path = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_Plus/data_video/datasets'
    pycocotools_path = os.path.join(dvis_daq_path, 'pycocotools')
    pycocotools_backup_path = os.path.join(dvis_daq_path, 'pycocotools_backup')
    
    # Check if pycocotools folder exists
    if not os.path.exists(pycocotools_path):
        print(f"Error: pycocotools folder not found at {pycocotools_path}")
        return None
    
    try:
        # Temporarily rename pycocotools to pycocotools_backup
        print("DVIS-DAQ: Renaming pycocotools folder...")
        if os.path.exists(pycocotools_backup_path):
            shutil.rmtree(pycocotools_backup_path)
        shutil.move(pycocotools_path, pycocotools_backup_path)
        
        # Add the datasets path to sys.path
        sys.path.insert(0, dvis_daq_path)
        
        # Now import the DVIS-DAQ evaluation components
        print("DVIS-DAQ: Importing evaluation components...")
        from pycocotools_backup.oviseval import OVISeval
        from ytvis_api.ytvos import YTVOS
        from detectron2.data import MetadataCatalog
        from detectron2.utils.file_io import PathManager
        
        # Register dataset
        dataset_name = "ytvis_fishway_val"
        MetadataCatalog.get(dataset_name).set(
            json_file=ground_truth_json_path,
            image_root="/data/fishway_ytvis/all_videos",
            evaluator_type="ytvis",
        )
        
        # Load ground truth
        metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(metadata.json_file)
        coco_gt = YTVOS(json_file)
        
        # Convert predictions
        coco_results = []
        for pred in predictions:
            segmentations = []
            for seg in pred['segmentations']:
                if seg is not None:
                    segmentations.append(seg)
                else:
                    segmentations.append(None)
            
            coco_results.append({
                'video_id': pred['video_id'],
                'score': pred['score'],
                'category_id': pred['category_id'],
                'segmentations': segmentations
            })
        
        # Load predictions
        coco_dt = coco_gt.loadRes(coco_results)
        
        # Run evaluation - use exact same parameters as DVIS-DAQ
        coco_eval = OVISeval(coco_gt, coco_dt)
        # For COCO, the default max_dets_per_image is [1, 10, 100].
        max_dets_per_image = [1, 10, 100]  # Default from COCOEval
        coco_eval.params.maxDets = max_dets_per_image
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {}
        if len(coco_eval.stats) >= 18:
            metrics['AP'] = coco_eval.stats[0]
            metrics['AP50'] = coco_eval.stats[1]
            metrics['AP75'] = coco_eval.stats[2]
            metrics['APs'] = coco_eval.stats[3]
            metrics['APm'] = coco_eval.stats[4]
            metrics['APl'] = coco_eval.stats[5]
            metrics['AR1'] = coco_eval.stats[9]
            metrics['AR10'] = coco_eval.stats[10]
            metrics['AR100'] = coco_eval.stats[11]

        # Extract comprehensive per-category metrics
        per_category_metrics = extract_per_category_metrics(coco_eval, coco_gt)
        print("Per-category metrics:")
        for category, cat_metrics in per_category_metrics.items():
            print(f"  {category}: AP={cat_metrics['AP']:.3f}, AP50={cat_metrics['AP50']:.3f}, AP75={cat_metrics['AP75']:.3f}")

        # Add to metrics
        metrics['per_category_metrics'] = per_category_metrics

        print("DVIS-DAQ: Evaluation completed successfully!")
        return metrics
        
    except Exception as e:
        print(f"DVIS-DAQ: Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Restore the pycocotools folder
        print("DVIS-DAQ: Restoring pycocotools folder...")
        try:
            if os.path.exists(pycocotools_backup_path):
                if os.path.exists(pycocotools_path):
                    shutil.rmtree(pycocotools_path)
                shutil.move(pycocotools_backup_path, pycocotools_path)
                print("DVIS-DAQ: Successfully restored pycocotools folder")
        except Exception as e:
            print(f"DVIS-DAQ: Warning: Failed to restore pycocotools folder: {e}")

def run_dvis_daq_evaluation_subprocess(predictions, ground_truth_json_path):
    """
    Run DVIS-DAQ evaluation as a subprocess to avoid import issues.
    """
    try:
        # Save predictions to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(predictions, f)
            temp_pred_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_output_file = f.name
        
        # Run DVIS-DAQ evaluation as subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [
            sys.executable, 
            os.path.join(script_dir, 'run_dvis_daq_eval.py'),
            '--results-json', temp_pred_file,
            '--val-json', ground_truth_json_path,
            '--output', temp_output_file
        ]
        
        print("Running DVIS-DAQ evaluation as subprocess...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            # Load results
            with open(temp_output_file, 'r') as f:
                metrics = json.load(f)
            
            # Clean up temporary files
            os.unlink(temp_pred_file)
            os.unlink(temp_output_file)
            
            print("DVIS-DAQ evaluation completed successfully!")
            return metrics
        else:
            print(f"DVIS-DAQ evaluation failed: {result.stderr}")
            # Clean up temporary files
            os.unlink(temp_pred_file)
            os.unlink(temp_output_file)
            return None
            
    except Exception as e:
        print(f"Error in DVIS-DAQ evaluation: {e}")
        return None

def compute_dvis_daq_metrics(predictions, ground_truth_json_path, dataset_name="ytvis_fishway_val"):
    """
    Compute metrics using the exact DVIS-DAQ evaluation methodology.
    This function uses the renamed pycocotools approach to avoid import issues.
    """
    return run_dvis_daq_evaluation_with_renamed_pycocotools(predictions, ground_truth_json_path)
