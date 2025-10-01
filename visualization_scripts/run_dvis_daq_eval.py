#!/usr/bin/env python3
"""
Standalone DVIS-DAQ evaluation script
This script can be run from the DVIS-DAQ environment to evaluate predictions
"""

import sys
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run DVIS-DAQ evaluation')
    parser.add_argument('--results-json', required=True, help='Path to results.json')
    parser.add_argument('--val-json', required=True, help='Path to val.json')
    parser.add_argument('--output', required=True, help='Output file for metrics')
    args = parser.parse_args()
    
    try:
        # Add DVIS-DAQ path
        dvis_daq_path = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_Plus/data_video/datasets'
        sys.path.insert(0, dvis_daq_path)
        
        # Import DVIS-DAQ components
        from pycocotools.oviseval import OVISeval
        from pycocotools.ytvis_api.ytvos import YTVOS
        from detectron2.data import MetadataCatalog
        from detectron2.utils.file_io import PathManager
        
        # Load data
        with open(args.results_json, 'r') as f:
            predictions = json.load(f)
        
        # Register dataset
        dataset_name = "ytvis_fishway_val"
        MetadataCatalog.get(dataset_name).set(
            json_file=args.val_json,
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
        
        # Run evaluation
        coco_eval = OVISeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.params.maxDets = [1, 10, 100]
        coco_eval.params.iouThrs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        coco_eval.params.recThrs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        
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
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("SUCCESS")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
