# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script with Attention Extraction.

This script is a modified version of the training script in detectron2/tools
with added attention extraction capabilities.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
except:
    pass

import copy
import itertools
import logging
import os
import gc
import json
import torch
import torch.nn.functional as F

from collections import OrderedDict
from typing import Any, Dict, List, Set

torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    hooks
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# Models
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from dvis_Plus import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    PanopticDatasetVideoMapper,
    SemanticDatasetVideoMapper,
    YTVISEvaluator,
    VPSEvaluator,
    VSSEvaluator,
    add_minvis_config,
    add_dvis_config,
    add_ctvis_config,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
    UniYTVISEvaluator,
    SOTDatasetMapper,
)
from dvis_daq.config import add_daq_config
from dvis_Plus.data_video.datasets.builtin import register_ytvis_instances


class AttentionExtractor:
    """Class to manage attention extraction hooks and data saving."""
    
    def __init__(self, model, output_dir, top_n=1):
        self.model = model
        self.output_dir = output_dir
        self.top_n = top_n
        self.hooks = []
        self.attention_data = []
        self.video_attention_data = {}  # video_id -> attention_data
        self.current_video_id = None
        self.video_pred_info = None  # Store prediction info for the current video
        
        # Register hooks for refiner attention layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention maps."""
        # Only register refiner_time_self_attn hooks
        for name, module in self.model.named_modules():
            if 'refiner_time_self_attn' in name and hasattr(module, 'self_attn'):
                hook = AttentionHook(name, module.self_attn)
                module.self_attn.register_forward_hook(hook)
                self.hooks.append(hook)
                print(f"Registered hook for: {name}")
    
    def clear_attention_data(self):
        """Clear accumulated attention data."""
        for hook in self.hooks:
            hook.attention_maps.clear()
        self.attention_data.clear()
    
    def accumulate_attention_data(self, video_id, pred_info=None):
        """Accumulate attention data for a video, filtered for top N predictions."""
        if not any(hook.attention_maps for hook in self.hooks):
            return
        
        # Collect all attention maps
        all_attention_maps = []
        for hook in self.hooks:
            all_attention_maps.extend(hook.attention_maps)
        
        if not all_attention_maps:
            return
        
        # Initialize video data structure if this is a new video
        if video_id != self.current_video_id:
            if self.current_video_id is not None:
                # Save the previous video's data
                self.save_video_attention_data()
            self.current_video_id = video_id
            self.video_attention_data[video_id] = {}
            self.video_pred_info = pred_info
        
        # Filter attention maps for top N predictions
        filtered_attention_maps = []
        top_instances_info = []
        
        if pred_info is not None and pred_info.get('confidence_scores') is not None:
            confidence_scores = pred_info['confidence_scores']
            pred_ids = pred_info.get('pred_ids', None)
            
            # Convert pred_ids to list if it's a tensor
            if pred_ids is not None and hasattr(pred_ids, 'cpu'):
                pred_ids = pred_ids.cpu().numpy().tolist()
            elif pred_ids is not None:
                pred_ids = list(pred_ids)
            
            # Create list of (confidence, instance_idx, seq_id) tuples
            instance_confidence_pairs = []
            for i, conf in enumerate(confidence_scores):
                seq_id = pred_ids[i] if pred_ids and i < len(pred_ids) else i
                instance_confidence_pairs.append((conf, i, seq_id))
            
            # Sort by confidence (descending) and take top N
            instance_confidence_pairs.sort(key=lambda x: x[0], reverse=True)
            top_instances = instance_confidence_pairs[:self.top_n]
            
            logger = logging.getLogger(__name__)
            logger.info(f"Selected top {len(top_instances)} instances out of {len(confidence_scores)} total instances")
            
            # Filter attention maps for top instances
            for attn_map in all_attention_maps:
                attn_weights = attn_map['attention_weights']  # Shape: [num_instances, num_frames, num_frames]
                
                # Extract attention maps for top instances
                for rank, (conf, instance_idx, seq_id) in enumerate(top_instances):
                    if instance_idx < attn_weights.shape[0]:  # Ensure valid index
                        filtered_weights = attn_weights[instance_idx].clone()
                        
                        filtered_attention_maps.append({
                            'layer': f"{attn_map['layer']}_top_{rank+1}",
                            'attention_weights': filtered_weights,
                            'shape': filtered_weights.shape,
                            'instance_info': {
                                'instance_id': instance_idx,
                                'sequence_id': seq_id,
                                'prediction_id': rank,  # Final prediction rank (0-based)
                                'confidence_score': conf,
                                'rank': rank + 1  # Human-readable rank (1-based)
                            }
                        })
                        
                        top_instances_info.append({
                            'rank': rank + 1,
                            'instance_id': instance_idx,
                            'sequence_id': seq_id,
                            'prediction_id': rank,
                            'confidence_score': conf
                        })
            
            logger.info(f"Filtered to {len(filtered_attention_maps)} attention maps for top {self.top_n} predictions")
        else:
            # If no prediction info, save all attention maps (fallback)
            filtered_attention_maps = all_attention_maps
            logger = logging.getLogger(__name__)
            logger.warning("No prediction info available, saving all attention maps")
        
        # Store attention data for this video
        self.video_attention_data[video_id] = {
            'attention_maps': filtered_attention_maps,
            'num_maps': len(filtered_attention_maps),
            'top_instances_info': top_instances_info
        }
        
        # Clear data for next video
        self.clear_attention_data()
    
    def save_video_attention_data(self):
        """Save accumulated attention data for the current video."""
        if self.current_video_id is None or self.current_video_id not in self.video_attention_data:
            return
        
        video_id = self.current_video_id
        video_data = self.video_attention_data[video_id]
        
        if not video_data:
            return
        
        # Get the number of frames from the attention weights shape
        num_frames = 0
        if video_data['attention_maps']:
            # Get the shape from the first attention map
            first_attn_map = video_data['attention_maps'][0]
            if 'attention_weights' in first_attn_map:
                attn_weights = first_attn_map['attention_weights']
                if hasattr(attn_weights, 'shape'):
                    # Shape should be [num_frames, num_frames] for temporal attention
                    num_frames = attn_weights.shape[0]
                elif isinstance(attn_weights, list):
                    # If it's already converted to list, use the length
                    num_frames = len(attn_weights)
        
        # Prepare final save data
        save_data = {
            'video_id': video_id,
            'num_frames': num_frames,
            'attention_maps': video_data['attention_maps'],
            'num_maps': video_data['num_maps'],
            'top_n': self.top_n,
            'top_instances_info': video_data['top_instances_info']
        }
        
        # Add prediction information for mapping instances
        if self.video_pred_info is not None:
            pred_logits = self.video_pred_info.get('pred_logits', None)
            pred_masks = self.video_pred_info.get('pred_masks', None)
            
            save_data['prediction_info'] = {
                'num_predictions': self.video_pred_info.get('num_predictions', 0),
                'pred_ids': self.video_pred_info.get('pred_ids', None),
                'confidence_scores': self.video_pred_info.get('confidence_scores', None),
                'pred_logits_shape': list(pred_logits.shape) if pred_logits is not None and hasattr(pred_logits, 'shape') else None,
                'pred_masks_shape': list(pred_masks.shape) if pred_masks is not None and hasattr(pred_masks, 'shape') else None
            }
        
        # Save to file
        filename = f"attention_maps_video_{video_id}_top_{self.top_n}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert tensors to lists for JSON serialization
        json_data = copy.deepcopy(save_data)
        for attn_map in json_data['attention_maps']:
            if isinstance(attn_map['attention_weights'], torch.Tensor):
                attn_map['attention_weights'] = attn_map['attention_weights'].tolist()
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Saved attention data for video {video_id} with {num_frames} frames to {filepath}")
        logger.info(f"Top {self.top_n} instances: {video_data['top_instances_info']}")
    
    def finalize_video(self):
        """Finalize and save the current video's attention data."""
        if self.current_video_id is not None:
            self.save_video_attention_data()


class AttentionHook:
    """Hook to capture attention weights from MultiheadAttention layers."""
    
    def __init__(self, layer_name, attention_module):
        self.layer_name = layer_name
        self.attention_module = attention_module
        self.attention_maps = []
    
    def __call__(self, module, input, output):
        """Capture attention weights from the forward pass."""
        if isinstance(output, tuple) and len(output) >= 2:
            # MultiheadAttention returns (attn_output, attn_output_weights)
            attn_output, attn_weights = output[0], output[1]
            
            if attn_weights is not None:
                # Store attention weights with layer information
                self.attention_maps.append({
                    'layer': self.layer_name,
                    'attention_weights': attn_weights.detach().clone(),
                    'shape': attn_weights.shape
                })


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return YTVISEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def test_with_attention(cls, cfg, model, target_video_id=None, output_dir=None, top_n=1):
        """
        Run evaluation with attention extraction on the test set.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting evaluation with attention extraction")
        
        # Create attention extractor
        attention_extractor = AttentionExtractor(model, output_dir, top_n=top_n)
        
        # Build data loader with correct mapper for video instance datasets
        dataset_name = cfg.DATASETS.TEST[0]
        dataset_type = cfg.DATASETS.DATASET_TYPE_TEST[0]
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
            'vos': SOTDatasetMapper,
        }
        if dataset_type not in mapper_dict.keys():
            raise NotImplementedError
        mapper = mapper_dict[dataset_type](cfg, is_train=False)
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        
        # Build evaluator
        evaluator = cls.build_evaluator(cfg, cfg.DATASETS.TEST[0])
        
        # Run inference with attention extraction
        results_i = cls._inference_with_attention(
            model, data_loader, evaluator, attention_extractor, target_video_id
        )
        
        return results_i
    
    @classmethod
    def _inference_with_attention(cls, model, data_loader, evaluator, attention_extractor, target_video_id=None):
        """
        Custom inference function that extracts attention maps.
        """
        logger = logging.getLogger(__name__)
        
        # Set model to eval mode
        model.eval()
        
        # Process data loader
        total = len(data_loader)
        logger.info(f"Starting inference on {total} batches")
        
        for idx, inputs in enumerate(data_loader):
            # Check if we should process this batch
            if target_video_id is not None:
                # Extract video ID from inputs (assuming it's in the first input)
                input_data = inputs[0] if isinstance(inputs, list) else inputs
                if hasattr(input_data, 'get'):
                    video_id = input_data.get('video_id', None)
                    if video_id is not None and video_id != target_video_id:
                        logger.info(f"Skipping batch {idx}: video_id {video_id} != target {target_video_id}")
                        continue
                else:
                    logger.warning(f"Could not extract video_id from input, processing batch {idx}")
            
            logger.info(f"Processing batch {idx}/{total}")
            
            # Clear previous attention data
            attention_extractor.clear_attention_data()
            
            # Run inference
            with torch.no_grad():
                outputs = model(inputs)
            
            # Save attention data for this batch along with prediction info
            video_id = target_video_id if target_video_id is not None else idx
            
            # Extract sequence IDs and prediction info for mapping
            pred_logits = outputs.get("pred_logits", None)
            pred_masks = outputs.get("pred_masks", None)
            pred_ids = outputs.get("pred_ids", None)
            
            # Debug: print available keys
            logger.info(f"Available output keys: {list(outputs.keys())}")
            
            # Calculate confidence scores from logits
            confidence_scores = None
            if pred_logits is not None:
                confidence_scores = F.softmax(pred_logits, dim=-1).max(dim=-1)[0].cpu().numpy().tolist()
            else:
                # Try to get confidence from other sources
                if "scores" in outputs:
                    scores = outputs["scores"]
                    if hasattr(scores, 'cpu'):
                        confidence_scores = scores.cpu().numpy().tolist()
                    else:
                        confidence_scores = scores
                elif "pred_scores" in outputs:
                    pred_scores = outputs["pred_scores"]
                    if hasattr(pred_scores, 'cpu'):
                        confidence_scores = pred_scores.cpu().numpy().tolist()
                    else:
                        confidence_scores = pred_scores
            
            pred_info = {
                "pred_logits": pred_logits,
                "pred_masks": pred_masks, 
                "pred_ids": pred_ids,
                "confidence_scores": confidence_scores,
                "num_predictions": len(pred_logits) if pred_logits is not None else 0,
                "available_keys": list(outputs.keys())
            }
            
            attention_extractor.accumulate_attention_data(video_id, pred_info=pred_info)
            
            # Log instance mapping information
            if pred_info.get('pred_ids') is not None:
                seq_ids = pred_info['pred_ids'].cpu().numpy().tolist() if hasattr(pred_info['pred_ids'], 'cpu') else pred_info['pred_ids']
                logger.info(f"Video {video_id}: Found {pred_info['num_predictions']} predictions with sequence IDs: {seq_ids}")
            else:
                logger.info(f"Video {video_id}: No predictions found (refiner may have been skipped)")
            
            # Run evaluation on this batch
            # Skip evaluator for attention extraction to avoid the _predictions bug
            # evaluator.process(inputs, outputs)
            
            # Break if we found our target video
            if target_video_id is not None:
                logger.info(f"Found and processed target video {target_video_id}, stopping inference")
                break
        
        # Finalize the current video's attention data
        attention_extractor.finalize_video()
        
        # Process outputs through evaluator (commented out for attention extraction)
        # return evaluator.evaluate()
        return {}


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    add_dvis_config(cfg)
    add_ctvis_config(cfg)
    add_daq_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="minvis")
    
    # Register datasets
    register_ytvis_instances(
        "ytvis_fishway_train",
        {},
        "/data/fishway_ytvis/train.json",
        "/data/fishway_ytvis/all_videos"
    )
    register_ytvis_instances(
        "ytvis_fishway_val",
        {},
        "/data/fishway_ytvis/val.json",
        "/data/fishway_ytvis/all_videos"
    )
    
    return cfg


def main(args):
    cfg = setup(args)
    
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    
    if args.eval_only:
        if hasattr(args, 'extract_attention') and args.extract_attention:
            # Extract attention maps
            target_video_id = getattr(args, 'target_video_id', None)
            output_dir = getattr(args, 'attention_output_dir', None)
            top_n = getattr(args, 'top_n', 1)
            res = Trainer.test_with_attention(cfg, model, target_video_id, output_dir, top_n)
        else:
            # Regular evaluation
            res = Trainer.test(cfg, model)
        return res
    
    return Trainer.train(cfg, model, resume=args.resume)


if __name__ == "__main__":
    # Add custom arguments for attention extraction
    parser = default_argument_parser()
    parser.add_argument("--extract-attention", action="store_true", help="Extract attention maps during evaluation")
    parser.add_argument("--target-video-id", type=int, help="Target video ID to extract attention for")
    parser.add_argument("--attention-output-dir", type=str, help="Output directory for attention maps")
    parser.add_argument("--top-n", type=int, default=1, help="Number of top-scoring predictions to extract attention for (default: 1)")
    
    args = parser.parse_args()
    print("Command Line Args:", args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )