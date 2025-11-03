# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script with Attention Extraction.

This script is a modified version of the training script in detectron2/tools
with added attention extraction capabilities.
"""

# Model configuration - Update these variables as needed
MODEL_DIR = "/home/simone/store/simone/dvis-model-outputs/trained_models/model_silhouette_lr5e-4_redo"
CHECKPOINT_NUM = "0003635"
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
import numpy as np

from collections import OrderedDict
from typing import Any, Dict, List, Set

# Import RLE utilities
try:
    from pycocotools import mask as coco_mask
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. Masks will be saved as raw data.")

torch.multiprocessing.set_sharing_strategy('file_system')

_CATEGORIES_CACHE = None

def _load_categories_from_val():
    global _CATEGORIES_CACHE
    if _CATEGORIES_CACHE is not None:
        return _CATEGORIES_CACHE
    try:
        val_json_path = os.path.join(MODEL_DIR, "val.json")
        with open(val_json_path, 'r') as f:
            val_data = json.load(f)
        categories = val_data.get('categories', [])
        id_to_name = {c.get('id'): c.get('name') for c in categories if 'id' in c}
        _CATEGORIES_CACHE = {
            'id_to_name': id_to_name,
            'num_categories': len(categories),
        }
    except Exception:
        _CATEGORIES_CACHE = {
            'id_to_name': {},
            'num_categories': 0,
        }
    return _CATEGORIES_CACHE

def _map_model_index_to_category(model_index: int):
    cats = _load_categories_from_val()
    # Default mapping assumes dataset category ids are 1..N and model indices are 0..N-1
    dataset_category_id = (model_index + 1) if model_index is not None else None
    category_name = cats['id_to_name'].get(dataset_category_id) if dataset_category_id is not None else None
    return dataset_category_id, category_name

def convert_mask_to_rle(mask_tensor):
    """
    Convert a binary mask tensor to RLE format for efficient storage.
    
    Args:
        mask_tensor: Binary mask tensor of shape (H, W) or (T, H, W)
        
    Returns:
        RLE encoded mask or original tensor if conversion fails
    """
    if not PYCOCOTOOLS_AVAILABLE:
        return mask_tensor.tolist() if isinstance(mask_tensor, torch.Tensor) else mask_tensor
    
    try:
        # Convert to numpy if it's a tensor
        if isinstance(mask_tensor, torch.Tensor):
            mask_np = mask_tensor.cpu().numpy()
        else:
            mask_np = mask_tensor
        
        # Ensure it's binary (0s and 1s)
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        # Handle different tensor shapes
        if len(mask_np.shape) == 2:
            # Single mask: (H, W)
            rle = coco_mask.encode(np.asfortranarray(mask_np))
            return rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle
        elif len(mask_np.shape) == 3:
            # Video mask: (T, H, W) - convert each frame
            rles = []
            for t in range(mask_np.shape[0]):
                frame_mask = mask_np[t]
                rle = coco_mask.encode(np.asfortranarray(frame_mask))
                # Convert bytes to string if necessary
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode('utf-8')
                rles.append(rle)
            return rles
        else:
            # Fallback to list conversion
            return mask_np.tolist()
            
    except Exception as e:
        print(f"Warning: Failed to convert mask to RLE: {e}")
        # Fallback to list conversion
        if isinstance(mask_tensor, torch.Tensor):
            return mask_tensor.tolist()
        else:
            return mask_tensor

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
    
    def __init__(self, model, output_dir, top_n=1, rollout=False, simulate_skip_connection=True):
        self.model = model
        self.output_dir = output_dir
        self.top_n = top_n
        self.rollout = rollout
        self.simulate_skip_connection = simulate_skip_connection
        self.num_layers = 6  # DVIS-DAQ refiner has 6 layers
        self.hooks = []
        self.attention_data = []
        self.video_attention_data = {}  # video_id -> attention_data
        self.current_video_id = None
        self.video_pred_info = None  # Store prediction info for the current video
        self.refiner_seq_ids = None  # Store sequence IDs from refiner
        self.refiner_seq_id_list = None  # Store seq_id_list from common_inference
        
        # Register hooks for refiner attention layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention maps."""
        # Register refiner temporal self-attention hooks
        for name, module in self.model.named_modules():
            if 'transformer_time_self_attention_layers' in name and hasattr(module, 'self_attn'):
                hook = AttentionHook(name, module.self_attn)
                module.self_attn.register_forward_hook(hook)
                self.hooks.append(hook)
        
        # Hook to capture refiner sequence IDs
        if hasattr(self.model, 'refiner'):
            def refiner_hook(module, input, output):
                if len(input) > 0:
                    instance_embeds = input[0]
                    if hasattr(instance_embeds, 'shape') and len(instance_embeds.shape) == 4:
                        n_instances = instance_embeds.shape[-1]
                        self.refiner_seq_ids = list(range(n_instances))
            
            self.model.refiner.register_forward_hook(refiner_hook)
        
        # Hook to capture sequence IDs from run_window_inference
        if hasattr(self.model, 'run_window_inference'):
            def window_inference_hook(module, input, output):
                if isinstance(output, dict) and 'pred_ids' in output and output['pred_ids'] is not None:
                    seq_ids = output['pred_ids']
                    if hasattr(seq_ids, 'cpu'):
                        seq_ids = seq_ids.cpu().numpy().tolist()
                    self.refiner_seq_ids = seq_ids
            
            original_run_window_inference = self.model.run_window_inference
            def hooked_run_window_inference(*args, **kwargs):
                result = original_run_window_inference(*args, **kwargs)
                window_inference_hook(self.model, args, result)
                return result
            self.model.run_window_inference = hooked_run_window_inference
        
        # Hook to capture seq_id_list from common_inference
        if hasattr(self.model, 'common_inference'):
            def common_inference_hook(module, input, output):
                if isinstance(output, dict) and 'seq_id_list' in output and output['seq_id_list'] is not None:
                    self.refiner_seq_id_list = output['seq_id_list']
                return output
            
            original_common_inference = self.model.common_inference
            def hooked_common_inference(*args, **kwargs):
                result = original_common_inference(*args, **kwargs)
                common_inference_hook(self.model, args, result)
                return result
            self.model.common_inference = hooked_common_inference
    
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
            
            # Create prediction-confidence pairs with correct refiner instance mapping
            instance_confidence_pairs = []
            for i, conf in enumerate(confidence_scores):
                seq_id = pred_ids[i] if pred_ids and i < len(pred_ids) else i
                refiner_instance_idx = seq_id
                instance_confidence_pairs.append((conf, refiner_instance_idx, seq_id))
            
            # Sort by confidence (descending) and take top N
            instance_confidence_pairs.sort(key=lambda x: x[0], reverse=True)
            top_instances = instance_confidence_pairs[:self.top_n]
            
            # Filter attention maps for top instances
            for attn_map in all_attention_maps:
                attn_weights = attn_map['attention_weights']
                
                for rank, (conf, refiner_instance_idx, seq_id) in enumerate(top_instances):
                    if refiner_instance_idx < attn_weights.shape[0]:
                        filtered_weights = attn_weights[refiner_instance_idx].clone()
                        
                        filtered_attention_maps.append({
                            'layer': f"{attn_map['layer']}_top_{rank+1}",
                            'attention_weights': filtered_weights,
                            'shape': filtered_weights.shape,
                            'instance_info': {
                                'refiner_instance_id': refiner_instance_idx,
                                'sequence_id': seq_id,
                                'prediction_rank': rank,
                                'confidence_score': conf,
                                'rank': rank + 1
                            }
                        })
                        
                        top_instances_info.append({
                            'rank': rank + 1,
                            'refiner_instance_id': refiner_instance_idx,
                            'sequence_id': seq_id,
                            'prediction_rank': rank,
                            'confidence_score': conf
                        })
        else:
            # If no prediction info, save all attention maps (fallback)
            filtered_attention_maps = all_attention_maps
        
        # Store attention data for this video (only top prediction data)
        self.video_attention_data[video_id] = {
            'attention_maps': filtered_attention_maps,
            'num_maps': len(filtered_attention_maps),
            'top_instances_info': top_instances_info,
            'top_prediction_info': pred_info.get('top_prediction_info', None) if pred_info else None,
            'top_prediction_masks': pred_info.get('top_prediction_masks', None) if pred_info else None
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
                    # Shape should be [num_instances, num_frames, num_frames] for temporal attention
                    # We want the second dimension (num_frames)
                    if len(attn_weights.shape) == 3:
                        num_frames = attn_weights.shape[1]  # Second dimension is num_frames
                    elif len(attn_weights.shape) == 2:
                        num_frames = attn_weights.shape[0]  # For 2D case, first dimension is num_frames
                elif isinstance(attn_weights, list):
                    # If it's already converted to list, use the length of the first sublist
                    if attn_weights and isinstance(attn_weights[0], list):
                        num_frames = len(attn_weights[0])
                    else:
                        num_frames = len(attn_weights)
        
        # Always prepare individual attention maps
        individual_attention_maps = video_data['attention_maps']
        
        # Apply rollout if requested
        rolled_out_attention_maps = None
        if self.rollout:
            logger = logging.getLogger(__name__)
            logger.info("Applying attention rollout to extracted attention maps")
            rolled_out_attention_maps = self.apply_rollout_to_attention_maps(video_data['attention_maps'])
            logger.info(f"Rollout completed: {len(rolled_out_attention_maps)} rolled-out attention maps")
        
        # Prepare final save data with both individual and rolled-out maps
        save_data = {
            'video_id': video_id,
            'num_frames': num_frames,
            'attention_maps': individual_attention_maps,
            'num_maps': len(individual_attention_maps),
            'top_n': self.top_n,
            'top_instances_info': video_data['top_instances_info']
        }
        
        # Add rolled-out attention maps if rollout was performed
        if self.rollout and rolled_out_attention_maps is not None:
            save_data['rolled_out_attention_maps'] = rolled_out_attention_maps
            save_data['num_rolled_out_maps'] = len(rolled_out_attention_maps)
        
        # Add rollout information if rollout was performed
        if self.rollout:
            save_data['rollout_info'] = {
                'method': 'standard_attention_rollout',
                'skip_connection_simulation': True,
                'skip_factor': 1,
                'num_layers_combined': self.num_layers,
                'layer_order': 'B_6 × B_5 × B_4 × B_3 × B_2 × B_1'
            }
        
        
        # Save to file
        if self.rollout:
            if self.simulate_skip_connection:
                filename = f"attention_maps_video_{video_id}_top_{self.top_n}_rollout.json"
            else:
                filename = f"attention_maps_video_{video_id}_top_{self.top_n}_rollout_no_skip.json"
        else:
            filename = f"attention_maps_video_{video_id}_top_{self.top_n}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert tensors to lists for JSON serialization
        json_data = copy.deepcopy(save_data)
        for attn_map in json_data['attention_maps']:
            if isinstance(attn_map['attention_weights'], torch.Tensor):
                attn_map['attention_weights'] = attn_map['attention_weights'].tolist()
        
        # Convert rolled-out attention maps if they exist
        if 'rolled_out_attention_maps' in json_data:
            for attn_map in json_data['rolled_out_attention_maps']:
                if isinstance(attn_map['attention_weights'], torch.Tensor):
                    attn_map['attention_weights'] = attn_map['attention_weights'].tolist()
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger = logging.getLogger(__name__)
        if self.rollout:
            logger.info(f"Saved attention data for video {video_id} with {num_frames} frames to {filepath}")
            logger.info(f"Saved {len(individual_attention_maps)} individual layer maps and {len(rolled_out_attention_maps) if rolled_out_attention_maps else 0} rolled-out maps")
        else:
            logger.info(f"Saved attention data for video {video_id} with {num_frames} frames to {filepath}")
        logger.info(f"Top {self.top_n} instances: {video_data['top_instances_info']}")
        
        # Save top prediction separately
        self.save_top_prediction(video_id, video_data)
    
    def save_top_prediction(self, video_id, video_data):
        """Save the top-scoring prediction annotations in a separate JSON file."""
        logger = logging.getLogger(__name__)
        
        # Prepare top prediction data
        top_prediction_data = {
            'video_id': video_id,
            'top_prediction_info': video_data.get('top_prediction_info', None),
            'top_prediction_masks': video_data.get('top_prediction_masks', None),
            'top_instances_info': video_data.get('top_instances_info', [])
        }
        
        # Convert masks to RLE format for efficient storage
        if top_prediction_data['top_prediction_masks'] is not None:
            # Store original shape before conversion
            if isinstance(top_prediction_data['top_prediction_masks'], torch.Tensor):
                top_prediction_data['top_prediction_masks_shape'] = list(top_prediction_data['top_prediction_masks'].shape)
            else:
                top_prediction_data['top_prediction_masks_shape'] = 'unknown'
            
            # Convert to RLE format
            top_prediction_data['top_prediction_masks'] = convert_mask_to_rle(top_prediction_data['top_prediction_masks'])
            
            # Add RLE format information
            if PYCOCOTOOLS_AVAILABLE:
                top_prediction_data['mask_format'] = 'RLE'
            else:
                top_prediction_data['mask_format'] = 'raw_list'
        else:
            top_prediction_data['top_prediction_masks_shape'] = None
            top_prediction_data['mask_format'] = None
        
        # Save to separate file
        if self.rollout:
            if self.simulate_skip_connection:
                filename = f"top_prediction_video_{video_id}_top_{self.top_n}_rollout.json"
            else:
                filename = f"top_prediction_video_{video_id}_top_{self.top_n}_rollout_no_skip.json"
        else:
            filename = f"top_prediction_video_{video_id}_top_{self.top_n}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(top_prediction_data, f, indent=2)
        
        # Log prediction information
        if video_data.get('top_prediction_info'):
            top_pred = video_data['top_prediction_info']
            logger.info(f"Saved top prediction: class_id={top_pred.get('class_id')}, confidence={top_pred.get('confidence_score'):.4f}, sequence_id={top_pred.get('sequence_id')}")
            
            # Log mask information
            if video_data.get('top_prediction_masks') is not None:
                mask_shape = video_data['top_prediction_masks'].shape if hasattr(video_data['top_prediction_masks'], 'shape') else 'unknown'
                logger.info(f"Top prediction masks saved with shape: {mask_shape}")
            else:
                logger.info("No top prediction masks available")
        else:
            logger.info("No top prediction information available")
        
        logger.info(f"Top prediction saved to: {filepath}")
    
    def apply_skip_connection_simulation(self, attention_matrix):
        """
        Simulate skip connection by blending attention with identity matrix.
        
        Args:
            attention_matrix: Attention matrix of shape (num_frames, num_frames)
            
        Returns:
            Blended matrix: 1 * (attention_matrix + identity)
        """
        num_frames = attention_matrix.shape[0]
        identity = torch.eye(num_frames, device=attention_matrix.device, dtype=attention_matrix.dtype)
        return 1 * (attention_matrix + identity)
    
    def perform_attention_rollout(self, attention_maps):
        """
        Perform attention rollout on the 6 refiner layers.
        
        Args:
            attention_maps: List of attention maps, one per layer
            
        Returns:
            Dictionary containing the rolled-out attention map
        """
        if len(attention_maps) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} attention maps, got {len(attention_maps)}")
        
        # Convert attention maps to tensors
        processed_maps = []
        for i, attn_map in enumerate(attention_maps):
            attention_weights = attn_map['attention_weights']
            if self.simulate_skip_connection:
                processed_map = self.apply_skip_connection_simulation(attention_weights)
            else:
                processed_map = attention_weights
            processed_maps.append(processed_map)
        
        # Perform attention rollout: T = B_6 × B_5 × B_4 × B_3 × B_2 × B_1
        # (latest to earliest, newest on the left)
        rolled_out_attention = processed_maps[5]  # Start with layer 6 (index 5)
        for i in range(4, -1, -1):  # Go from layer 5 down to layer 1
            rolled_out_attention = torch.matmul(rolled_out_attention, processed_maps[i])
        
        # Determine rollout method description
        if self.simulate_skip_connection:
            rollout_method = 'standard_with_skip_simulation'
            skip_factor = 1
        else:
            rollout_method = 'pure_matrix_multiplication'
            skip_factor = None
        
        return {
            'layer': 'refiner_time_self_attn_rollout',
            'attention_weights': rolled_out_attention,
            'shape': list(rolled_out_attention.shape),
            'rollout_method': rollout_method,
            'skip_factor': skip_factor,
            'layer_order': 'B_6 × B_5 × B_4 × B_3 × B_2 × B_1'
        }
    
    def apply_rollout_to_attention_maps(self, attention_maps):
        """
        Apply attention rollout to the extracted attention maps.
        
        Args:
            attention_maps: List of attention maps from different layers
            
        Returns:
            List of rolled-out attention maps (one per instance)
        """
        # Group attention maps by instance
        instance_attention_maps = {}
        
        for attn_map in attention_maps:
            layer_name = attn_map['layer']
            instance_info = attn_map.get('instance_info', {})
            instance_id = instance_info.get('refiner_instance_id', 0)
            
            # Extract layer number from layer name (e.g., "refiner.transformer_time_self_attention_layers.0_top_1" -> 0)
            if 'transformer_time_self_attention_layers.' in layer_name:
                # Extract the number after the dot and before the underscore
                parts = layer_name.split('transformer_time_self_attention_layers.')[1]
                layer_num = int(parts.split('_')[0])
                
                if instance_id not in instance_attention_maps:
                    instance_attention_maps[instance_id] = {}
                
                instance_attention_maps[instance_id][layer_num] = attn_map
        
        # Perform rollout for each instance
        rolled_out_maps = []
        for instance_id, layer_maps in instance_attention_maps.items():
            if len(layer_maps) == self.num_layers:
                # Sort layers by number (0-5)
                sorted_layers = [layer_maps[i] for i in range(self.num_layers)]
                
                # Perform attention rollout
                rolled_out_map = self.perform_attention_rollout(sorted_layers)
                
                # Add instance information
                if sorted_layers[0].get('instance_info'):
                    rolled_out_map['instance_info'] = sorted_layers[0]['instance_info']
                
                rolled_out_maps.append(rolled_out_map)
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"Instance {instance_id} has {len(layer_maps)} layers, expected {self.num_layers}")
        
        return rolled_out_maps

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
    def test_with_attention(cls, cfg, model, target_video_id=None, output_dir=None, top_n=1, rollout=False, simulate_skip_connection=True):
        """
        Run evaluation with attention extraction on the test set.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting evaluation with attention extraction")
        if rollout:
            if simulate_skip_connection:
                logger.info("Attention rollout will be applied with skip connection simulation")
            else:
                logger.info("Attention rollout will be applied with pure matrix multiplication")
        
        # Create attention extractor
        attention_extractor = AttentionExtractor(model, output_dir, top_n=top_n, rollout=rollout, simulate_skip_connection=simulate_skip_connection)
        
        # Build data loader with correct mapper for video instance datasets
        dataset_name = cfg.DATASETS.TEST[0]
        dataset_type = cfg.DATASETS.DATASET_TYPE_TEST[0]
        logger.info(f"Dataset name: {dataset_name}")
        logger.info(f"Dataset type: {dataset_type}")
        
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
        
        logger.info(f"Data loader created with {len(data_loader)} batches")
        
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
                input_data = inputs[0] if isinstance(inputs, list) else inputs
                video_id = None
                if hasattr(input_data, 'get'):
                    video_id = input_data.get('video_id', None)
                elif hasattr(input_data, 'video_id'):
                    video_id = input_data.video_id
                
                if video_id is not None and video_id != target_video_id:
                    continue
            
            logger.info(f"Processing batch {idx}/{total}")
            
            # Clear previous attention data
            attention_extractor.clear_attention_data()
            
            # Run inference
            from torch.cuda.amp import autocast
            with torch.no_grad():
                with autocast():
                    outputs = model(inputs)
            
            # Extract prediction info
            video_id = target_video_id if target_video_id is not None else idx
            pred_logits = outputs.get("pred_logits", None)
            pred_masks = outputs.get("pred_masks", None)
            pred_ids = outputs.get("pred_ids", None)
            # Additional possible keys present in some DVIS variants
            scores_per_class = outputs.get("pred_scores_per_class", None) or outputs.get("class_logits", None) or outputs.get("logits", None)
            generic_scores = outputs.get("pred_scores", None) or outputs.get("scores", None)
            possible_class_keys = [
                "pred_classes",
                "pred_class",
                "pred_labels",
                "labels",
                "classes",
            ]
            
            # Calculate confidence scores and class predictions
            confidence_scores = None
            class_predictions = None
            top_prediction_info = None

            # Prefer final, post-processed outputs used by evaluator/results.json
            if "scores" in outputs:
                pred_scores = outputs["scores"]
                confidence_scores = pred_scores.cpu().numpy().tolist() if hasattr(pred_scores, 'cpu') else list(pred_scores)
                # Classes may be provided under different keys
                final_class_keys = ["category_id", "pred_classes", "pred_class", "labels", "classes"]
                for k in final_class_keys:
                    if k in outputs and outputs[k] is not None:
                        vals = outputs[k]
                        class_predictions = vals.cpu().numpy().tolist() if hasattr(vals, 'cpu') else list(vals)
                        break
                if len(confidence_scores) > 0:
                    top_idx = int(np.argmax(confidence_scores))
                    model_class_index = (class_predictions[top_idx] if isinstance(class_predictions, list) and len(class_predictions) > top_idx else class_predictions if isinstance(class_predictions, int) else None)
                    category_id, category_name = _map_model_index_to_category(model_class_index) if model_class_index is not None else (None, None)
                    seq_id = None
                    if pred_ids is not None:
                        try:
                            seq_id = pred_ids[top_idx].item() if hasattr(pred_ids[top_idx], 'item') else pred_ids[top_idx]
                        except Exception:
                            seq_id = None
                    if seq_id is None:
                        logger.warning("pred_ids missing or unavailable; sequence_id cannot be determined reliably for top prediction")
                    top_prediction_info = {
                        "class_id": model_class_index,
                        "category_id": category_id,
                        "category_name": category_name,
                        "confidence_score": confidence_scores[top_idx],
                        "prediction_rank": 0,
                        "sequence_id": seq_id
                    }
            elif pred_logits is not None:
                # Get class predictions and confidence scores
                class_probs = F.softmax(pred_logits, dim=-1)
                confidence_scores = class_probs.max(dim=-1)[0].cpu().numpy().tolist()
                class_predictions = class_probs.argmax(dim=-1).cpu().numpy().tolist()
                
                # Get top-scoring prediction info
                if len(confidence_scores) > 0:
                    top_idx = confidence_scores.index(max(confidence_scores))
                    model_class_index = class_predictions[top_idx]
                    category_id, category_name = _map_model_index_to_category(model_class_index)
                    seq_id = None
                    if pred_ids is not None:
                        try:
                            seq_id = pred_ids[top_idx].item() if hasattr(pred_ids[top_idx], 'item') else pred_ids[top_idx]
                        except Exception:
                            seq_id = None
                    if seq_id is None:
                        logger.warning("pred_ids missing or unavailable; sequence_id cannot be determined reliably for top prediction")
                    top_prediction_info = {
                        "class_id": model_class_index,
                        "category_id": category_id,
                        "category_name": category_name,
                        "confidence_score": confidence_scores[top_idx],
                        "prediction_rank": 0,
                        "sequence_id": seq_id
                    }
            elif scores_per_class is not None:
                # Scores/logits provided per class; derive class id and confidence
                tensor = scores_per_class
                if hasattr(tensor, 'softmax'):
                    try:
                        probs = F.softmax(tensor, dim=-1)
                        confidence_scores = probs.max(dim=-1)[0].cpu().numpy().tolist()
                        class_predictions = probs.argmax(dim=-1).cpu().numpy().tolist()
                    except Exception:
                        # Fall back to argmax/max directly (already probabilities)
                        confidence_scores = tensor.max(dim=-1)[0].cpu().numpy().tolist()
                        class_predictions = tensor.argmax(dim=-1).cpu().numpy().tolist()
                else:
                    # Assume list-like [num_inst, num_classes]
                    arr = np.array(tensor)
                    class_predictions = arr.argmax(axis=-1).tolist()
                    confidence_scores = arr.max(axis=-1).tolist()
                if len(confidence_scores) > 0:
                    top_idx = confidence_scores.index(max(confidence_scores))
                    model_class_index = class_predictions[top_idx]
                    category_id, category_name = _map_model_index_to_category(model_class_index)
                    seq_id = None
                    if pred_ids is not None:
                        try:
                            seq_id = pred_ids[top_idx].item() if hasattr(pred_ids[top_idx], 'item') else pred_ids[top_idx]
                        except Exception:
                            seq_id = None
                    if seq_id is None:
                        logger.warning("pred_ids missing or unavailable; sequence_id cannot be determined reliably for top prediction")
                    top_prediction_info = {
                        "class_id": model_class_index,
                        "category_id": category_id,
                        "category_name": category_name,
                        "confidence_score": confidence_scores[top_idx],
                        "prediction_rank": 0,
                        "sequence_id": seq_id
                    }
            elif generic_scores is not None:
                # Only per-instance scores present; attempt to fetch classes from any known key
                pred_scores = generic_scores
                confidence_scores = pred_scores.cpu().numpy().tolist() if hasattr(pred_scores, 'cpu') else pred_scores
                for k in possible_class_keys:
                    if k in outputs and outputs[k] is not None:
                        vals = outputs[k]
                        class_predictions = vals.cpu().numpy().tolist() if hasattr(vals, 'cpu') else vals
                        break
                if len(confidence_scores) > 0:
                    top_idx = confidence_scores.index(max(confidence_scores))
                    model_class_index = (class_predictions[top_idx] if isinstance(class_predictions, list) and len(class_predictions) > top_idx else class_predictions if isinstance(class_predictions, int) else None)
                    category_id, category_name = _map_model_index_to_category(model_class_index) if model_class_index is not None else (None, None)
                    seq_id = None
                    if pred_ids is not None:
                        try:
                            seq_id = pred_ids[top_idx].item() if hasattr(pred_ids[top_idx], 'item') else pred_ids[top_idx]
                        except Exception:
                            seq_id = None
                    if seq_id is None:
                        logger.warning("pred_ids missing or unavailable; sequence_id cannot be determined reliably for top prediction")
                    top_prediction_info = {
                        "class_id": model_class_index,
                        "category_id": category_id,
                        "category_name": category_name,
                        "confidence_score": confidence_scores[top_idx],
                        "prediction_rank": 0,
                        "sequence_id": seq_id
                    }
            
            # Extract mask data for top prediction
            top_prediction_masks = None
            if top_prediction_info and pred_masks is not None:
                top_idx = confidence_scores.index(max(confidence_scores)) if confidence_scores else 0
                if top_idx < len(pred_masks):
                    top_prediction_masks = pred_masks[top_idx]
                    # Convert to CPU and detach if it's a tensor
                    if hasattr(top_prediction_masks, 'cpu'):
                        top_prediction_masks = top_prediction_masks.cpu().detach()
            # Prefer using final segmentations if available for exact parity with results.json
            if top_prediction_info and "segmentations" in outputs:
                segs = outputs["segmentations"]
                top_idx = confidence_scores.index(max(confidence_scores)) if confidence_scores else 0
                if isinstance(segs, list) and top_idx < len(segs):
                    top_prediction_masks = segs[top_idx]
            
            pred_info = {
                "pred_logits": pred_logits,
                "pred_masks": pred_masks, 
                "pred_ids": pred_ids,
                "confidence_scores": confidence_scores,
                "class_predictions": class_predictions,
                "top_prediction_info": top_prediction_info,
                "top_prediction_masks": top_prediction_masks,
                "num_predictions": len(pred_logits) if pred_logits is not None else 0,
                "available_keys": list(outputs.keys())
            }
            
            # Accumulate attention data
            attention_extractor.accumulate_attention_data(video_id, pred_info=pred_info)
            
            # Log completion
            if pred_info.get('pred_ids') is not None:
                seq_ids = pred_info['pred_ids'].cpu().numpy().tolist() if hasattr(pred_info['pred_ids'], 'cpu') else pred_info['pred_ids']
                logger.info(f"Video {video_id}: Attention extraction completed successfully")
            
            # Break if we found our target video
            if target_video_id is not None:
                logger.info(f"Found and processed target video {target_video_id}, stopping inference")
                break
        
        # Finalize the current video's attention data
        attention_extractor.finalize_video()
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
    
    # Use config file from model directory
    model_config_file = os.path.join(MODEL_DIR, "config.yaml")
    if not os.path.exists(model_config_file):
        raise FileNotFoundError(f"Config file not found: {model_config_file}")
    
    cfg.merge_from_file(model_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="minvis")
    
    # Register validation dataset from model directory
    val_json_path = os.path.join(MODEL_DIR, "val.json")
    if not os.path.exists(val_json_path):
        raise FileNotFoundError(f"Validation JSON not found: {val_json_path}")
    
    # Determine video directory based on config (camera vs silhouette)
    # Check if the config contains camera or silhouette in the dataset paths
    with open(model_config_file, 'r') as f:
        config_content = f.read()
        if 'camera' in config_content.lower():
            video_dir = "/data/fishway_ytvis/all_videos"
        elif 'silhouette' in config_content.lower():
            video_dir = "/data/fishway_ytvis/all_videos_mask"
        else:
            # Default to camera if unclear
            video_dir = "/data/fishway_ytvis/all_videos"
    
    register_ytvis_instances(
        "ytvis_fishway_val_camera",
        {},
        val_json_path,
        video_dir
    )
    
    return cfg


def main(args):
    cfg = setup(args)
    
    model = Trainer.build_model(cfg)
    # Use model weights from model directory
    model_weights_path = os.path.join(MODEL_DIR, f"model_{CHECKPOINT_NUM}.pth")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")
    
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        model_weights_path, resume=args.resume
    )
    
    if args.eval_only:
        if hasattr(args, 'extract_attention') and args.extract_attention:
            # Extract attention maps
            target_video_id = getattr(args, 'target_video_id', None)
            output_dir = getattr(args, 'attention_output_dir', None)
            top_n = getattr(args, 'top_n', 1)
            rollout = getattr(args, 'rollout', False)
            simulate_skip_connection = not getattr(args, 'no_skip_connection', False)
            res = Trainer.test_with_attention(cfg, model, target_video_id, output_dir, top_n, rollout, simulate_skip_connection)
        else:
            # Regular evaluation
            res = Trainer.test(cfg, model)
        return res
    
    return Trainer.train(cfg, model, resume=args.resume)


if __name__ == "__main__":
    # Add custom arguments for attention extraction
    parser = default_argument_parser()
    # Override the config-file help text since we use the one from model directory
    parser._actions[1].help = "Config file (ignored, uses config from model directory)"
    parser.add_argument("--extract-attention", action="store_true", help="Extract attention maps during evaluation")
    parser.add_argument("--target-video-id", type=int, help="Target video ID to extract attention for")
    parser.add_argument("--attention-output-dir", type=str, help="Output directory for attention maps")
    parser.add_argument("--top-n", type=int, default=1, help="Number of top-scoring predictions to extract attention for (default: 1)")
    parser.add_argument("--rollout", action="store_true", help="Perform attention rollout on the extracted attention maps")
    parser.add_argument("--no-skip-connection", action="store_true", help="Disable skip connection simulation during rollout (use pure matrix multiplication)")
    
    args = parser.parse_args()
    
    # Print model configuration
    print(f"Using model directory: {MODEL_DIR}")
    print(f"Using checkpoint: {CHECKPOINT_NUM}")
    print(f"Model weights: {os.path.join(MODEL_DIR, f'model_{CHECKPOINT_NUM}.pth')}")
    print(f"Config file: {os.path.join(MODEL_DIR, 'config.yaml')}")
    print(f"Validation data: {os.path.join(MODEL_DIR, 'val.json')}")
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )