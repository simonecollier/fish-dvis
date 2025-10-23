#!/usr/bin/env python3
"""
DVIS-DAQ Attention Map Extraction Script

This script extracts ALL attention maps from a trained DVIS-DAQ model for video instance segmentation.
It covers attention maps from:
1. DINOv2 ViT-L Backbone
2. Multi-Scale Deformable Attention (Pixel Decoder)
3. Transformer Decoder (10 layers)
4. VideoInstanceCutter/Tracker (6 layers)
5. ReID Branch
6. Temporal Refiner (6 layers)

Author: Generated for DVIS-DAQ attention analysis
Date: 2024
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import gc

from datetime import datetime

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Config imports (same as train_net.py)
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from dvis_Plus import (
    add_minvis_config,
    add_dvis_config,
    add_ctvis_config,
)
from dvis_daq.config import add_daq_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionExtractor:
    """
    Comprehensive attention map extractor for DVIS-DAQ model
    """
    
    def __init__(self, model_path: str, output_dir: str = "/store/simone/attention/"):
        """
        Initialize the attention extractor
        
        Args:
            model_path: Path to the trained DVIS-DAQ model checkpoint
            output_dir: Directory to save attention maps (only used if save_attention=True)
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        
        # Store all attention maps or summaries
        self.attention_maps = {}
        self.attention_summaries = {}  # Store only shape/stats instead of full tensors
        
        # Memory management settings
        self.enable_memory_management = True
        self.memory_threshold_mb = 2000  # 2GB threshold for memory cleanup
        self.aggressive_memory_mode = True  # Enable aggressive memory optimization
        self.store_attention_maps = False  # By default, don't store full attention maps
        self.attention_summary_only = True  # Only store shape/stats summaries
        self.selective_storage = False  # Enable selective storage mode
        self.target_attention_keys = set()  # Keys to store when in selective mode
        
        # Logging system for capturing all output
        self.log_buffer = []
        self.enable_logging = True
        
        # Load the model
        self.model = self._load_model(model_path)
        
        # Register hooks for all attention modules
        self._register_attention_hooks()
        
        logger.info(f"Attention extractor initialized. Output directory: {self.output_dir}")
        logger.info(f"Memory management enabled: {self.enable_memory_management}")
    
    def log_print(self, *args, **kwargs):
        """
        Custom print function that logs everything to buffer and also prints to console
        """
        # Convert all arguments to string and join them
        message = ' '.join(str(arg) for arg in args)
        
        # Print to console
        print(message, **kwargs)
        
        # Log to buffer if logging is enabled
        if self.enable_logging:
            self.log_buffer.append(message)
    
    def get_log_content(self) -> List[str]:
        """Get all logged content"""
        return self.log_buffer.copy()
    
    def clear_log(self):
        """Clear the log buffer"""
        self.log_buffer.clear()
    
    def _process_attention_immediately(self, key: str, tensor: torch.Tensor) -> dict:
        """
        Process attention tensor immediately and return only summary statistics
        This avoids storing large tensors in memory
        
        Args:
            key: Attention map identifier
            tensor: Attention tensor
            
        Returns:
            Dictionary with summary statistics
        """
        if tensor is None:
            return None
            
        # Move to CPU for processing if needed
        if tensor.is_cuda:
            cpu_tensor = tensor.detach().cpu()
        else:
            cpu_tensor = tensor.detach()
        
        # Extract summary statistics
        summary = {
            'key': key,
            'shape': list(cpu_tensor.shape),
            'dtype': str(cpu_tensor.dtype),
            'min_value': float(cpu_tensor.min().item()),
            'max_value': float(cpu_tensor.max().item()),
            'mean_value': float(cpu_tensor.mean().item()),
            'std_value': float(cpu_tensor.std().item()),
            'device_original': str(tensor.device),
            'memory_usage_mb': cpu_tensor.numel() * cpu_tensor.element_size() / 1024**2
        }
        
        # Clean up the CPU tensor immediately
        del cpu_tensor
        
        return summary
    
    def enable_full_storage(self):
        """Enable full attention map storage (for visualization)"""
        self.store_attention_maps = True
        self.attention_summary_only = False
        logger.info("Full attention map storage enabled - WARNING: High memory usage!")
    
    def enable_summary_only(self):
        """Enable summary-only mode (default, memory efficient)"""
        self.store_attention_maps = False
        self.attention_summary_only = True
        logger.info("Summary-only mode enabled - memory efficient")
    
    def enable_selective_storage(self, target_keys: List[str]):
        """
        Enable selective storage - only store specific attention maps
        
        Args:
            target_keys: List of attention map keys to store (e.g., ['backbone_vit_layer_5'])
        """
        self.store_attention_maps = True
        self.attention_summary_only = False
        self.selective_storage = True
        self.target_attention_keys = set(target_keys)
        logger.info(f"Selective attention storage enabled for: {target_keys}")
        logger.info("This will save significant memory by only storing needed maps!")
    
    def _clear_gpu_memory(self, force: bool = False):
        """
        Aggressively clear GPU memory to prevent accumulation between windows
        
        Args:
            force: Force memory cleanup even if below threshold
        """
        if not torch.cuda.is_available():
            return
        
        # Get current GPU memory usage
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved_memory = torch.cuda.memory_reserved() / 1024**2  # MB
        
        logger.info(f"GPU Memory - Allocated: {current_memory:.1f}MB, Reserved: {reserved_memory:.1f}MB")
        
        # Clear if above threshold or forced
        if force or current_memory > self.memory_threshold_mb:
            logger.info("Aggressively clearing GPU memory...")
            
            # Clear attention maps from GPU memory
            for key in list(self.attention_maps.keys()):
                if self.attention_maps[key] is not None and hasattr(self.attention_maps[key], 'is_cuda') and self.attention_maps[key].is_cuda:
                    # Move to CPU if still on GPU
                    self.attention_maps[key] = self.attention_maps[key].cpu()
            
            # Clear PyTorch cache multiple times
            for _ in range(3):
            torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection multiple times
            for _ in range(2):
            gc.collect()
            
            # Additional CUDA memory management
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            # Check memory after cleanup
            new_memory = torch.cuda.memory_allocated() / 1024**2
            freed_memory = current_memory - new_memory
            logger.info(f"GPU Memory after aggressive cleanup: {new_memory:.1f}MB (freed {freed_memory:.1f}MB)")
            
            # If still using too much memory, warn
            if new_memory > self.memory_threshold_mb * 0.8:
                logger.warning(f"GPU memory still high after cleanup: {new_memory:.1f}MB")
                logger.warning("Consider reducing window size or using CPU-only mode")
    


    
    def _load_model(self, model_path: str):
        """
        Load the trained model and configuration
        """
        logger.info(f"Loading model from: {model_path}")
        
        # Load the model configuration
        config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
        if os.path.exists(config_path):
            # Load config from the same directory as the model
            cfg = get_cfg()
            
            # Apply all necessary config functions (same as train_net.py)
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
            add_maskformer2_video_config(cfg)
            add_minvis_config(cfg)
            add_dvis_config(cfg)
            add_ctvis_config(cfg)
            add_daq_config(cfg)
            
            # Merge the model's config file
            cfg.merge_from_file(config_path)
            
            # Set default values for missing keys (same as training)
            if not hasattr(cfg.DATASETS, 'DATASET_NEED_MAP'):
                cfg.DATASETS.DATASET_NEED_MAP = [False]
            if not hasattr(cfg.DATASETS, 'DATASET_RATIO'):
                cfg.DATASETS.DATASET_RATIO = [1.0]
            if not hasattr(cfg.DATASETS, 'DATASET_TYPE'):
                cfg.DATASETS.DATASET_TYPE = ['video_instance']
            if not hasattr(cfg.DATASETS, 'DATASET_TYPE_TEST'):
                cfg.DATASETS.DATASET_TYPE_TEST = ['video_instance']
            
            cfg.freeze()
            
            # Build the model using the config
            model = build_model(cfg)
            
            # Load the trained weights
            DetectionCheckpointer(model).resume_or_load(model_path, resume=False)
            
            # Store the config for later use
            model.cfg = cfg
            
        else:
            # Fallback: load model directly (less ideal)
            logger.warning(f"Config file not found at {config_path}, loading model directly")
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
        
        # Move model to GPU if available (required for deformable attention operations)
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model moved to GPU")
        else:
            logger.warning("CUDA not available, model will run on CPU (may cause errors)")
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    def get_config_parameter(self, param_path: str, default_value=None):
        """
        Dynamically get configuration parameters using dot notation
        Example: get_config_parameter('MODEL.MASK_FORMER.TEST.WINDOW_SIZE', 31)
        """
        if not hasattr(self.model, 'cfg'):
            logger.warning("Model config not available, using default value")
            return default_value
        
        try:
            # Navigate the config using dot notation
            value = self.model.cfg
            for key in param_path.split('.'):
                value = getattr(value, key)
            return value
        except AttributeError:
            logger.warning(f"Config parameter {param_path} not found, using default: {default_value}")
            return default_value
    
    def _register_attention_hooks(self):
        """
        Register forward hooks for ALL attention modules in the model
        """
        logger.info("Registering attention hooks...")
        
        # 1. DINOv2 ViT-L Backbone Attention
        # Location: fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/meta_architecture.py -> model.backbone
        self._register_backbone_hooks()
        
        # 2. Multi-Scale Deformable Attention (Pixel Decoder)
        # Location: fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/meta_architecture.py -> model.sem_seg_head.pixel_decoder
        self._register_pixel_decoder_hooks()
        
        # 3. Transformer Decoder Attention
        # Location: fish-dvis/DVIS_Plus/DVIS_DAQ/mask2former_video/modeling/transformer_decoder/video_mask2former_transformer_decoder.py
        self._register_decoder_hooks()
        
        # 4. VideoInstanceCutter/Tracker Attention
        # Location: fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/track_module.py -> model.tracker
        self._register_tracker_hooks()
        
        # 5. ReID Branch Attention
        # Location: fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/meta_architecture.py -> model.sem_seg_head.predictor.reid_branch
        self._register_reid_hooks()
        
        # 6. Temporal Refiner Attention
        # Location: fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/refiner.py -> model.refiner
        self._register_refiner_hooks()
        
        logger.info("All attention hooks registered")
    
    def _register_backbone_hooks(self):
        """
        Register hooks for DINOv2 ViT-L backbone attention
        """
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'vit_module'):
            vit_module = self.model.backbone.vit_module
            
            # Hook into the ViT blocks directly
            if hasattr(vit_module, 'blocks'):
                for i, block in enumerate(vit_module.blocks):
                    if hasattr(block, 'attn'):
                        # Use a proper closure to avoid lambda late binding issues
                        def make_backbone_hook(layer_idx):
                            def hook(module, input, output):
                                # The Attention module doesn't return attention weights
                                # We need to compute them manually from the input
                                try:
                                    x = input[0]  # input tensor
                                    B, N, C = x.shape
                                    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                                    q, k, v = qkv[0] * module.scale, qkv[1], qkv[2]
                                    attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
                                    self._store_backbone_attention(layer_idx, (None, attn))
                                except Exception as e:
                                    logger.warning(f"Could not extract backbone attention weights for layer {layer_idx}: {e}")
                            return hook
                        
                        block.attn.register_forward_hook(make_backbone_hook(i))
                        logger.info(f"Registered backbone ViT block attention hook for layer {i}")
            
            # Also hook into the interaction blocks which process the ViT outputs
            if hasattr(self.model.backbone, 'interactions'):
                for i, interaction in enumerate(self.model.backbone.interactions):
                    if hasattr(interaction, 'extractor') and hasattr(interaction.extractor, 'attn'):
                        # Use a proper closure to avoid lambda late binding issues
                        def make_interaction_hook(layer_idx):
                            def hook(module, input, output):
                                self._store_backbone_interaction_attention(layer_idx, output)
                            return hook
                        
                        interaction.extractor.attn.register_forward_hook(make_interaction_hook(i))
                        logger.info(f"Registered backbone interaction attention hook for layer {i}")
                    
                    if hasattr(interaction, 'injector') and hasattr(interaction.injector, 'attn'):
                        # Use a proper closure to avoid lambda late binding issues
                        def make_injector_hook(layer_idx):
                            def hook(module, input, output):
                                self._store_backbone_injector_attention(layer_idx, output)
                            return hook
                        
                        interaction.injector.attn.register_forward_hook(make_injector_hook(i))
                        logger.info(f"Registered backbone injector attention hook for layer {i}")
    
    def _register_pixel_decoder_hooks(self):
        """
        Register hooks for multi-scale deformable attention in pixel decoder
        """
        if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'pixel_decoder'):
            pixel_decoder = self.model.sem_seg_head.pixel_decoder
            logger.info(f"Found pixel decoder: {type(pixel_decoder)}")
            logger.info(f"Pixel decoder attributes: {[attr for attr in dir(pixel_decoder) if not attr.startswith('_')]}")
            
            # Hook into the transformer (MSDeformAttnTransformerEncoderOnly)
            if hasattr(pixel_decoder, 'transformer'):
                logger.info(f"Found transformer: {type(pixel_decoder.transformer)}")
                transformer = pixel_decoder.transformer
                
                # Hook into the encoder layers
                if hasattr(transformer, 'encoder'):
                    logger.info(f"Found encoder: {type(transformer.encoder)}")
                    if hasattr(transformer.encoder, 'layers'):
                        logger.info(f"Found encoder layers with {len(transformer.encoder.layers)} layers")
                        for i, layer in enumerate(transformer.encoder.layers):
                            logger.info(f"Encoder layer {i} attributes: {[attr for attr in dir(layer) if not attr.startswith('_')]}")
                            
                            # Hook into the self-attention (MSDeformAttn)
                            if hasattr(layer, 'self_attn'):
                                logger.info(f"Found self_attn in encoder layer {i}: {type(layer.self_attn)}")
                                # Use a proper closure to avoid lambda late binding issues
                                def make_pixel_decoder_hook(layer_idx):
                                    def hook(module, input, output):
                                        self._store_pixel_decoder_attention(layer_idx, input, output)
                                    return hook
                                
                                layer.self_attn.register_forward_hook(make_pixel_decoder_hook(i))
                                logger.info(f"Registered pixel decoder self-attention hook for encoder layer {i}")
                            else:
                                logger.warning(f"No self_attn found in encoder layer {i}")
                            
                            # Hook into cross-attention if it exists
                            if hasattr(layer, 'cross_attn'):
                                logger.info(f"Found cross_attn in encoder layer {i}: {type(layer.cross_attn)}")
                                # Use a proper closure to avoid lambda late binding issues
                                def make_pixel_decoder_cross_hook(layer_idx):
                                    def hook(module, input, output):
                                        self._store_pixel_decoder_cross_attention(layer_idx, output)
                                    return hook
                                
                                layer.cross_attn.register_forward_hook(make_pixel_decoder_cross_hook(i))
                                logger.info(f"Registered pixel decoder cross-attention hook for encoder layer {i}")
                    else:
                        logger.warning("No layers found in encoder")
                else:
                    logger.warning("No encoder found in transformer")
            else:
                logger.warning("No transformer found in pixel decoder")
        else:
            logger.warning("No pixel decoder found in sem_seg_head")
    
    def _register_decoder_hooks(self):
        """
        Register hooks for transformer decoder attention (10 layers)
        """
        if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'predictor'):
            predictor = self.model.sem_seg_head.predictor
            
            # Self-attention layers
            if hasattr(predictor, 'transformer_self_attention_layers'):
                for i, layer in enumerate(predictor.transformer_self_attention_layers):
                    layer.self_attn.register_forward_hook(
                        lambda module, input, output, layer_idx=i: 
                        self._store_decoder_self_attention(layer_idx, output)
                    )
                    logger.info(f"Registered decoder self-attention hook for layer {i}")
            
            # Cross-attention layers
            if hasattr(predictor, 'transformer_cross_attention_layers'):
                for i, layer in enumerate(predictor.transformer_cross_attention_layers):
                    layer.multihead_attn.register_forward_hook(
                        lambda module, input, output, layer_idx=i: 
                        self._store_decoder_cross_attention(layer_idx, output)
                    )
                    logger.info(f"Registered decoder cross-attention hook for layer {i}")
    
    def _register_tracker_hooks(self):
        """
        Register hooks for VideoInstanceCutter attention (6 layers)
        """
        if hasattr(self.model, 'tracker'):
            tracker = self.model.tracker
            
            # Self-attention layers
            if hasattr(tracker, 'transformer_self_attention_layers'):
                for i, layer in enumerate(tracker.transformer_self_attention_layers):
                    layer.self_attn.register_forward_hook(
                        lambda module, input, output, layer_idx=i: 
                        self._store_tracker_self_attention(layer_idx, output)
                    )
                    logger.info(f"Registered tracker self-attention hook for layer {i}")
            
            # Cross-attention layers
            if hasattr(tracker, 'transformer_cross_attention_layers'):
                for i, layer in enumerate(tracker.transformer_cross_attention_layers):
                    layer.multihead_attn.register_forward_hook(
                        lambda module, input, output, layer_idx=i: 
                        self._store_tracker_cross_attention(layer_idx, output)
                    )
                    logger.info(f"Registered tracker cross-attention hook for layer {i}")
            
            # Slot cross-attention layers
            if hasattr(tracker, 'slot_cross_attention_layers'):
                for i, layer in enumerate(tracker.slot_cross_attention_layers):
                    # Hook into the slot attention mechanism
                    if hasattr(layer, 'slot_attn'):
                        layer.slot_attn.register_forward_hook(
                            lambda module, input, output, layer_idx=i: 
                            self._store_slot_attention(layer_idx, input, output)
                        )
                        logger.info(f"Registered slot attention hook for layer {i}")
    
    def _register_reid_hooks(self):
        """
        Register hooks for ReID branch attention
        """
        logger.info("Checking for ReID branch...")
        
        # Check multiple possible locations for ReID branch
        possible_locations = [
            ('sem_seg_head.predictor.reid_branch', 
             lambda m: m.sem_seg_head.predictor.reid_branch if hasattr(m, 'sem_seg_head') and hasattr(m.sem_seg_head, 'predictor') else None),
            ('sem_seg_head.reid_branch', 
             lambda m: m.sem_seg_head.reid_branch if hasattr(m, 'sem_seg_head') else None),
            ('predictor.reid_branch', 
             lambda m: m.predictor.reid_branch if hasattr(m, 'predictor') else None),
            ('reid_branch', 
             lambda m: m.reid_branch if hasattr(m, 'reid_branch') else None)
        ]
        
        reid_branch = None
        reid_location = None
        
        for location_name, getter_func in possible_locations:
            try:
                potential_reid = getter_func(self.model)
                if potential_reid is not None:
                    logger.info(f"Found ReID branch at: {location_name}")
                    reid_branch = potential_reid
                    reid_location = location_name
                    break
            except Exception as e:
                logger.debug(f"Error checking {location_name}: {e}")
        
        if reid_branch is None:
            logger.warning("No ReID branch found in any expected location")
            logger.info("Available model attributes:")
            if hasattr(self.model, 'sem_seg_head'):
                logger.info(f"  sem_seg_head attributes: {[attr for attr in dir(self.model.sem_seg_head) if not attr.startswith('_')]}")
                if hasattr(self.model.sem_seg_head, 'predictor'):
                    logger.info(f"  predictor attributes: {[attr for attr in dir(self.model.sem_seg_head.predictor) if not attr.startswith('_')]}")
                    
                    # Check if ReID is integrated into the transformer layers
                    predictor = self.model.sem_seg_head.predictor
                    if hasattr(predictor, 'reid_embed'):
                        logger.info(f"Found reid_embed: {type(predictor.reid_embed)}")
                        logger.info(f"reid_embed attributes: {[attr for attr in dir(predictor.reid_embed) if not attr.startswith('_')]}")
                        
                        # The ReID attention might be integrated into the transformer layers
                        # Let's check if there are separate ReID attention layers
                        if hasattr(predictor, 'reid_attention_layers'):
                            logger.info(f"Found reid_attention_layers with {len(predictor.reid_attention_layers)} layers")
                            for i, layer in enumerate(predictor.reid_attention_layers):
                                if hasattr(layer, 'attention'):
                                    def make_reid_hook(layer_idx):
                                        def hook(module, input, output):
                                            self._store_reid_attention(layer_idx, output)
                                        return hook
                                    
                                    layer.attention.register_forward_hook(make_reid_hook(i))
                                    logger.info(f"Registered ReID attention hook for layer {i}")
                        else:
                            logger.info("No separate reid_attention_layers found - ReID might be integrated into main transformer")
            return
        
        logger.info(f"ReID branch type: {type(reid_branch)}")
        logger.info(f"ReID branch attributes: {[attr for attr in dir(reid_branch) if not attr.startswith('_')]}")
        
        # Hook into ReID layers
        if hasattr(reid_branch, 'layers'):
            logger.info(f"Found ReID layers with {len(reid_branch.layers)} layers")
            for i, layer in enumerate(reid_branch.layers):
                logger.info(f"ReID layer {i} attributes: {[attr for attr in dir(layer) if not attr.startswith('_')]}")
                if hasattr(layer, 'attention'):
                    logger.info(f"Found attention in ReID layer {i}: {type(layer.attention)}")
                    # Use a proper closure to avoid lambda late binding issues
                    def make_reid_hook(layer_idx):
                        def hook(module, input, output):
                            self._store_reid_attention(layer_idx, output)
                        return hook
                    
                    layer.attention.register_forward_hook(make_reid_hook(i))
                    logger.info(f"Registered ReID attention hook for layer {i}")
                else:
                    logger.warning(f"No attention found in ReID layer {i}")
        else:
            logger.warning("No layers found in ReID branch")
    
    def _register_refiner_hooks(self):
        """
        Register hooks for Temporal Refiner attention (6 layers)
        """
        if hasattr(self.model, 'refiner'):
            refiner = self.model.refiner
            
            # Long temporal self-attention layers
            if hasattr(refiner, 'transformer_time_self_attention_layers'):
                for i, layer in enumerate(refiner.transformer_time_self_attention_layers):
                    layer.self_attn.register_forward_hook(
                        lambda module, input, output, layer_idx=i: 
                        self._store_refiner_long_temp_attention(layer_idx, output)
                    )
                    logger.info(f"Registered refiner long temporal attention hook for layer {i}")
            
            # Object self-attention layers
            if hasattr(refiner, 'transformer_obj_self_attention_layers'):
                for i, layer in enumerate(refiner.transformer_obj_self_attention_layers):
                    layer.self_attn.register_forward_hook(
                        lambda module, input, output, layer_idx=i: 
                        self._store_refiner_obj_attention(layer_idx, output)
                    )
                    logger.info(f"Registered refiner object attention hook for layer {i}")
            
            # Cross-attention layers
            if hasattr(refiner, 'transformer_cross_attention_layers'):
                for i, layer in enumerate(refiner.transformer_cross_attention_layers):
                    layer.multihead_attn.register_forward_hook(
                        lambda module, input, output, layer_idx=i: 
                        self._store_refiner_cross_attention(layer_idx, output)
                    )
                    logger.info(f"Registered refiner cross-attention hook for layer {i}")
    
    # Attention storage methods
    def _store_backbone_attention(self, layer_idx: int, output):
        """Store backbone ViT block attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            key = f'backbone_vit_layer_{layer_idx}'
            attn_tensor = output[1]
            
            # Check if we should store this specific attention map
            should_store_full = False
            if self.selective_storage:
                should_store_full = key in self.target_attention_keys
            elif not self.attention_summary_only:
                should_store_full = True
            
            if should_store_full:
                # Store full tensor for this specific key
                if self.aggressive_memory_mode and attn_tensor.is_cuda:
                    attn_tensor = attn_tensor.detach().cpu()
                else:
                    attn_tensor = attn_tensor.detach()
                self.attention_maps[key] = attn_tensor
                logger.debug(f"Stored full attention map: {key}")
            else:
                # Process immediately and store only summary
                summary = self._process_attention_immediately(key, attn_tensor)
                if summary:
                    self.attention_summaries[key] = summary
    
    def _store_backbone_interaction_attention(self, layer_idx: int, output):
        """Store backbone interaction attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'backbone_interaction_layer_{layer_idx}'] = output[1].detach()
    
    def _store_backbone_injector_attention(self, layer_idx: int, output):
        """Store backbone injector attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'backbone_injector_layer_{layer_idx}'] = output[1].detach()
    
    def _store_pixel_decoder_attention(self, layer_idx: int, input, output):
        """Store pixel decoder self-attention maps from MSDeformAttn"""
        try:
            # MSDeformAttn doesn't expose attention weights directly
            # We'll store the output features and input for later analysis
            if len(input) >= 2:
                # Store the input query and reference points for analysis
                query = input[0]  # query tensor
                reference_points = input[1] if len(input) > 1 else None
                
                # Store the attention module output (deformed features)
                if isinstance(output, torch.Tensor):
                    self.attention_maps[f'pixel_decoder_self_attn_layer_{layer_idx}'] = output.detach()
                    logger.info(f"Captured pixel decoder output for layer {layer_idx}: {output.shape}")
                    
                    # Also store input tensors for potential manual attention computation
                    self.attention_maps[f'pixel_decoder_query_layer_{layer_idx}'] = query.detach()
                    if reference_points is not None:
                        self.attention_maps[f'pixel_decoder_ref_points_layer_{layer_idx}'] = reference_points.detach()
                        
        except Exception as e:
            logger.warning(f"Could not extract pixel decoder attention weights for layer {layer_idx}: {e}")
            # Fallback: store the output features if attention weights can't be extracted
            if isinstance(output, torch.Tensor):
                self.attention_maps[f'pixel_decoder_features_layer_{layer_idx}'] = output.detach()
    
    def _store_pixel_decoder_cross_attention(self, layer_idx: int, output):
        """Store pixel decoder cross-attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'pixel_decoder_cross_attn_layer_{layer_idx}'] = output[1].detach()
    
    def _store_decoder_self_attention(self, layer_idx: int, output):
        """Store decoder self-attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'decoder_self_attn_layer_{layer_idx}'] = output[1].detach()
    
    def _store_decoder_cross_attention(self, layer_idx: int, output):
        """Store decoder cross-attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'decoder_cross_attn_layer_{layer_idx}'] = output[1].detach()
    
    def _store_tracker_self_attention(self, layer_idx: int, output):
        """Store tracker self-attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'tracker_self_attn_layer_{layer_idx}'] = output[1].detach()
    
    def _store_tracker_cross_attention(self, layer_idx: int, output):
        """Store tracker cross-attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'tracker_cross_attn_layer_{layer_idx}'] = output[1].detach()
    
    def _store_slot_attention(self, layer_idx: int, input, output):
        """Store slot attention maps"""
        # Slot attention returns attention weights directly
        self.attention_maps[f'slot_attention_layer_{layer_idx}'] = output.detach()
    
    def _store_reid_attention(self, layer_idx: int, output):
        """Store ReID attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'reid_attention_layer_{layer_idx}'] = output[1].detach()
    
    def _store_refiner_long_temp_attention(self, layer_idx: int, output):
        """Store refiner long temporal attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'refiner_long_temp_attn_layer_{layer_idx}'] = output[1].detach()
    
    def _store_refiner_obj_attention(self, layer_idx: int, output):
        """Store refiner object attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'refiner_obj_attention_layer_{layer_idx}'] = output[1].detach()
    
    def _store_refiner_cross_attention(self, layer_idx: int, output):
        """Store refiner cross-attention maps"""
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps[f'refiner_cross_attn_layer_{layer_idx}'] = output[1].detach()
    
    def extract_attention_for_video(self, video_id: str, save_attention: bool = True, print_shapes: bool = True):
        """
        Extract attention maps for a video using proper window inference
        
        Args:
            video_id: Video ID from the dataset
            save_attention: Whether to save attention maps to disk
            print_shapes: Whether to print attention map shapes
        """
        logger.info(f"Extracting attention maps for video ID: {video_id}")
        
        # Load video data using the same method as training
        video_input = self._load_video_from_dataset(video_id)
        
        # Get video length
        num_frames = len(video_input['image'])
        logger.info(f"Video has {num_frames} frames")
        
        # Get window inference parameters from config (not hardcoded)
        window_inference = self.get_config_parameter('MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE', False)
        window_size = self.get_config_parameter('MODEL.MASK_FORMER.TEST.WINDOW_SIZE', 31)
        
        if not window_inference:
            logger.warning("Window inference not enabled in config, using full video processing")
            # Fall back to full video processing
            return self._extract_attention_full_video(video_input, save_attention, print_shapes)
        
        logger.info(f"Using window inference with window size: {window_size} (from config)")
        logger.info(f"Processing video with {num_frames} frames using proper window-based attention extraction")
        
        # Clear previous attention maps
        self.attention_maps.clear()
        
        # Initial memory cleanup to start fresh
        self._clear_gpu_memory(force=True)
        
        # Configure the model for window inference using the same approach as training
        with torch.no_grad():
            try:
                # Set window inference mode on the model (same as in training config)
                if hasattr(self.model, 'window_inference'):
                    self.model.window_inference = True
                if hasattr(self.model, 'window_size'):
                    self.model.window_size = window_size
                
                # Set keep attribute if available (used in some DVIS implementations)
                if not hasattr(self.model, 'keep'):
                    self.model.keep = False
                
                # Convert video input to the format expected by the model
                # The method expects images_tensor: [T, C, H, W] format
                if 'image' in video_input and isinstance(video_input['image'], list):
                    # Convert list of tensors to stacked tensor
                    # The mapper produces a list of tensors, each with shape [C, H, W]
                    images_tensor = torch.stack(video_input['image'], dim=0)  # [T, C, H, W]
                    logger.info(f"Converted video input to tensor with shape: {images_tensor.shape}")
                    
                    # Verify the tensor format is correct
                    if len(images_tensor.shape) == 4 and images_tensor.shape[1] == 3:  # [T, C, H, W]
                        logger.info(f"Tensor format verified: {images_tensor.shape[0]} frames, {images_tensor.shape[1]} channels, {images_tensor.shape[2]}x{images_tensor.shape[3]} resolution")
                    else:
                        logger.warning(f"Unexpected tensor shape: {images_tensor.shape}, expected [T, 3, H, W]")
                        
                else:
                    logger.error("Video input format not compatible with window inference")
                    logger.error(f"Expected 'image' key with list of tensors, got: {type(video_input.get('image', 'missing'))}")
                    return self._extract_attention_full_video(video_input, save_attention, print_shapes)
                
                # Process the video using the complete DVIS-DAQ pipeline to capture ALL attention maps
                # This follows the same pipeline as run_window_inference but captures attention maps
                window_results = self._extract_attention_complete_pipeline(images_tensor, window_size)
                
                logger.info("Window-based attention extraction completed successfully")
                
                # Check if attention maps were captured
                num_attention_maps = len(self.attention_maps)
                logger.info(f"Captured {num_attention_maps} attention maps during window inference")
                
                if num_attention_maps == 0:
                    logger.warning("No attention maps captured. This might indicate an issue with the attention hooks.")
                else:
                    logger.info("Attention maps captured successfully during window inference")
                
                return window_results
                    
            except Exception as e:
                logger.error(f"Error during window inference: {e}")
                logger.info("Falling back to full video processing")
                return self._extract_attention_full_video(video_input, save_attention, print_shapes)
        
        # Print shapes if requested
        if print_shapes:
            self._print_attention_shapes()
        
        # Save attention maps if requested
        if save_attention:
            self._save_attention_maps(str(video_input.get('video_id', 'unknown')))
        
        # Return window results
        return window_results
    
    def _extract_attention_windows(self, images_tensor: torch.Tensor, window_size: int):
        """
        Extract attention maps using window-based processing that matches DVIS-DAQ's approach
        
        Args:
            images_tensor: Video tensor [T, C, H, W]
            window_size: Size of each window
            
        Returns:
            List of window results with attention maps
        """
        num_frames = images_tensor.shape[0]
        logger.info(f"Processing {num_frames} frames with window size {window_size}")
        
        # Calculate number of windows needed
        num_windows = (num_frames + window_size - 1) // window_size
        logger.info(f"Will process {num_windows} windows")
        
        window_results = []
        all_window_attention_maps = {}
        
        for window_idx in range(num_windows):
            start_frame = window_idx * window_size
            end_frame = min((window_idx + 1) * window_size, num_frames)
            actual_window_size = end_frame - start_frame
            
            logger.info(f"Processing window {window_idx + 1}/{num_windows}: frames {start_frame}-{end_frame} (size: {actual_window_size})")
            
            # Clear attention maps for this window to start fresh
            self.attention_maps.clear()
            
            # Extract frames for this window
            window_frames = images_tensor[start_frame:end_frame]
            
            # Process this window through the model's segmenter
            # This mimics what happens in segmenter_windows_inference
            try:
                # Move frames to GPU if available and convert to proper dtype
                if torch.cuda.is_available():
                    window_frames = window_frames.cuda()
                
                # Convert from uint8 to float32 and normalize to [0, 1] range if needed
                if window_frames.dtype == torch.uint8:
                    window_frames = window_frames.float() / 255.0
                
                # Process through backbone and sem_seg_head (same as segmenter_windows_inference)
                features = self.model.backbone(window_frames)
                out = self.model.sem_seg_head(features)
                
                # Clean up intermediate features to save memory
                if 'res2' in features:
                    del features['res2']
                if 'res3' in features:
                    del features['res3']
                if 'res4' in features:
                    del features['res4']
                if 'res5' in features:
                    del features['res5']
                
                # Clean up auxiliary outputs to save memory
                if 'aux_outputs' in out:
                    for j in range(len(out['aux_outputs'])):
                        if 'pred_masks' in out['aux_outputs'][j]:
                            del out['aux_outputs'][j]['pred_masks']
                        if 'pred_logits' in out['aux_outputs'][j]:
                            del out['aux_outputs'][j]['pred_logits']
                
                # Capture attention maps from this window and store with window-specific keys
                window_attention_maps = {}
                for key, tensor in self.attention_maps.items():
                    if tensor is not None:
                        # Add window information to the key
                        window_key = f"window_{window_idx}_{key}"
                        window_attention_maps[window_key] = tensor.detach().clone()
                        all_window_attention_maps[window_key] = tensor.detach().clone()
                
                logger.info(f"Window {window_idx + 1} captured {len(window_attention_maps)} attention maps")
                
                # Debug: Log the types of attention maps captured in this window
                attention_types_in_window = set()
                for key in window_attention_maps.keys():
                    if 'backbone_vit' in key:
                        attention_types_in_window.add('backbone_vit')
                    elif 'pixel_decoder' in key:
                        attention_types_in_window.add('pixel_decoder')
                    elif 'decoder' in key:
                        attention_types_in_window.add('decoder')
                    elif 'tracker' in key:
                        attention_types_in_window.add('tracker')
                    elif 'refiner' in key:
                        attention_types_in_window.add('refiner')
                logger.info(f"Window {window_idx + 1} attention types: {sorted(attention_types_in_window)}")
                
                # Store window result
                window_results.append({
                    'window_idx': window_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'actual_window_size': actual_window_size,
                    'attention_maps': window_attention_maps,
                    'num_attention_maps': len(window_attention_maps)
                })
                
                # Clean up output tensors
                del out
                
            except Exception as e:
                logger.error(f"Error processing window {window_idx}: {e}")
                # Still add the window to results but with empty attention maps
                window_results.append({
                    'window_idx': window_idx,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'actual_window_size': actual_window_size,
                    'attention_maps': {},
                    'num_attention_maps': 0
                })
            
            # Memory cleanup after each window
            if self.enable_memory_management:
                self._clear_gpu_memory()
        
        # Set the main attention_maps to contain all collected window maps
        self.attention_maps = all_window_attention_maps
        
        logger.info(f"Completed processing {num_windows} windows")
        logger.info(f"Total attention maps collected: {len(all_window_attention_maps)}")
        
        # Log window size verification
        for key, tensor in all_window_attention_maps.items():
            if tensor is not None and 'window_0_' in key:  # Check first window as example
                logger.info(f"Sample attention map '{key}' shape: {tensor.shape}")
                # Check if any dimension matches the window size
                if window_size in tensor.shape:
                    logger.info(f"✓ Window size {window_size} found in dimensions of '{key}'")
        
        return window_results
    
    def _extract_attention_complete_pipeline(self, images_tensor: torch.Tensor, window_size: int):
        """
        Memory-optimized attention extraction using the complete DVIS-DAQ pipeline
        
        This follows the same steps as run_window_inference but with aggressive memory management:
        1. Process windows individually with immediate CPU offloading
        2. Only store essential data for refiner processing
        3. Clear GPU memory aggressively between operations
        
        Args:
            images_tensor: Video tensor [T, C, H, W]
            window_size: Size of each window
            
        Returns:
            List of window results with attention maps
        """
        num_frames = images_tensor.shape[0]
        logger.info(f"Processing {num_frames} frames using memory-optimized DVIS-DAQ pipeline with window size {window_size}")
        
        # Clear previous attention maps
        self.attention_maps.clear()
        
        # Store attention maps from all windows (moved to CPU immediately)
        self.window_attention_maps = {}  # window_idx -> attention_maps
        
        # Memory optimization: Store only essential outputs for refiner
        all_frame_embeds = []
        all_mask_features = []
        all_instance_embeds = []
        all_padding_masks = []
        
        try:
            # We need to process each window individually to capture per-window attention maps
            num_windows = (num_frames + window_size - 1) // window_size
            logger.info(f"Processing {num_windows} windows with aggressive memory optimization")
            
            for window_idx in range(num_windows):
                start_frame = window_idx * window_size
                end_frame = min((window_idx + 1) * window_size, num_frames)
                actual_window_size = end_frame - start_frame
                
                logger.info(f"Processing window {window_idx}: frames {start_frame}-{end_frame} (size: {actual_window_size})")
                
                # Clear everything before processing this window
                self.attention_maps.clear()
                torch.cuda.empty_cache()
                
                # Extract frames for this window
                window_frames = images_tensor[start_frame:end_frame]
                
                # Make sure images are on GPU and convert to proper dtype
                if torch.cuda.is_available():
                    window_frames = window_frames.cuda()
                
                # Convert from uint8 to float32 and normalize to [0, 1] range if needed
                if window_frames.dtype == torch.uint8:
                    window_frames = window_frames.float() / 255.0
                
                # Process this window through the pipeline
                with torch.cuda.amp.autocast():  # Use mixed precision to save memory
                    common_out = self.model.common_inference(window_frames, window_size=actual_window_size, long_video_start_fidx=-1, to_store="cpu")
                
                # Process attention maps immediately - either store summaries or full tensors
                window_attention_data = {}
                total_memory_saved = 0
                
                # Process attention maps based on storage mode
                for key, tensor in self.attention_maps.items():
                    if tensor is not None:
                        window_key = f"window_{window_idx}_{key}"
                        
                        # Check if this is a targeted key for selective storage
                        should_store_full = False
                        if self.selective_storage:
                            should_store_full = key in self.target_attention_keys
                        elif not self.attention_summary_only:
                            should_store_full = True
                        
                        if should_store_full:
                            # Store full tensor
                            cpu_tensor = tensor.detach().cpu()
                            window_attention_data[window_key] = cpu_tensor
                            logger.debug(f"Stored full tensor for: {window_key}")
                        else:
                            # Store only summary
                            summary = self._process_attention_immediately(window_key, tensor)
                            if summary:
                                window_attention_data[window_key] = summary
                                total_memory_saved += summary.get('memory_usage_mb', 0)
                        
                        del tensor
                
                if self.selective_storage:
                    full_stored = sum(1 for data in window_attention_data.values() if isinstance(data, torch.Tensor))
                    summaries_stored = len(window_attention_data) - full_stored
                    logger.info(f"Window {window_idx}: stored {full_stored} full tensors, {summaries_stored} summaries")
                elif self.attention_summary_only:
                    logger.info(f"Processed {len(window_attention_data)} attention summaries for window {window_idx} (saved ~{total_memory_saved:.1f}MB RAM)")
                else:
                    logger.info(f"Stored {len(window_attention_data)} full attention maps for window {window_idx} (moved to CPU)")
                
                self.window_attention_maps[window_idx] = window_attention_data
                
                # Store essential outputs for refiner (keep on CPU to save GPU memory)
                if "frame_embeds" in common_out:
                    all_frame_embeds.append(common_out["frame_embeds"].cpu())
                if "mask_features" in common_out:
                    all_mask_features.append(common_out["mask_features"].cpu())
                if "instance_embeds" in common_out:
                    all_instance_embeds.append(common_out["instance_embeds"].cpu())
                if "padding_masks" in common_out:
                    all_padding_masks.append(common_out["padding_masks"].cpu())
                
                # Clear intermediate variables
                del window_frames
                del common_out
                
                # Aggressive GPU memory cleanup after each window
                self.attention_maps.clear()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Log memory usage
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024**2
                    logger.info(f"GPU memory after window {window_idx}: {current_memory:.1f}MB")
            
            # Collect all attention data from all windows 
            all_attention_data = {}
            total_window_items = 0
            total_memory_saved = 0
            
            # Add window-specific attention data (summaries or full tensors)
            for window_idx, window_data in self.window_attention_maps.items():
                for key, data in window_data.items():
                    all_attention_data[key] = data
                    total_window_items += 1
                    if isinstance(data, dict) and 'memory_usage_mb' in data:
                        total_memory_saved += data['memory_usage_mb']
            
            # Update storage based on mode
            if self.attention_summary_only:
                self.attention_summaries.update(all_attention_data)
                self.attention_maps.clear()  # Clear any remaining full tensors
                logger.info(f"Stored {total_window_items} attention summaries (saved ~{total_memory_saved:.1f}MB total)")
            else:
                self.attention_maps = all_attention_data
            
            total_attention_items = len(all_attention_data)
            logger.info(f"Memory-optimized pipeline captured {total_attention_items} attention items")
            logger.info(f"  - Window-based items: {total_window_items}")
            if self.attention_summary_only:
                logger.info(f"  - Memory saved by using summaries: ~{total_memory_saved:.1f}MB")
            else:
                logger.info(f"  - All attention maps moved to CPU to save GPU memory")
            
            # Log the types of attention maps captured
            attention_types_captured = set()
            for key in all_attention_data.keys():
                # Remove window prefix to get clean key
                clean_key = key
                for i in range(10):  # Support up to 10 windows
                    clean_key = clean_key.replace(f'window_{i}_', '')
                
                if 'backbone_vit' in clean_key:
                    attention_types_captured.add('backbone_vit')
                elif 'pixel_decoder' in clean_key:
                    attention_types_captured.add('pixel_decoder')
                elif 'decoder' in clean_key:
                    attention_types_captured.add('decoder')
                elif 'tracker' in clean_key:
                    attention_types_captured.add('tracker')
                elif 'slot' in clean_key:
                    attention_types_captured.add('slot')
                elif 'refiner' in clean_key:
                    attention_types_captured.add('refiner')
                elif 'reid' in clean_key:
                    attention_types_captured.add('reid')
            
            logger.info(f"Attention types captured: {sorted(attention_types_captured)}")
            
            # Return results in the expected format
            window_results = [{
                'window_idx': 'all', 
                'start_frame': 0, 
                'end_frame': num_frames, 
                'actual_window_size': num_frames,
                'attention_maps': all_attention_data if not self.attention_summary_only else {},
                'attention_summaries': all_attention_data if self.attention_summary_only else {},
                'num_attention_maps': len(all_attention_data),
                'pipeline_stages': ['segmenter', 'tracker'],  # Skip refiner to save memory
                'attention_types': sorted(attention_types_captured),
                'num_windows': len(self.window_attention_maps),
                'memory_optimized': True
            }]
            
            return window_results
            
        except Exception as e:
            logger.error(f"Error during complete pipeline processing: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return empty results
        return [{
            'window_idx': 0, 
            'start_frame': 0, 
            'end_frame': num_frames, 
                'actual_window_size': num_frames,
                'attention_maps': {},
                'num_attention_maps': 0,
                'error': str(e)
        }]
    
    def _extract_attention_full_video(self, video_input, save_attention: bool, print_shapes: bool):
        """Fallback method for full video processing when window inference is disabled"""
        logger.info("Processing full video without windowing")
        
        # Clear previous attention maps
        self.attention_maps.clear()
        
        # Wrap in list for batch processing (same as training)
        batched_input = [video_input]
        
        # Run inference to trigger attention hooks
        with torch.no_grad():
            outputs = self.model(batched_input)
        
        logger.info(f"Extracted {len(self.attention_maps)} attention maps")
        
        # Print shapes if requested
        if print_shapes:
            self._print_attention_shapes()
        
        # Save attention maps if requested
        if save_attention:
            self._save_attention_maps(str(video_input.get('video_id', 'unknown')))
        
        return [{'window_idx': 0, 'start_frame': 0, 'end_frame': len(video_input['image']), 'attention_maps': self.attention_maps}]

    
    def _save_detailed_summary_to_file(self, detailed_lines: List[str], total_maps: int):
        """Save detailed attention summary to the report file"""
        try:
            # Get the model directory from the model path
            model_dir = Path(self.model_path).parent
            report_path = model_dir / "attention_maps_report.txt"
            
            # Append to existing report (since this is called after window report)
            with open(report_path, 'a') as f:
                for line in detailed_lines:
                    f.write(line + "\n")
                
                f.write(f"\nTotal attention maps extracted: {total_maps}\n")
                f.write(f"Report saved to: {report_path}\n")
            
            logger.info(f"Detailed summary appended to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving detailed summary to file: {e}")
    
    def save_complete_log_to_file(self):
        """Save the complete log buffer to the report file"""
        try:
            # Get the model directory from the model path
            model_dir = Path(self.model_path).parent
            report_path = model_dir / "attention_maps_report.txt"
            
            # Get all logged content
            log_content = self.get_log_content()
            
            # Append to existing report (since this is called after detailed summary)
            with open(report_path, 'a') as f:
                f.write("\n" + "="*80 + "\n")
                f.write("COMPLETE EXECUTION LOG")
                f.write("\n" + "="*80 + "\n")
                
                for line in log_content:
                    f.write(line + "\n")
            
            logger.info(f"Complete execution log appended to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving complete log to file: {e}")
    
    
    def _load_video_from_dataset(self, video_id: str) -> Dict[str, Any]:
        """
        Load video data from the registered Detectron2 dataset using the same method as training
        
        Args:
            video_id: Video ID from the dataset
            
        Returns:
            Video input in the format expected by the model
        """
        logger.info(f"Loading video {video_id} from registered dataset using proper data loader")
        
        try:
            # Import necessary modules for dataset loading
            from detectron2.data import DatasetCatalog, MetadataCatalog
            from dvis_Plus.data_video.datasets.ytvis import register_ytvis_instances
            from dvis_Plus import YTVISDatasetMapper
            
            # Register the dataset if not already registered
            if "ytvis_fishway_val" not in DatasetCatalog.list():
                register_ytvis_instances(
                    "ytvis_fishway_val",
                    {},
                    "/home/simone/shared-data/fishway_ytvis/val.json",
                    "/home/simone/shared-data/fishway_ytvis/all_videos"
                )
            
            # Use the same data loading method as training/evaluation
            # Create the same mapper that's used during training
            mapper = YTVISDatasetMapper(self.model.cfg, is_train=False)
            
            # Get the dataset
            dataset = DatasetCatalog.get("ytvis_fishway_val")
            
            # Find the specific video in the dataset
            video_data = None
            video_id_int = None
            video_id_str = str(video_id)
            
            try:
                video_id_int = int(video_id)
            except (ValueError, TypeError):
                pass
            
            for item in dataset:
                item_video_id = item.get('video_id')
                if (item_video_id == video_id or 
                    item_video_id == video_id_int or 
                    str(item_video_id) == video_id_str):
                    video_data = item
                    break
            
            if video_data is None:
                # Let's debug what video IDs are actually in the dataset
                available_ids = [item.get('video_id') for item in dataset]
                logger.error(f"Video ID {video_id} not found in dataset. Available IDs: {available_ids}")
                raise ValueError(f"Video ID {video_id} not found in dataset. Available IDs: {available_ids}")
            
            logger.info(f"Found video data: {video_data['file_names'][:3]}... (showing first 3 frames)")
            
            # Use the mapper to process the data in the same way as training
            # This ensures the data format is exactly what the model expects
            processed_data = mapper(video_data)
            
            logger.info(f"Data processed by mapper. Keys: {list(processed_data.keys())}")
            if 'image' in processed_data:
                logger.info(f"Image data type: {type(processed_data['image'])}, length: {len(processed_data['image'])}")
                if len(processed_data['image']) > 0:
                    logger.info(f"First frame type: {type(processed_data['image'][0])}, shape: {processed_data['image'][0].shape if hasattr(processed_data['image'][0], 'shape') else 'N/A'}")
                    
                    # Additional debugging for the first few frames
                    for i in range(min(3, len(processed_data['image']))):
                        frame = processed_data['image'][i]
                        if hasattr(frame, 'shape'):
                            logger.info(f"Frame {i}: shape={frame.shape}, dtype={frame.dtype}, range=[{frame.min():.3f}, {frame.max():.3f}]")
                        else:
                            logger.info(f"Frame {i}: no shape attribute, type={type(frame)}")
            else:
                logger.warning("No 'image' key found in processed data")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error loading video from dataset: {e}")
            raise
    
    def extract_attention_for_video_file(self, video_path: str, save_attention: bool = False, print_shapes: bool = True) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for a raw video file (fallback method)
        
        Args:
            video_path: Path to raw video file
            save_attention: Whether to save attention maps to disk
            print_shapes: Whether to print attention map shapes
            
        Returns:
            Dictionary containing all attention maps
        """
        logger.warning("Using raw video file - this may not match the training data format exactly")
        
        # Clear previous attention maps
        self.attention_maps.clear()
        
        # Load and preprocess video
        video_input = self._load_raw_video(video_path)
        
        # Run inference to trigger attention hooks
        with torch.no_grad():
            outputs = self.model(video_input)
        
        logger.info(f"Extracted {len(self.attention_maps)} attention maps")
        
        # Print shapes if requested
        if print_shapes:
            self._print_attention_shapes()
        
        # Save attention maps if requested
        if save_attention:
            self._save_attention_maps(video_path)
        
        return self.attention_maps
    
    def _load_raw_video(self, video_path: str) -> Dict[str, Any]:
        """
        Load raw video file (fallback method)
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed video input
        """
        logger.info(f"Loading raw video from: {video_path}")
        
        try:
            # Load video frames (implement based on your video format)
            video_frames = self._load_video_frames(video_path)
            
            # Preprocess frames (normalize, resize, etc.)
            processed_frames = self._preprocess_frames(video_frames)
            
            # Create input format expected by your model
            video_input = {
                'image': processed_frames,
                'height': processed_frames.shape[-2],
                'width': processed_frames.shape[-1],
                'video_id': os.path.basename(video_path)
            }
            
            logger.info(f"Video loaded successfully. Shape: {processed_frames.shape}")
            return video_input
            
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            raise
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """
        Load video frames from file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video frames tensor
        """
        # Implement video loading based on your video format
        # This is a placeholder - replace with actual implementation
        
        # Example: Using OpenCV
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames loaded from video")
            
            # Convert to tensor: [T, H, W, C]
            frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
            
            # Normalize to [0, 1]
            frames_tensor = frames_tensor / 255.0
            
            return frames_tensor
            
        except ImportError:
            logger.warning("OpenCV not available. Using dummy video data.")
            # Fallback: create dummy video data for testing
            dummy_frames = torch.randn(31, 480, 640, 3)  # 31 frames, 480x640, 3 channels
            return dummy_frames
    
    def _preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Preprocess video frames for the model
        
        Args:
            frames: Raw video frames [T, H, W, C]
            
        Returns:
            Preprocessed frames
        """
        # This should match your model's preprocessing requirements
        
        # Resize frames to model input size (modify based on your config)
        target_height, target_width = 480, 640  # Adjust based on your model config
        
        if frames.shape[1] != target_height or frames.shape[2] != target_width:
            frames = F.interpolate(
                frames.permute(0, 3, 1, 2),  # [T, C, H, W]
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # Back to [T, H, W, C]
        
        # Convert to list of frames as expected by your model
        frame_list = [frames[i] for i in range(frames.shape[0])]
        
        return frame_list
    
    def _print_attention_shapes(self, save_to_file: bool = True):
        """Print shapes of all captured attention maps and optionally save to file"""
        # Prepare the output content
        output_lines = []
        
        output_lines.append("\n" + "="*80)
        output_lines.append(f"TOTAL ATTENTION MAPS: {len(self.attention_maps)}")
        output_lines.append("="*80)
        
        if not self.attention_maps:
            output_lines.append("No attention maps captured!")
            # Print to console
            for line in output_lines:
                print(line)
            return
        
        # Debug: Print all keys to see what's actually stored
        output_lines.append(f"\nDEBUG: All attention map keys ({len(self.attention_maps)} total):")
        for i, key in enumerate(sorted(self.attention_maps.keys())):
            output_lines.append(f"  {i+1:2d}. {key}")
        
        output_lines.append("\n" + "="*80)
        output_lines.append("ATTENTION MAPS STATISTICAL SUMMARY")
        output_lines.append("="*80)
        
        # Group by type for better organization, handling window-based keys
        attention_types = {}
        window_info = {}
        
        for key, tensor in self.attention_maps.items():
            # Extract window information if present
            window_idx = None
            clean_key = key
            if key.startswith('window_'):
                parts = key.split('_', 2)
                if len(parts) >= 3:
                    window_idx = int(parts[1])
                    clean_key = parts[2]
                    if window_idx not in window_info:
                        window_info[window_idx] = []
                    window_info[window_idx].append(key)
            
            # Determine group based on cleaned key
            if 'backbone' in clean_key:
                if 'backbone_vit' in clean_key:
                    group = 'backbone_vit'
                elif 'backbone_interaction' in clean_key:
                    group = 'backbone_interaction'
                elif 'backbone_injector' in clean_key:
                    group = 'backbone_injector'
                else:
                    group = 'backbone_other'
            elif 'pixel_decoder' in clean_key:
                group = 'pixel_decoder'
            elif 'decoder' in clean_key:
                group = 'decoder'
            elif 'tracker' in clean_key:
                group = 'tracker'
            elif 'slot' in clean_key:
                group = 'slot'
            elif 'refiner' in clean_key:
                group = 'refiner'
            else:
                group = 'other'
            
            if group not in attention_types:
                attention_types[group] = []
            attention_types[group].append((key, tensor, window_idx))
        
        # Add window information summary if windows were processed
        if window_info:
            output_lines.append(f"\nWINDOW PROCESSING SUMMARY:")
            output_lines.append(f"Number of windows processed: {len(window_info)}")
            for window_idx in sorted(window_info.keys()):
                output_lines.append(f"  Window {window_idx}: {len(window_info[window_idx])} attention maps")
        
        # Print summary by type with dimension interpretations
        for group_name, group_items in attention_types.items():
            output_lines.append(f"\n{group_name.upper()} ATTENTION MAPS ({len(group_items)} maps):")
            for key, tensor, window_idx in group_items:
                if tensor is not None:
                    shape_str = str(list(tensor.shape))
                    range_str = f"[{tensor.min().item():.4f}, {tensor.max().item():.4f}]"
                    mean_str = f"{tensor.mean().item():.4f}"
                    std_str = f"{tensor.std().item():.4f}"
                    
                    # Get dimension interpretation
                    dims_origin = self._get_dimension_interpretation(key, list(tensor.shape))
                    
                    # Add window information if applicable
                    window_prefix = f"[W{window_idx}] " if window_idx is not None else ""
                    
                    output_lines.append(f"  {window_prefix}{key}:")
                    output_lines.append(f"    Shape: {shape_str}")
                    output_lines.append(f"    Dims_origin: {dims_origin}")
                    output_lines.append(f"    Range: {range_str}")
                    output_lines.append(f"    Mean: {mean_str}")
                    output_lines.append(f"    Std: {std_str}")
                    
                    # Highlight if window size is found in dimensions
                    if window_idx is not None:
                        for dim in tensor.shape:
                            if dim in [31, 30, 29, 28, 27, 26, 25]:
                                output_lines.append(f"    ✓ Window size {dim} detected in tensor dimensions")
                else:
                    output_lines.append(f"  {key}: None")
        
        # Print to console and log
        for line in output_lines:
            self.log_print(line)
        
        # Save to file if requested
        if save_to_file:
            self._save_attention_report_to_file(output_lines)
    
    def _save_attention_report_to_file(self, output_lines: List[str]):
        """Save attention report to a text file in the model directory"""
        try:
            # Get the model directory from the model path
            model_dir = Path(self.model_path).parent
            report_path = model_dir / "attention_maps_report.txt"
            
            # Write the report to file
            with open(report_path, 'w') as f:
                f.write("DVIS-DAQ ATTENTION MAPS EXTRACTION REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Model: {self.model_path}\n")
                f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Attention Maps: {len(self.attention_maps)}\n")
                f.write("\n")
                
                for line in output_lines:
                    f.write(line + "\n")
            
            logger.info(f"Attention report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving attention report to file: {e}")
    
    def _save_attention_maps(self, video_identifier: str):
        """
        Save all extracted attention maps to disk
        
        Args:
            video_identifier: Video ID or path (used for naming)
        """
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use video_identifier as directory name
        video_output_dir = self.output_dir / str(video_identifier)
        video_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving attention maps to: {video_output_dir}")
        
        # Save each attention map
        for attn_name, attn_tensor in self.attention_maps.items():
            if attn_tensor is not None:
                # Create filename
                filename = f"{attn_name}.pt"
                filepath = video_output_dir / filename
                
                # Save tensor
                torch.save(attn_tensor, filepath)
                
                # Also save metadata
                metadata = {
                    'name': attn_name,
                    'shape': list(attn_tensor.shape),
                    'dtype': str(attn_tensor.dtype),
                    'device': str(attn_tensor.device),
                    'video_source': video_identifier
                }
                
                metadata_filepath = video_output_dir / f"{attn_name}_metadata.pt"
                torch.save(metadata, metadata_filepath)
                
                logger.info(f"Saved {attn_name}: {attn_tensor.shape} -> {filepath}")
        
        # Save summary of all attention maps
        summary = {
            'video_identifier': video_identifier,
            'num_attention_maps': len(self.attention_maps),
            'attention_map_names': list(self.attention_maps.keys()),
            'extraction_timestamp': str(torch.tensor(0))  # Placeholder for timestamp
        }
        
        summary_filepath = video_output_dir / "attention_summary.pt"
        torch.save(summary, summary_filepath)
        
        logger.info(f"Saved attention summary to: {summary_filepath}")
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """
        Get summary of extracted attention maps
        
        Returns:
            Summary dictionary
        """
        summary = {}
        
        if self.attention_summary_only:
            # Return pre-computed summaries
            for attn_name, attn_summary in self.attention_summaries.items():
                if attn_summary is not None:
                    summary[attn_name] = {
                        'shape': attn_summary['shape'],
                        'dtype': attn_summary['dtype'],
                        'device': attn_summary.get('device_original', 'unknown'),
                        'min_value': attn_summary['min_value'],
                        'max_value': attn_summary['max_value'],
                        'mean_value': attn_summary['mean_value'],
                        'std_value': attn_summary['std_value']
                    }
        else:
            # Compute summaries from full tensors
        for attn_name, attn_tensor in self.attention_maps.items():
            if attn_tensor is not None:
                summary[attn_name] = {
                    'shape': list(attn_tensor.shape),
                    'dtype': str(attn_tensor.dtype),
                    'device': str(attn_tensor.device),
                    'min_value': float(attn_tensor.min()),
                    'max_value': float(attn_tensor.max()),
                    'mean_value': float(attn_tensor.mean()),
                    'std_value': float(attn_tensor.std())
                }
        
        return summary
    
    def list_available_videos(self) -> list:
        """
        List available video IDs from the registered dataset
        
        Returns:
            List of available video IDs
        """
        try:
            from detectron2.data import DatasetCatalog
            from dvis_Plus.data_video.datasets.ytvis import register_ytvis_instances
            
            # Register the dataset if not already registered
            if "ytvis_fishway_val" not in DatasetCatalog.list():
                register_ytvis_instances(
                    "ytvis_fishway_val",
                    {},
                    "/home/simone/shared-data/fishway_ytvis/val.json",
                    "/home/simone/shared-data/fishway_ytvis/all_videos"
                )
            
            # Get dataset
            dataset = DatasetCatalog.get("ytvis_fishway_val")
            
            # Extract unique video IDs and show some debug info
            video_ids = []
            video_info = []
            
            for i, item in enumerate(dataset):
                video_id = item.get('video_id')
                if video_id not in video_ids:
                    video_ids.append(video_id)
                    # Show some info about the first few items
                    if len(video_info) < 3:
                        video_info.append({
                            'index': i,
                            'video_id': video_id,
                            'type': type(video_id).__name__,
                            'file_names_count': len(item.get('file_names', [])),
                            'has_annotations': 'annotations' in item
                        })
            
            video_ids.sort()
            
            # Print debug info
            logger.info(f"Dataset contains {len(dataset)} items")
            logger.info(f"Found {len(video_ids)} unique video IDs")
            if video_info:
                logger.info("Sample dataset items:")
                for info in video_info:
                    logger.info(f"  Item {info['index']}: video_id={info['video_id']} (type: {info['type']}), frames: {info['file_names_count']}, has_annotations: {info['has_annotations']}")
            
            return video_ids
            
        except Exception as e:
            logger.error(f"Error listing available videos: {e}")
            return []

    def _get_dimension_interpretation(self, attn_name: str, tensor_shape: List[int]) -> str:
        """
        Get human-readable interpretation of tensor dimensions for attention maps
        
        Args:
            attn_name: Name of the attention map
            tensor_shape: List of tensor dimensions
            
        Returns:
            String describing what each dimension represents
        """
        try:
            # Extract window size from attention name if present
            window_size_note = ""
            if 'window_' in attn_name:
                # Check if any dimension matches common window sizes (31, 30, etc.)
                for dim in tensor_shape:
                    if dim in [31, 30, 29, 28, 27, 26, 25]:  # Common window sizes
                        window_size_note = f" (window_size={dim})"
                        break
            
            if 'backbone_vit' in attn_name:
                if len(tensor_shape) == 4:
                    # For ViT backbone, the last two dimensions are typically H*W (spatial dimensions)
                    return f"[batch_size, num_heads, H*W, H*W]{window_size_note}"
                else:
                    return f"[unknown]{window_size_note}"
            
            elif 'backbone_interaction' in attn_name or 'backbone_injector' in attn_name:
                if len(tensor_shape) == 4:
                    return f"[batch_size, num_heads, sequence_length, sequence_length]{window_size_note}"
                else:
                    return f"[unknown]{window_size_note}"
            
            elif 'pixel_decoder' in attn_name:
                if len(tensor_shape) == 4:
                    # For pixel decoder, this is typically query-key attention
                    return f"[batch_size, query_length, num_heads, H*W]{window_size_note}"
                else:
                    return f"[unknown]{window_size_note}"
            
            elif 'decoder' in attn_name:
                if len(tensor_shape) == 4:
                    if 'self_attn' in attn_name:
                        return f"[batch_size, num_heads, num_queries, num_queries]{window_size_note}"
                    elif 'cross_attn' in attn_name:
                        return f"[batch_size, num_heads, num_queries, H*W]{window_size_note}"
                    else:
                        return f"[batch_size, num_heads, query_length, key_length]{window_size_note}"
                else:
                    return f"[unknown]{window_size_note}"
            
            elif 'tracker' in attn_name:
                if len(tensor_shape) == 4:
                    if 'self_attn' in attn_name:
                        return f"[batch_size, num_heads, num_instances, num_instances]{window_size_note}"
                    elif 'cross_attn' in attn_name:
                        return f"[batch_size, num_heads, num_instances, H*W]{window_size_note}"
                    else:
                        return f"[batch_size, num_heads, num_instances, feature_length]{window_size_note}"
                else:
                    return f"[unknown]{window_size_note}"
            
            elif 'slot' in attn_name:
                if len(tensor_shape) == 3:
                    return f"[batch_size, num_slots, feature_dim]{window_size_note}"
                else:
                    return f"[unknown]{window_size_note}"
            
            elif 'reid' in attn_name:
                if len(tensor_shape) == 4:
                    return f"[batch_size, num_heads, reid_sequence, reid_sequence]{window_size_note}"
                else:
                    return f"[unknown]{window_size_note}"
            
            elif 'refiner' in attn_name:
                if len(tensor_shape) == 4:
                    if 'long_temp' in attn_name:
                        # Check if temporal dimension matches window size
                        temporal_note = ""
                        for dim in tensor_shape:
                            if dim > 20 and dim < 50:  # Likely temporal dimension
                                temporal_note = f" (temporal_frames={dim})"
                                break
                        return f"[batch_size, num_heads, temporal_length, temporal_length]{temporal_note}"
                    elif 'obj' in attn_name:
                        return f"[batch_size, num_heads, num_objects, num_objects]{window_size_note}"
                    elif 'cross' in attn_name:
                        return f"[batch_size, num_heads, query_length, H*W]{window_size_note}"
                    else:
                        return f"[batch_size, num_heads, query_length, key_length]{window_size_note}"
                else:
                    return f"[unknown]{window_size_note}"
            
            else:
                # Generic interpretation for unknown attention types
                if len(tensor_shape) == 4:
                    return f"[batch_size, num_heads, dim1, dim2]{window_size_note}"
                else:
                    return f"[unknown]{window_size_note}"
                    
        except Exception as e:
            # Fallback to basic interpretation if config extraction fails
            logger.debug(f"Could not extract config info for {attn_name}: {e}")
            window_size_note = ""
            if 'window_' in attn_name:
                for dim in tensor_shape:
                    if dim in [31, 30, 29, 28, 27, 26, 25]:
                        window_size_note = f" (window_size={dim})"
                        break
            
            if len(tensor_shape) == 4:
                return f"[batch_size, num_heads, dim1, dim2]{window_size_note}"
            else:
                return f"[unknown]{window_size_note}"
    
    def _get_backbone_num_heads(self) -> Optional[int]:
        """Get number of attention heads from backbone configuration"""
        try:
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'vit_module'):
                vit_module = self.model.backbone.vit_module
                if hasattr(vit_module, 'blocks') and len(vit_module.blocks) > 0:
                    first_block = vit_module.blocks[0]
                    if hasattr(first_block, 'attn') and hasattr(first_block.attn, 'num_heads'):
                        return first_block.attn.num_heads
            return None
        except:
            return None
    
    def _get_pixel_decoder_num_heads(self) -> Optional[int]:
        """Get number of attention heads from pixel decoder configuration"""
        try:
            if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'pixel_decoder'):
                pixel_decoder = self.model.sem_seg_head.pixel_decoder
                if hasattr(pixel_decoder, 'transformer') and hasattr(pixel_decoder.transformer, 'encoder'):
                    if hasattr(pixel_decoder.transformer.encoder, 'layers') and len(pixel_decoder.transformer.encoder.layers) > 0:
                        first_layer = pixel_decoder.transformer.encoder.layers[0]
                        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'n_heads'):
                            return first_layer.self_attn.n_heads
            return None
        except:
            return None
    
    def _get_pixel_decoder_num_levels(self) -> Optional[int]:
        """Get number of feature levels from pixel decoder configuration"""
        try:
            if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'pixel_decoder'):
                pixel_decoder = self.model.sem_seg_head.pixel_decoder
                if hasattr(pixel_decoder, 'transformer') and hasattr(pixel_decoder.transformer, 'encoder'):
                    if hasattr(pixel_decoder.transformer.encoder, 'layers') and len(pixel_decoder.transformer.encoder.layers) > 0:
                        first_layer = pixel_decoder.transformer.encoder.layers[0]
                        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'n_levels'):
                            return first_layer.self_attn.n_levels
            return None
        except:
            return None
    
    def _get_pixel_decoder_num_points(self) -> Optional[int]:
        """Get number of reference points from pixel decoder configuration"""
        try:
            if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'pixel_decoder'):
                pixel_decoder = self.model.sem_seg_head.pixel_decoder
                if hasattr(pixel_decoder, 'transformer') and hasattr(pixel_decoder.transformer, 'encoder'):
                    if hasattr(pixel_decoder.transformer.encoder, 'layers') and len(pixel_decoder.transformer.encoder.layers) > 0:
                        first_layer = pixel_decoder.transformer.encoder.layers[0]
                        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'n_points'):
                            return first_layer.self_attn.n_points
            return None
        except:
            return None
    
    def _get_decoder_num_heads(self) -> Optional[int]:
        """Get number of attention heads from decoder configuration"""
        try:
            if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'predictor'):
                predictor = self.model.sem_seg_head.predictor
                if hasattr(predictor, 'transformer_self_attention_layers') and len(predictor.transformer_self_attention_layers) > 0:
                    first_layer = predictor.transformer_self_attention_layers[0]
                    if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'num_heads'):
                        return first_layer.self_attn.num_heads
            return None
        except:
            return None
    
    def _get_decoder_num_queries(self) -> Optional[int]:
        """Get number of queries from decoder configuration"""
        try:
            return self.get_config_parameter('MODEL.MASK_FORMER.NUM_OBJECT_QUERIES', None)
        except:
            return None
    
    def _get_tracker_num_heads(self) -> Optional[int]:
        """Get number of attention heads from tracker configuration"""
        try:
            if hasattr(self.model, 'tracker'):
                tracker = self.model.tracker
                if hasattr(tracker, 'transformer_self_attention_layers') and len(tracker.transformer_self_attention_layers) > 0:
                    first_layer = tracker.transformer_self_attention_layers[0]
                    if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'num_heads'):
                        return first_layer.self_attn.num_heads
            return None
        except:
            return None
    
    def _get_tracker_num_instances(self) -> Optional[int]:
        """Get number of instances from tracker configuration"""
        try:
            return self.get_config_parameter('MODEL.MASK_FORMER.NUM_OBJECT_QUERIES', None)
        except:
            return None
    
    def _get_tracker_num_slots(self) -> Optional[int]:
        """Get number of slots from tracker configuration"""
        try:
            if hasattr(self.model, 'tracker'):
                tracker = self.model.tracker
                if hasattr(tracker, 'slot_cross_attention_layers') and len(tracker.slot_cross_attention_layers) > 0:
                    first_layer = tracker.slot_cross_attention_layers[0]
                    if hasattr(first_layer, 'slot_attn') and hasattr(first_layer.slot_attn, 'num_slots'):
                        return first_layer.slot_attn.num_slots
            return None
        except:
            return None
    
    def _get_reid_num_heads(self) -> Optional[int]:
        """Get number of attention heads from ReID configuration"""
        try:
            if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'predictor'):
                predictor = self.model.sem_seg_head.predictor
                if hasattr(predictor, 'reid_embed') and hasattr(predictor.reid_embed, 'attention_layers'):
                    if len(predictor.reid_embed.attention_layers) > 0:
                        first_layer = predictor.reid_embed.attention_layers[0]
                        if hasattr(first_layer, 'attention') and hasattr(first_layer.attention, 'num_heads'):
                            return first_layer.attention.num_heads
            return None
        except:
            return None
    
    def _get_refiner_num_heads(self) -> Optional[int]:
        """Get number of attention heads from refiner configuration"""
        try:
            if hasattr(self.model, 'refiner'):
                refiner = self.model.refiner
                if hasattr(refiner, 'transformer_time_self_attention_layers') and len(refiner.transformer_time_self_attention_layers) > 0:
                    first_layer = refiner.transformer_time_self_attention_layers[0]
                    if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'num_heads'):
                        return first_layer.self_attn.num_heads
            return None
        except:
            return None


def main():
    """
    Main function to demonstrate usage
    """
    # Configuration
    model_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth"
    output_dir = "/store/simone/attention/"
    
    # Initialize extractor
    extractor = AttentionExtractor(model_path, output_dir)
    
    # List available videos
    print("Available video IDs:")
    available_videos = extractor.list_available_videos()
    for video_id in available_videos[:10]:  # Show first 10
        print(f"  {video_id}")
    if len(available_videos) > 10:
        print(f"  ... and {len(available_videos) - 10} more")
    
    if available_videos:
        # Extract attention maps for the first available video (just print shapes, don't save)
        video_id = available_videos[0]
        print(f"\nExtracting attention maps for video: {video_id}")
        
        attention_maps = extractor.extract_attention_for_video(
            video_id, 
            save_attention=False,  # Don't save to disk
            print_shapes=True      # Print shapes to console
        )
        
        # Print detailed summary
        summary = extractor.get_attention_summary()
        print("\n" + "="*80)
        print("DETAILED ATTENTION MAPS SUMMARY")
        print("="*80)
        
        for attn_name, attn_info in summary.items():
            print(f"\n{attn_name}:")
            print(f"  Shape: {attn_info['shape']}")
            print(f"  Range: [{attn_info['min_value']:.4f}, {attn_info['max_value']:.4f}]")
            print(f"  Mean: {attn_info['mean_value']:.4f}")
            print(f"  Std: {attn_info['std_value']:.4f}")
        
        print(f"\nTotal attention maps extracted: {len(summary)}")
        print(f"Output directory: {output_dir}")
    else:
        print("No videos found in dataset")


class TargetedAttentionExtractor:
    """
    Wrapper for memory-efficient targeted attention extraction
    Only extracts and stores the specific attention maps needed for visualization
    """
    
    def __init__(self, model_path: str, output_dir: str = "/store/simone/attention/"):
        self.extractor = AttentionExtractor(model_path, output_dir)
    
    def extract_for_visualization(self, video_id: str, layer_idx: int, frame_idx: int) -> Dict[str, torch.Tensor]:
        """
        Extract only the specific attention map needed for visualization
        
        Args:
            video_id: Video ID from dataset
            layer_idx: Which backbone layer to extract (e.g., 5 for layer 5)
            frame_idx: Global frame index in the video
            
        Returns:
            Dictionary with only the needed attention map
        """
        # Determine which window contains this frame
        window_size = self.extractor.get_config_parameter('MODEL.MASK_FORMER.TEST.WINDOW_SIZE', 31)
        window_idx = frame_idx // window_size
        
        # Target only the specific attention map we need
        target_key = f'backbone_vit_layer_{layer_idx}'
        self.extractor.enable_selective_storage([target_key])
        
        logger.info(f"Extracting targeted attention for frame {frame_idx} (window {window_idx}), layer {layer_idx}")
        logger.info(f"Target key: {target_key}")
        
        # Extract attention maps
        results = self.extractor.extract_attention_for_video(
            video_id, 
            save_attention=False,
            print_shapes=False
        )
        
        # Return only the attention maps (not summaries)
        attention_maps = {}
        for key, tensor in self.extractor.attention_maps.items():
            if isinstance(tensor, torch.Tensor):
                attention_maps[key] = tensor
        
        logger.info(f"Successfully extracted {len(attention_maps)} targeted attention maps")
        return attention_maps
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """Get summary including both stored tensors and summaries"""
        return self.extractor.get_attention_summary()
    
    # Expose needed methods from the underlying extractor
    def _load_video_from_dataset(self, video_id: str):
        """Delegate to underlying extractor"""
        return self.extractor._load_video_from_dataset(video_id)
    
    @property
    def attention_maps(self):
        """Access to underlying attention maps"""
        return self.extractor.attention_maps


if __name__ == "__main__":
    main()
