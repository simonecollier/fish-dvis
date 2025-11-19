import torch
import torch.nn as nn
import logging
import math

class RefinerAttentionHook:
    """Hook to capture attention weights from refiner MultiheadAttention layers."""
    
    def __init__(self, layer_name, attention_module, sink_list=None, extractor=None):
        self.layer_name = layer_name
        self.attention_module = attention_module
        self.refiner_attention_maps = []
        self._sink_list = sink_list
        self._extractor = extractor  # Reference to extractor for immediate saving
    
    def __call__(self, module, input, output):
        """Capture attention weights from the forward pass."""
        if isinstance(output, tuple) and len(output) >= 2:
            # MultiheadAttention returns (attn_output, attn_output_weights)
            attn_output, attn_weights = output[0], output[1]
            
            if attn_weights is not None:
                # Convert to numpy immediately to free GPU memory
                attn_weights_np = attn_weights.detach().cpu().numpy()
                
                # Store attention weights with layer information
                entry = {
                    'layer': self.layer_name,
                    'attention_weights': attn_weights_np,  # Store as numpy array
                    'shape': attn_weights.shape
                }
                
                # Debug: print when we capture attention
                print(f"[REFINER_HOOK] Captured attention from {self.layer_name}, shape: {attn_weights.shape}, save_immediately: {self._extractor.save_immediately_from_hook if self._extractor else False}", flush=True)
                
                # Save immediately from hook if enabled (prevents accumulation)
                if self._extractor is not None and self._extractor.save_immediately_from_hook:
                    # Save directly to disk without accumulating in lists
                    self._save_from_hook(entry)
                else:
                    # Store entry in sink list (will be cleared after saving)
                    self.refiner_attention_maps.append(entry)
                    if self._sink_list is not None:
                        self._sink_list.append(entry)
    
    def _save_from_hook(self, entry):
        """Save attention map directly from hook without accumulating in lists."""
        if self._extractor is None:
            print(f"[REFINER_HOOK] {self.layer_name} - Cannot save: extractor is None", flush=True)
            return
        if not self._extractor._output_dir:
            print(f"[REFINER_HOOK] {self.layer_name} - Cannot save: output_dir is not set (value: {self._extractor._output_dir})", flush=True)
            return
        
        try:
            import os
            
            # Get video context if available
            video_context = self._extractor._current_video_context
            if video_context is not None:
                video_id = video_context.get('video_id', 'unknown')
                frame_idx = video_context.get('frame_idx', None)
                frame_start = video_context.get('frame_start', None)
                frame_end = video_context.get('frame_end', None)
                window_idx = video_context.get('window_idx', None)
            else:
                video_id = 'unknown'
                frame_idx = None
                frame_start = None
                frame_end = None
                window_idx = None
            
            attn_dir = os.path.join(self._extractor._output_dir, "attention_maps")
            os.makedirs(attn_dir, exist_ok=True)
            
            # Build filename with video context
            # Extract layer index from layer name (e.g., "transformer_time_self_attention_layers.0.self_attn" -> 0)
            layer_idx = None
            if 'transformer_time_self_attention_layers.' in self.layer_name:
                try:
                    layer_idx = int(self.layer_name.split('transformer_time_self_attention_layers.')[1].split('.')[0])
                except (ValueError, IndexError):
                    pass
            
            # For refiner attention, check if frame_idx is a list (all frames)
            # The refiner processes all frames at once, not per window
            # So if frame_idx is a list, use the full range from the list
            if isinstance(frame_idx, (list, tuple)) and len(frame_idx) > 0:
                # Use the full frame range from the list
                actual_frame_start = min(frame_idx)
                actual_frame_end = max(frame_idx)
                if layer_idx is not None:
                    filename = f"video_{video_id}_frames{actual_frame_start}-{actual_frame_end}_refiner_temporal_layer_{layer_idx}_attn"
                else:
                    filename = f"video_{video_id}_frames{actual_frame_start}-{actual_frame_end}_refiner_temporal_{self.layer_name.replace('.', '_')}_attn"
            elif frame_start is not None and frame_end is not None:
                # Fall back to window-level frame range if frame_idx is not a list
                if layer_idx is not None:
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_refiner_temporal_layer_{layer_idx}_attn"
                else:
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_refiner_temporal_{self.layer_name.replace('.', '_')}_attn"
            elif frame_idx is not None:
                # Single frame index
                if isinstance(frame_idx, (list, tuple)) and len(frame_idx) > 0:
                    frame_idx = frame_idx[0]
                if layer_idx is not None:
                    filename = f"video_{video_id}_frame_{frame_idx}_refiner_temporal_layer_{layer_idx}_attn"
                else:
                    filename = f"video_{video_id}_frame_{frame_idx}_refiner_temporal_{self.layer_name.replace('.', '_')}_attn"
            else:
                if layer_idx is not None:
                    filename = f"video_{video_id}_unknown_refiner_temporal_layer_{layer_idx}_attn"
                else:
                    filename = f"video_{video_id}_unknown_refiner_temporal_{self.layer_name.replace('.', '_')}_attn"
            
            # Add video context to entry
            entry['video_id'] = video_id
            entry['frame_idx'] = frame_idx
            entry['frame_start'] = frame_start
            entry['frame_end'] = frame_end
            entry['window_idx'] = window_idx
            entry['source'] = 'refiner_temporal'
            
            # Save synchronously to disk
            if self._extractor is not None:
                print(f"[REFINER_HOOK] Saving {filename} to {attn_dir}", flush=True)
                self._extractor._save_synchronously(entry, filename, attn_dir)
                print(f"[REFINER_HOOK] Successfully saved {filename}", flush=True)
        except Exception as e:
            import logging
            import traceback
            error_msg = f"Failed to save refiner attention from hook: {e}\n{traceback.format_exc()}"
            logging.getLogger(__name__).warning(error_msg)
            print(f"[REFINER_HOOK] ERROR: {error_msg}", flush=True)


class TrackerAttentionHook:
    """Hook to capture attention weights from tracker attention layers."""
    
    def __init__(self, layer_name, layer_idx, attention_type, attention_module, sink_list=None, extractor=None):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.attention_type = attention_type  # 'cross', 'self', or 'slot_cross'
        self.attention_module = attention_module
        self.attention_maps = []
        self._sink_list = sink_list
        self._extractor = extractor  # Reference to extractor for immediate saving
    
    def __call__(self, module, input, output):
        """Capture attention weights from the forward pass."""
        # Debug: print output type to understand what we're getting
        # print(f"[TRACKER_HOOK] {self.layer_name} output type: {type(output)}, is_tuple: {isinstance(output, tuple)}, len: {len(output) if isinstance(output, tuple) else 'N/A'}")
        
        if isinstance(output, tuple) and len(output) >= 2:
            # MultiheadAttention returns (attn_output, attn_output_weights)
            attn_output, attn_weights = output[0], output[1]
            
            if attn_weights is not None:
                # Convert to numpy immediately to free GPU memory
                attn_weights_np = attn_weights.detach().cpu().numpy()
                
                # Get tracker_id -> sequence_id mapping from the tracker
                tracker_id_to_seq_id = None
                if self._extractor is not None:
                    tracker_id_to_seq_id = self._extractor._get_tracker_id_mapping()
                
                entry = {
                    'layer': self.layer_name,
                    'layer_idx': self.layer_idx,
                    'attention_type': self.attention_type,
                    'attention_weights': attn_weights_np,  # Store as numpy array
                    'shape': attn_weights.shape,
                    'tracker_id_to_seq_id': tracker_id_to_seq_id,  # Mapping: tracker_id (index) -> sequence_id
                    'source': 'tracker'
                }
                
                # Debug: print when we capture attention
                print(f"[TRACKER_HOOK] Captured attention from {self.layer_name}, shape: {attn_weights.shape}, save_immediately: {self._extractor.save_immediately_from_hook if self._extractor else False}", flush=True)
                
                # Save immediately from hook if enabled
                if self._extractor is not None and self._extractor.save_immediately_from_hook:
                    self._save_from_hook(entry)
                else:
                    # Store entry in sink list
                    if self._sink_list is not None:
                        self._sink_list.append(entry)
                    else:
                        self.attention_maps.append(entry)
            else:
                # Debug: print when attn_weights is None
                print(f"[TRACKER_HOOK] {self.layer_name} - attn_weights is None (need_weights might be False)", flush=True)
        else:
            # Debug: print when output format is unexpected
            print(f"[TRACKER_HOOK] {self.layer_name} - Unexpected output format: type={type(output)}, is_tuple={isinstance(output, tuple)}, len={len(output) if isinstance(output, (tuple, list)) else 'N/A'}", flush=True)
    
    def _save_from_hook(self, entry):
        """Save attention map directly from hook without accumulating in lists."""
        if self._extractor is None or not self._extractor._output_dir:
            return
        
        try:
            import os
            import json
            from detectron2.utils.file_io import PathManager
            
            # Get video context if available
            video_context = self._extractor._current_video_context
            if video_context is not None:
                video_id = video_context.get('video_id', 'unknown')
                frame_idx = video_context.get('frame_idx', None)
                frame_start = video_context.get('frame_start', None)
                frame_end = video_context.get('frame_end', None)
                window_idx = video_context.get('window_idx', None)
            else:
                video_id = 'unknown'
                frame_idx = None
                frame_start = None
                frame_end = None
                window_idx = None
            
            # IMPORTANT: For tracker attention, use the current tracker frame index
            # which is updated by the frame tracker hook. This ensures we get the correct
            # frame index even if the video context hasn't been updated yet.
            if self._extractor._current_tracker_frame_idx is not None:
                frame_idx = self._extractor._current_tracker_frame_idx
                print(f"[TRACKER_HOOK] Using tracker frame index: {frame_idx} (from frame tracker hook)", flush=True)
            elif self._extractor._tracker_start_frame_id is not None:
                # Fallback: if frame tracker hook hasn't fired yet, try to calculate from start_frame_id
                # This shouldn't normally happen, but provides a safety net
                print(f"[TRACKER_HOOK] Warning: _current_tracker_frame_idx is None, but start_frame_id={self._extractor._tracker_start_frame_id}. Using fallback.", flush=True)
                # Don't use fallback - let it use frame_idx from context or frame range
            
            attn_dir = os.path.join(self._extractor._output_dir, "attention_maps")
            os.makedirs(attn_dir, exist_ok=True)
            
            # Build filename with video context
            attention_type_str = self.attention_type.replace('_', '_')
            layer_idx = entry.get('layer_idx', 'unknown')
            
            # For tracker attention, prefer frame_idx if available (per-frame attention)
            # Otherwise fall back to frame range (window-level)
            if frame_idx is not None:
                # Use single frame index for per-frame attention maps
                if isinstance(frame_idx, (list, tuple)) and len(frame_idx) > 0:
                    # If it's a list, use the first element (should be single frame for tracker)
                    frame_idx = frame_idx[0]
                filename = f"video_{video_id}_frame_{frame_idx}_tracker_{attention_type_str}_layer_{layer_idx}_attn"
            elif frame_start is not None and frame_end is not None:
                filename = f"video_{video_id}_frames{frame_start}-{frame_end}_tracker_{attention_type_str}_layer_{layer_idx}_attn"
            else:
                filename = f"video_{video_id}_unknown_tracker_{attention_type_str}_layer_{layer_idx}_attn"
            
            # Add video context to entry
            entry['video_id'] = video_id
            entry['frame_idx'] = frame_idx
            entry['frame_start'] = frame_start
            entry['frame_end'] = frame_end
            entry['window_idx'] = window_idx
            
            # Save synchronously to disk
            if self._extractor is not None:
                print(f"[TRACKER_HOOK] Saving {filename} to {attn_dir}", flush=True)
                self._extractor._save_synchronously(entry, filename, attn_dir)
                print(f"[TRACKER_HOOK] Successfully saved {filename}", flush=True)
        except Exception as e:
            import logging
            import traceback
            error_msg = f"Failed to save tracker attention from hook: {e}\n{traceback.format_exc()}"
            logging.getLogger(__name__).warning(error_msg)
            print(f"[TRACKER_HOOK] ERROR: {error_msg}", flush=True)


class PredictorCrossAttentionHook:
    """Hook to capture cross-attention weights from predictor (transformer decoder) MultiheadAttention layers."""
    
    def __init__(self, layer_name, layer_idx, attention_module, sink_list=None, extractor=None):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.attention_module = attention_module
        self.attention_maps = []
        self._sink_list = sink_list
        self._extractor = extractor  # Reference to extractor for immediate saving
        self._feature_level = None  # Will be set by predictor forward hook
        self._spatial_shape = None  # Will be set by predictor forward hook (H_feat, W_feat)
    
    def __call__(self, module, input, output):
        """Capture attention weights from the forward pass."""
        if isinstance(output, tuple) and len(output) >= 2:
            # MultiheadAttention returns (attn_output, attn_output_weights)
            attn_output, attn_weights = output[0], output[1]
            
            if attn_weights is not None:
                # Convert to numpy immediately to free GPU memory
                # Shape: (batch*num_heads, num_queries, H*W) or (num_queries, H*W) depending on batching
                # We need to handle both cases
                attn_weights_np = attn_weights.detach().cpu().numpy()
                
                # Get feature level and spatial shape from extractor (set by predictor forward hook)
                feature_level = None
                spatial_shape = None
                if self._extractor is not None:
                    predictor_info = getattr(self._extractor, '_current_predictor_info', None)
                    if predictor_info is not None:
                        # Calculate feature level: level_index = layer_idx % num_feature_levels
                        num_feature_levels = predictor_info.get('num_feature_levels', 3)
                        feature_level = self.layer_idx % num_feature_levels
                        # Get spatial shape for this feature level
                        size_list = predictor_info.get('size_list', None)
                        if size_list is not None and feature_level < len(size_list):
                            spatial_shape = tuple(size_list[feature_level])  # (H_feat, W_feat)
                
                # Determine actual shape dimensions
                if len(attn_weights_np.shape) == 3:
                    # Shape: (batch*num_heads, num_queries, H*W)
                    batch_heads, num_queries, spatial_length = attn_weights_np.shape
                    # We can't easily separate batch and heads without additional info
                    # Store as-is and let visualization handle it
                elif len(attn_weights_np.shape) == 2:
                    # Shape: (num_queries, H*W) - single batch, single head or already averaged
                    num_queries, spatial_length = attn_weights_np.shape
                    batch_heads = 1
                else:
                    # Unexpected shape, store as-is
                    batch_heads = num_queries = spatial_length = None
                
                entry = {
                    'layer': self.layer_name,
                    'layer_idx': self.layer_idx,
                    'attention_weights': attn_weights_np,  # Store as numpy array
                    'shape': attn_weights.shape,
                    'spatial_shape': spatial_shape,  # (H_feat, W_feat) for this feature level
                    'feature_level': feature_level,  # Which of the 3 feature levels (0, 1, or 2)
                    'spatial_length': spatial_length,  # H*W
                    'num_queries': num_queries,
                    'source': 'predictor_cross_attn'
                }
                
                # Debug: print when we capture attention
                print(f"[PREDICTOR_HOOK] Captured attention from {self.layer_name}, shape: {attn_weights.shape}, save_immediately: {self._extractor.save_immediately_from_hook if self._extractor else False}", flush=True)
                
                # Save immediately from hook if enabled
                if self._extractor is not None and self._extractor.save_immediately_from_hook:
                    self._save_from_hook(entry)
                else:
                    # Store entry in sink list
                    if self._sink_list is not None:
                        self._sink_list.append(entry)
                    else:
                        self.attention_maps.append(entry)
    
    def _save_from_hook(self, entry):
        """Save attention map directly from hook without accumulating in lists."""
        if self._extractor is None:
            print(f"[PREDICTOR_HOOK] {self.layer_name} - Cannot save: extractor is None", flush=True)
            return
        if not self._extractor._output_dir:
            print(f"[PREDICTOR_HOOK] {self.layer_name} - Cannot save: output_dir is not set (value: {self._extractor._output_dir})", flush=True)
            return
        
        try:
            import os
            import json
            from detectron2.utils.file_io import PathManager
            
            # Get video context if available
            video_context = self._extractor._current_video_context
            if video_context is not None:
                video_id = video_context.get('video_id', 'unknown')
                frame_idx = video_context.get('frame_idx', None)
                frame_start = video_context.get('frame_start', None)
                frame_end = video_context.get('frame_end', None)
                window_idx = video_context.get('window_idx', None)
            else:
                video_id = 'unknown'
                frame_idx = None
                frame_start = None
                frame_end = None
                window_idx = None
            
            # Try to get original image dimensions from image_dimensions.json for mapping
            original_img_height = None
            original_img_width = None
            try:
                inference_dir = os.path.join(self._extractor._output_dir, "inference")
                dims_path = os.path.join(inference_dir, "image_dimensions.json")
                if os.path.exists(dims_path):
                    with PathManager.open(dims_path, "r") as f:
                        dims_dict = json.load(f)
                        video_key = str(video_id)
                        if video_key in dims_dict:
                            dims = dims_dict[video_key]
                            # Use original_image_height/width (after resize, before padding)
                            original_img_height = dims.get('original_image_height')
                            original_img_width = dims.get('original_image_width')
            except Exception:
                pass  # If we can't get dimensions, continue without them
            
            attn_dir = os.path.join(self._extractor._output_dir, "attention_maps")
            os.makedirs(attn_dir, exist_ok=True)
            
            # Build filename with video context
            layer_idx = entry.get('layer_idx', 'unknown')
            feature_level = entry.get('feature_level')
            
            if frame_start is not None and frame_end is not None:
                if feature_level is not None:
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_predictor_cross_attn_layer_{layer_idx}_level_{feature_level}"
                else:
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_predictor_cross_attn_layer_{layer_idx}"
            elif frame_idx is not None:
                if feature_level is not None:
                    filename = f"video_{video_id}_frame_{frame_idx}_predictor_cross_attn_layer_{layer_idx}_level_{feature_level}"
                else:
                    filename = f"video_{video_id}_frame_{frame_idx}_predictor_cross_attn_layer_{layer_idx}"
            else:
                if feature_level is not None:
                    filename = f"video_{video_id}_unknown_predictor_cross_attn_layer_{layer_idx}_level_{feature_level}"
                else:
                    filename = f"video_{video_id}_unknown_predictor_cross_attn_layer_{layer_idx}"
            
            # Add video context to entry
            entry['video_id'] = video_id
            entry['frame_idx'] = frame_idx
            entry['frame_start'] = frame_start
            entry['frame_end'] = frame_end
            entry['window_idx'] = window_idx
            
            # Add image dimension info for mapping
            if original_img_height is not None and original_img_width is not None:
                entry['original_image_height'] = original_img_height
                entry['original_image_width'] = original_img_width
                
                # Compute feature stride if we have spatial_shape
                spatial_shape = entry.get('spatial_shape')
                if spatial_shape is not None and len(spatial_shape) == 2:
                    H_feat, W_feat = spatial_shape
                    # Compute stride (downsampling factor)
                    feature_stride_h = original_img_height / H_feat if H_feat > 0 else None
                    feature_stride_w = original_img_width / W_feat if W_feat > 0 else None
                    # They should be approximately equal, use average
                    if feature_stride_h is not None and feature_stride_w is not None:
                        feature_stride = (feature_stride_h + feature_stride_w) / 2.0
                        entry['feature_stride'] = float(feature_stride)
                        entry['feature_stride_h'] = float(feature_stride_h)
                        entry['feature_stride_w'] = float(feature_stride_w)
            
            # Save synchronously to disk
            if self._extractor is not None:
                print(f"[PREDICTOR_HOOK] Saving {filename} to {attn_dir}", flush=True)
                self._extractor._save_synchronously(entry, filename, attn_dir)
                print(f"[PREDICTOR_HOOK] Successfully saved {filename}", flush=True)
        except Exception as e:
            import logging
            import traceback
            error_msg = f"Failed to save predictor cross-attention from hook: {e}\n{traceback.format_exc()}"
            logging.getLogger(__name__).warning(error_msg)
            print(f"[PREDICTOR_HOOK] ERROR: {error_msg}", flush=True)


class AttentionExtractor:
    """Class to manage attention extraction hooks and data saving."""
    
    def __init__(self, model, output_dir, max_accumulated_maps=100, 
                 extract_refiner=True, save_immediately_from_hook=False,
                 window_size=None, extract_backbone=None):
        """
        Args:
            model: The model to extract attention from
            output_dir: Directory to save attention maps
            max_accumulated_maps: Maximum attention maps to accumulate before warning
            extract_refiner: If False, skip refiner attention extraction
            save_immediately_from_hook: If True, save directly from hook without accumulating in lists.
                                        This prevents RAM accumulation but requires thread-safe file writing.
            window_size: Window size for video processing
            extract_backbone: Deprecated parameter (kept for compatibility, ignored)
        """
        self.model = model
        self.output_dir = output_dir
        self.num_layers = 6  # DVIS-DAQ refiner has 6 layers
        self.refiner_hooks = []  # Hooks for refiner temporal attention
        self.predictor_hooks = []  # Hooks for predictor cross-attention
        self.tracker_hooks = []  # Hooks for tracker attention
        self.refiner_attention_data = []  # Refiner attention data
        self.refiner_attention_maps = []  # global sink for refiner temporal attention maps
        self.predictor_attention_maps = []  # global sink for predictor cross-attention maps
        self.tracker_attention_maps = []  # global sink for tracker attention maps
        self._current_predictor_info = None  # Store predictor forward info (size_list, num_feature_levels)
        self._refiner_id_to_seq_id_maps = {}  # Store refiner_id -> sequence_id mapping: {video_id: {window_key: [seq_id_0, seq_id_1, ...]}}
        self.video_attention_data = {}  # video_id -> attention_data
        self.current_video_id = None
        self.video_pred_info = None  # Store prediction info for the current video
        self._save_counter = {}  # Track save counter per video to append incrementally
        self._output_dir = output_dir
        self._save_immediately = True  # Save immediately to avoid accumulation
        self._max_accumulated_maps = max_accumulated_maps  # Max entries before forced clear
        self._current_accumulated = 0  # Track current accumulation
        self.extract_refiner = extract_refiner
        self.save_immediately_from_hook = save_immediately_from_hook  # Save directly from hook
        self._current_video_context = None  # Current video_id/frame_idx for hooks to use
        self._window_counter_per_video = {}  # Track window index per video for frame range calculation
        self._window_size = window_size  # Store window_size from config (or None to get from model)
        self._total_frames_per_video = {}  # Track total frames per video (calculated once)
        self._total_windows_per_video = {}  # Track total windows per video (calculated once)
        self._current_tracker_frame_idx = None  # Current frame index within tracker inference loop
        self._tracker_start_frame_id = None  # Start frame ID for current tracker inference call
        self._tracker_frame_counter = {}  # Track frame counter per video/window
        
        # Register hooks for refiner attention layers and predictor cross-attention
        self._register_hooks()
        
        # Register a pre-forward hook on the model to set video context before forward pass
        self._model_forward_hook = self.model.register_forward_pre_hook(self._model_forward_pre_hook)
        
        # Register a pre-forward hook on the backbone to update window context each time it's called
        # The backbone is called once per window in segmenter_windows_inference, so this hook will fire
        # for each window, allowing us to track the window index correctly
        if hasattr(self.model, 'backbone'):
            self._backbone_forward_hook = self.model.backbone.register_forward_pre_hook(self._backbone_forward_pre_hook)
        
        # Register a forward hook on the predictor to capture spatial dimensions
        if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'predictor'):
            self._predictor_forward_hook = self.model.sem_seg_head.predictor.register_forward_hook(self._predictor_forward_hook)
        
        # Hook into common_inference to capture final sequence ID mappings
        # This is where the final pred_ids are created from seq_id_list
        # We need to capture the seq_id_list order to map refiner_id (index) -> sequence_id
        if hasattr(self.model, 'common_inference'):
            original_common_inference = self.model.common_inference
            
            def hooked_common_inference(*args, **kwargs):
                # Call original method - this builds seq_id_list and stores it in common_out
                result = original_common_inference(*args, **kwargs)
                
                # Capture seq_id_list from the return value (common_out)
                # This is the order that determines refiner_id -> sequence_id mapping
                if isinstance(result, dict) and 'seq_id_list' in result:
                    seq_id_list = result['seq_id_list']
                    self._store_refiner_id_mapping(seq_id_list)
                
                return result
            
            self.model.common_inference = hooked_common_inference
            self._common_inference_hook = (self.model, original_common_inference)
        else:
            self._common_inference_hook = None
        
        # Hook into tracker's inference method to track frame index
        if hasattr(self.model, 'tracker') and hasattr(self.model.tracker, 'inference'):
            original_tracker_inference = self.model.tracker.inference
            
            def hooked_tracker_inference(frame_embeds, mask_features, frames_info, start_frame_id, resume=False, to_store="cpu"):
                # Store the start_frame_id for this inference call
                # IMPORTANT: Set this BEFORE calling original_tracker_inference so the frame tracker hook can read it
                # start_frame_id is the absolute frame index where this window starts (e.g., 0 for window 0, 100 for window 1)
                self._tracker_start_frame_id = start_frame_id
                
                # Get video_id from current context
                video_id = None
                if self._current_video_context is not None:
                    video_id = self._current_video_context.get('video_id')
                
                print(f"[TRACKER_INFERENCE_HOOK] Called with start_frame_id={start_frame_id}, video_id={video_id}, resume={resume}", flush=True)
                
                # Create a unique key for this inference call (video_id + start_frame_id)
                # Use a global counter key instead of per-window key, since frames should be numbered 0, 1, 2, ... across all windows
                if video_id is not None:
                    # Use video_id only as the key, so frame counter accumulates across windows
                    inference_key = f"{video_id}_global"
                    # Initialize or continue the global counter
                    # IMPORTANT: The tracker processes all frames from 0 regardless of which window is being processed
                    # So we always start the counter at -1 (so first frame becomes 0), not at start_frame_id - 1
                    if inference_key not in self._tracker_frame_counter:
                        # First time seeing this video - always start counter at -1
                        # This ensures the first frame gets index 0, regardless of which window triggers the tracker
                        self._tracker_frame_counter[inference_key] = -1
                        print(f"[TRACKER_INFERENCE_HOOK] Initializing global frame counter for {inference_key}, starting at 0 (tracker processes all frames from beginning)", flush=True)
                    else:
                        # Counter already exists - continue counting from where we left off
                        current_counter = self._tracker_frame_counter[inference_key]
                        print(f"[TRACKER_INFERENCE_HOOK] Continuing global frame counter for {inference_key}, current value: {current_counter}, next frame will be {current_counter + 1}", flush=True)
                
                # Call original inference - the frame tracking will happen via the cross-attention hooks
                # which will check self._current_tracker_frame_idx
                result = original_tracker_inference(frame_embeds, mask_features, frames_info, start_frame_id, resume, to_store)
                
                # Clear frame tracking after inference completes
                # Don't clear _current_tracker_frame_idx here - it's used by attention hooks that fire after inference
                # Don't clear _tracker_start_frame_id here either - it might be needed for subsequent frames
                # Only clear them when a new video starts (handled in model forward hook)
                # Don't delete the global counter - it should persist across windows for the same video
                print(f"[TRACKER_INFERENCE_HOOK] Completed inference for start_frame_id={start_frame_id}, global counter value: {self._tracker_frame_counter.get(f'{video_id}_global', 'N/A')}", flush=True)
                
                return result
            
            self.model.tracker.inference = hooked_tracker_inference
            self._tracker_inference_hook = (self.model.tracker, original_tracker_inference)
        else:
            self._tracker_inference_hook = None
    
    def _model_forward_pre_hook(self, module, input):
        """Pre-forward hook on model to extract video_id from inputs and set context."""
        try:
            # Extract video_id from batched_inputs
            if isinstance(input, tuple) and len(input) > 0:
                batched_inputs = input[0]
                if isinstance(batched_inputs, list) and len(batched_inputs) > 0:
                    first_input = batched_inputs[0]
                    if isinstance(first_input, dict):
                        video_id = first_input.get("video_id")
                        frame_idx = first_input.get("frame_idx", None)
                        
                        # Get window_size from config (if provided), otherwise from model, otherwise default to 30
                        window_size = self._window_size
                        if window_size is None:
                            window_size = getattr(module, 'window_size', None)
                        if window_size is None:
                            window_size = 30  # Final fallback default
                        
                        # Set initial video context (window tracking is done in backbone hook)
                        # The model forward hook only extracts video_id, window tracking happens
                        # in the backbone forward hook which fires once per window
                        if video_id is not None:
                            # Calculate and store total frames and windows once per video
                            if "image" in first_input:
                                num_frames = len(first_input["image"])
                                if num_frames is not None and video_id not in self._total_frames_per_video:
                                    self._total_frames_per_video[video_id] = num_frames
                                    # Calculate total windows (ceiling of num_frames/window_size)
                                    self._total_windows_per_video[video_id] = math.ceil(num_frames / window_size)
                            
                            self.set_video_context(
                                video_id=video_id, 
                                frame_idx=frame_idx,
                                frame_start=None,  # Will be set by backbone hook
                                frame_end=None,    # Will be set by backbone hook
                                window_idx=None,   # Will be set by backbone hook
                                forward_pass_id=None
                            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not extract video_id from model input: {e}")
    
    def _backbone_forward_pre_hook(self, module, input):
        """Pre-forward hook on backbone to update window context each time it's called.
        
        The backbone is called once per window in segmenter_windows_inference,
        so this hook fires for each window, allowing us to track the window index correctly.
        """
        try:
            # Get video_id from the current context (set by model forward hook)
            video_id = None
            if self._current_video_context is not None:
                video_id = self._current_video_context.get('video_id')
            
            if video_id is None:
                return
            
            # Get window_size
            window_size = self._window_size
            if window_size is None:
                window_size = getattr(module, 'window_size', None)
            if window_size is None:
                window_size = 30  # Final fallback default
            
            # Get actual number of frames in this window's input tensor
            num_frames_in_this_window = None
            if isinstance(input, tuple) and len(input) > 0:
                input_tensor = input[0]
                if hasattr(input_tensor, 'shape') and len(input_tensor.shape) >= 2:
                    # Input is likely (B, T, C, H, W) or similar
                    num_frames_in_this_window = input_tensor.shape[1] if len(input_tensor.shape) == 5 else input_tensor.shape[0]
            
            # Initialize window counter if needed (new video)
            if video_id not in self._window_counter_per_video:
                self._window_counter_per_video[video_id] = -1
            
            # Increment window index (each backbone forward call is a new window)
            self._window_counter_per_video[video_id] += 1
            
            # Get current window index
            window_idx = self._window_counter_per_video[video_id]
            
            # Calculate frame range for this window
            # frame_start and frame_end are absolute frame indices in the video (inclusive)
            # Window 0: frames 0-30 (31 frames)
            # Window 1: frames 31-61 (31 frames)
            # Window 2: frames 62-92 (31 frames)
            frame_start = window_idx * window_size
            frame_end = (window_idx + 1) * window_size - 1  # Inclusive end, so subtract 1
            
            # Cap frame_end based on total frames in video (for last window which may be shorter)
            total_frames = self._total_frames_per_video.get(video_id)
            if total_frames is not None:
                # frame_end is inclusive, so max is total_frames - 1
                frame_end = min(frame_end, total_frames - 1)
            
            # Debug: print window info
            total_windows = self._total_windows_per_video.get(video_id, None)
            total_info = ""
            if total_frames is not None and total_windows is not None:
                total_info = f", total_frames={total_frames}, total_windows={total_windows}"
            print(f"[WINDOW] video_id={video_id}, window_idx={window_idx}, frames={frame_start}-{frame_end}, num_frames_in_this_window={num_frames_in_this_window}{total_info}", flush=True)
            
            # Update video context for hooks to use (with frame range info)
            if video_id is not None:
                self.set_video_context(
                    video_id=video_id, 
                    frame_idx=self._current_video_context.get('frame_idx') if self._current_video_context else None,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    window_idx=window_idx,
                    forward_pass_id=None
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not update window context in backbone hook: {e}")
    
    def _predictor_forward_hook(self, module, input, output):
        """Forward hook on predictor to capture spatial dimensions (size_list) for each feature level.
        
        The predictor forward method receives multi_scale_features (x) as a list of tensors
        with shape (batch, channels, H, W). We extract size_list from x[i].shape[-2:] 
        (same as the predictor does internally at line 374).
        """
        try:
            # The predictor forward receives x (list of multi-scale features) as first argument
            # Format: x is a list of tensors with shape (batch, channels, H, W)
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]  # x is a list of multi-scale feature tensors
                if isinstance(x, (list, tuple)) and len(x) > 0:
                    size_list = []
                    for i, feat in enumerate(x):
                        if hasattr(feat, 'shape') and len(feat.shape) >= 2:
                            # Features should be in format (batch, channels, H, W) before flattening
                            if len(feat.shape) == 4:
                                # (batch, channels, H, W) - extract H, W from last 2 dimensions
                                H, W = feat.shape[-2:]
                                size_list.append((H, W))
                            elif len(feat.shape) == 3:
                                # (H*W, batch, channels) - already flattened, can't get H, W
                                # This shouldn't happen if hook is called at the right time
                                spatial_length, batch, channels = feat.shape
                                size_list.append(None)
                                print(f"[PREDICTOR_HOOK] Warning: Feature {i} is already flattened (shape: {feat.shape}), cannot extract H, W")
                            else:
                                size_list.append(None)
                                print(f"[PREDICTOR_HOOK] Warning: Feature {i} has unexpected shape: {feat.shape}")
                        else:
                            size_list.append(None)
                    
                    num_feature_levels = len(size_list)
                    
                    # Only store predictor info if we got valid dimensions
                    if num_feature_levels > 0 and any(s is not None for s in size_list):
                        self._current_predictor_info = {
                            'size_list': size_list,
                            'num_feature_levels': num_feature_levels
                        }
                        # Debug: print when we successfully capture
                        print(f"[PREDICTOR_HOOK] Captured size_list: {size_list}")
                    else:
                        # Debug: print when we fail to capture
                        print(f"[PREDICTOR_HOOK] Failed to capture valid size_list. Input type: {type(x)}, length: {len(x) if isinstance(x, (list, tuple)) else 'N/A'}")
                        if isinstance(x, (list, tuple)) and len(x) > 0:
                            print(f"[PREDICTOR_HOOK] First feature shape: {x[0].shape if hasattr(x[0], 'shape') else 'N/A'}")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not capture predictor info: {e}")
            # Also print for debugging
            print(f"[PREDICTOR_HOOK] Error capturing predictor info: {e}")
    
    
    
    def _get_tracker_id_mapping(self):
        """Get the tracker_id -> sequence_id mapping from the tracker.
        
        When cross-attention is computed, the queries are: trc_det_queries = cat([track_queries, new_ins_embeds])
        - The first len(track_queries) queries correspond to last_seq_ids (from previous frame)
        - The remaining queries are new instances that don't have seq_ids yet
        
        Returns:
            dict or None: Mapping where:
                - Keys are tracker query indices (0, 1, 2, ...)
                - Values are sequence_ids (int) for track queries, or None for new queries
                - Also includes 'num_track_queries' key indicating how many queries have seq_ids
        """
        try:
            if not hasattr(self.model, 'tracker'):
                return None
            
            tracker = self.model.tracker
            if not hasattr(tracker, 'last_seq_ids') or tracker.last_seq_ids is None:
                return None
            
            # Get the number of track queries (these have sequence IDs)
            num_track_queries = len(tracker.last_seq_ids) if tracker.last_seq_ids is not None else 0
            
            # Build mapping: tracker_id (index) -> sequence_id
            # For track queries (indices 0 to num_track_queries-1), we have seq_ids
            # For new queries (beyond num_track_queries), we don't have seq_ids yet
            mapping = {}
            if tracker.last_seq_ids is not None:
                for i, seq_id in enumerate(tracker.last_seq_ids):
                    mapping[i] = seq_id
            
            # Also store the number of track queries for reference
            mapping['num_track_queries'] = num_track_queries
            
            return mapping
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not get tracker_id mapping: {e}")
            return None
    
    def _store_refiner_id_mapping(self, seq_id_list):
        """Store the refiner_id -> sequence_id mapping for the current window."""
        try:
            video_context = self._current_video_context
            if video_context is None:
                return
            
            video_id = video_context.get('video_id')
            frame_start = video_context.get('frame_start')
            frame_end = video_context.get('frame_end')
            
            if video_id is None:
                return
            
            if video_id not in self._refiner_id_to_seq_id_maps:
                self._refiner_id_to_seq_id_maps[video_id] = {}
            
            # Use frame range as key
            if frame_start is not None and frame_end is not None:
                window_key = f"frames_{frame_start}_{frame_end}"
            else:
                window_key = "unknown"
            
            # Store the mapping: refiner_id (index) -> sequence_id
            # seq_id_list[i] is the sequence_id for refiner_id = i
            self._refiner_id_to_seq_id_maps[video_id][window_key] = (
                seq_id_list if isinstance(seq_id_list, list) else seq_id_list.tolist()
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not store refiner_id mapping: {e}")
    
    
    def _register_hooks(self):
        """Register forward hooks to capture attention maps."""
        # Register refiner temporal self-attention hooks
        if self.extract_refiner:
            for name, module in self.model.named_modules():
                if 'transformer_time_self_attention_layers' in name and hasattr(module, 'self_attn'):
                    hook = RefinerAttentionHook(name, module.self_attn, sink_list=self.refiner_attention_maps, extractor=self)
                    module.self_attn.register_forward_hook(hook)
                    self.refiner_hooks.append(hook)
        else:
            print("Skipping refiner attention extraction (disabled)")
        
        # Register predictor cross-attention hooks
        if hasattr(self.model, 'sem_seg_head') and hasattr(self.model.sem_seg_head, 'predictor'):
            predictor = self.model.sem_seg_head.predictor
            if hasattr(predictor, 'transformer_cross_attention_layers'):
                predictor_layer_indices = []
                for i, cross_attn_layer in enumerate(predictor.transformer_cross_attention_layers):
                    if hasattr(cross_attn_layer, 'multihead_attn'):
                        layer_name = f"sem_seg_head.predictor.transformer_cross_attention_layers.{i}"
                        hook = PredictorCrossAttentionHook(
                            layer_name, i, cross_attn_layer.multihead_attn,
                            sink_list=self.predictor_attention_maps,
                            extractor=self
                        )
                        cross_attn_layer.multihead_attn.register_forward_hook(hook)
                        self.predictor_hooks.append(hook)
                        predictor_layer_indices.append(i)
                        print(f"Registered predictor cross-attention hook for layer {i}")
                
                print(f"Extracting attention from {len(predictor_layer_indices)} predictor cross-attention layers: {predictor_layer_indices}")
            else:
                print("Warning: Could not find transformer_cross_attention_layers in predictor")
        else:
            print("Skipping predictor cross-attention extraction (predictor not found)")
        
        # Register tracker cross-attention hooks only
        if hasattr(self.model, 'tracker'):
            tracker = self.model.tracker
            tracker_layer_indices = []
            print(f"[ATTENTION_EXTRACTOR] Found tracker: {type(tracker)}")
            
            # Register tracker cross-attention hooks only
            if hasattr(tracker, 'transformer_cross_attention_layers'):
                num_layers = len(tracker.transformer_cross_attention_layers)
                print(f"[ATTENTION_EXTRACTOR] Found {num_layers} tracker cross-attention layers")
                for i, cross_attn_layer in enumerate(tracker.transformer_cross_attention_layers):
                    if hasattr(cross_attn_layer, 'multihead_attn'):
                        layer_name = f"tracker.transformer_cross_attention_layers.{i}"
                        hook = TrackerAttentionHook(
                            layer_name, i, 'cross', cross_attn_layer.multihead_attn,
                            sink_list=self.tracker_attention_maps,
                            extractor=self
                        )
                        cross_attn_layer.multihead_attn.register_forward_hook(hook)
                        
                        # Add pre-forward hook on layer 0 to track frame index
                        # Layer 0 is called once per frame, so we can use it to increment frame counter
                        if i == 0:
                            def make_frame_tracker(layer_idx):
                                def frame_tracker_hook(module, input):
                                    # Get video_id and start_frame_id
                                    video_id = None
                                    start_frame_id = None
                                    if self._current_video_context is not None:
                                        video_id = self._current_video_context.get('video_id')
                                    start_frame_id = self._tracker_start_frame_id
                                    
                                    # If start_frame_id is None, try to get it from frame_start in video context
                                    # This is a fallback for when the inference hook hasn't been called yet
                                    if start_frame_id is None and self._current_video_context is not None:
                                        frame_start = self._current_video_context.get('frame_start')
                                        if frame_start is not None:
                                            # Use frame_start as start_frame_id (this is the window start)
                                            start_frame_id = frame_start
                                            print(f"[TRACKER_FRAME_TRACKER] Using frame_start={frame_start} as start_frame_id fallback", flush=True)
                                    
                                    if video_id is not None:
                                        # Use global counter key (across all windows) instead of per-window key
                                        inference_key = f"{video_id}_global"
                                        
                                        # Initialize counter if it doesn't exist (should be initialized in inference hook, but safety check)
                                        # IMPORTANT: Always start at -1, not start_frame_id - 1, because tracker processes all frames from 0
                                        if inference_key not in self._tracker_frame_counter:
                                            self._tracker_frame_counter[inference_key] = -1
                                            print(f"[TRACKER_FRAME_TRACKER] Initializing global frame counter for {inference_key}, starting at 0 (tracker processes all frames from beginning)", flush=True)
                                        
                                        # Increment frame counter (layer 0 is called once per frame)
                                        self._tracker_frame_counter[inference_key] += 1
                                        absolute_frame_idx = self._tracker_frame_counter[inference_key]
                                        
                                        # Calculate frame index within window for debugging (not used for filename, just for logging)
                                        frame_idx_in_window = absolute_frame_idx - start_frame_id if start_frame_id is not None else absolute_frame_idx
                                        
                                        # Update video context with current frame index
                                        if self._current_video_context is not None:
                                            self._current_video_context['frame_idx'] = absolute_frame_idx
                                        # Also update the tracker frame index (used by attention hooks)
                                        self._current_tracker_frame_idx = absolute_frame_idx
                                        
                                        print(f"[TRACKER_FRAME_TRACKER] Layer 0 pre-forward hook: frame_idx_in_window={frame_idx_in_window}, absolute_frame_idx={absolute_frame_idx}, start_frame_id={start_frame_id}", flush=True)
                                    else:
                                        print(f"[TRACKER_FRAME_TRACKER] Warning: Missing video_id or start_frame_id. video_id={video_id}, start_frame_id={start_frame_id}, frame_start={self._current_video_context.get('frame_start') if self._current_video_context else None}", flush=True)
                                return frame_tracker_hook
                            
                            cross_attn_layer.multihead_attn.register_forward_pre_hook(make_frame_tracker(0))
                            print(f"[ATTENTION_EXTRACTOR] Registered frame tracker pre-forward hook on layer 0", flush=True)
                        
                        self.tracker_hooks.append(hook)
                        tracker_layer_indices.append(i)
                        print(f"[ATTENTION_EXTRACTOR] Registered tracker cross-attention hook for layer {i}")
                    else:
                        print(f"[ATTENTION_EXTRACTOR] Warning: Layer {i} does not have multihead_attn attribute")
                
                if len(tracker_layer_indices) > 0:
                    print(f"[ATTENTION_EXTRACTOR] Successfully registered {len(tracker_layer_indices)} tracker cross-attention hooks: {tracker_layer_indices}")
                else:
                    print("[ATTENTION_EXTRACTOR] Warning: Could not find tracker cross-attention layers to hook")
            else:
                print("[ATTENTION_EXTRACTOR] Warning: Could not find transformer_cross_attention_layers in tracker")
                print(f"[ATTENTION_EXTRACTOR] Tracker attributes: {[attr for attr in dir(tracker) if not attr.startswith('_')]}")
        else:
            print("[ATTENTION_EXTRACTOR] Skipping tracker attention extraction (tracker not found)")
            print(f"[ATTENTION_EXTRACTOR] Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
    
    def set_video_context(self, video_id=None, frame_idx=None, frame_start=None, frame_end=None, window_idx=None, forward_pass_id=None):
        """Set video context for hooks to use when saving directly."""
        self._current_video_context = {
            'video_id': video_id,
            'frame_idx': frame_idx,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'window_idx': window_idx
        }
    
    def clear_video_context(self):
        """Clear video context after forward pass."""
        self._current_video_context = None
    
    def _save_synchronously(self, entry, filename, attn_dir):
        """Save attention map synchronously with progress print."""
        try:
            import os
            import numpy as np
            import json
            from detectron2.utils.file_io import PathManager
            
            arr = entry.get('attention_weights')
            if isinstance(arr, np.ndarray):
                # Define npz_path early so it's available for duplicate checking
                npz_path = os.path.join(attn_dir, f"{filename}.npz")
                
                # Print progress
                print(f"[SAVE] Saving {filename} (shape: {arr.shape})", flush=True)
                
                # Save attention map
                np.savez_compressed(npz_path, attention_weights=arr)
                
                # Save metadata
                meta = {
                    'source': entry.get('source', 'predictor_cross_attn'),
                    'layer': entry.get('layer'),
                    'shape': entry.get('shape'),
                    'spatial_shape': entry.get('spatial_shape'),
                    'num_heads': entry.get('num_heads'),
                    'sequence_length': entry.get('sequence_length'),
                    'layer_idx': entry.get('layer_idx'),
                    'feature_level': entry.get('feature_level'),
                    'spatial_length': entry.get('spatial_length'),
                    'num_queries': entry.get('num_queries'),
                    'original_image_height': entry.get('original_image_height'),
                    'original_image_width': entry.get('original_image_width'),
                    'feature_stride': entry.get('feature_stride'),
                    'feature_stride_h': entry.get('feature_stride_h'),
                    'feature_stride_w': entry.get('feature_stride_w'),
                    'video_id': entry.get('video_id'),
                    'frame_idx': entry.get('frame_idx'),
                    'frame_start': entry.get('frame_start'),
                    'frame_end': entry.get('frame_end'),
                    'window_idx': entry.get('window_idx'),
                    'attention_type': entry.get('attention_type'),  # For tracker attention
                    'tracker_id_to_seq_id': entry.get('tracker_id_to_seq_id')  # Mapping for tracker attention
                }
                meta_path = os.path.join(attn_dir, f"{filename}.meta.json")
                with PathManager.open(meta_path, "w") as mf:
                    mf.write(json.dumps(meta))
                    mf.flush()
                
                # Free memory
                del arr
                del entry
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save synchronously: {e}")
    
    def clear_attention_maps(self):
        """Clear all captured attention maps and free GPU memory."""
        # Explicitly delete entries to free numpy arrays
        for entry in self.refiner_attention_maps:
            if 'attention_weights' in entry:
                del entry['attention_weights']
        for entry in self.predictor_attention_maps:
            if 'attention_weights' in entry:
                del entry['attention_weights']
        
        self.refiner_attention_maps.clear()
        self.predictor_attention_maps.clear()
        self.tracker_attention_maps.clear()
        
        for hook in self.refiner_hooks:
            for entry in hook.refiner_attention_maps:
                if 'attention_weights' in entry:
                    del entry['attention_weights']
            hook.refiner_attention_maps.clear()
        
        for hook in self.predictor_hooks:
            for entry in hook.attention_maps:
                if 'attention_weights' in entry:
                    del entry['attention_weights']
            hook.attention_maps.clear()
        
        for hook in self.tracker_hooks:
            for entry in hook.attention_maps:
                if 'attention_weights' in entry:
                    del entry['attention_weights']
            hook.attention_maps.clear()
        
        # Force memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_attention_maps_immediately(self, video_id=None, frame_idx=None, force_save=False):
        """Save attention maps immediately to disk to avoid RAM accumulation."""
        if not self._output_dir:
            return
        
        attn_dict = self.get_attention_maps()
        # Check if there's anything to save
        # Note: when save_immediately_from_hook=True, predictor_cross_attn will be empty
        # because it is saved directly from hooks
        has_refiner = attn_dict.get('refiner_temporal') and len(attn_dict.get('refiner_temporal', [])) > 0
        has_predictor = attn_dict.get('predictor_cross_attn') and len(attn_dict.get('predictor_cross_attn', [])) > 0
        if not has_refiner and not has_predictor:
            return
        
        try:
            import os
            import numpy as np
            import json
            from detectron2.utils.file_io import PathManager
            
            attn_dir = os.path.join(self._output_dir, "attention_maps")
            os.makedirs(attn_dir, exist_ok=True)
            
            # Use video_id if provided, otherwise use a temporary identifier
            if video_id is None:
                video_id = "temp"
            
            # Process and save attention maps
            arrays = {}
            meta = []
            idx = 0
            
            # Refiner attention (temporal)
            refiner_count = 0
            for entry in attn_dict.get('refiner_temporal', []):
                arr = entry.get('attention_weights')
                key = f"attn_{idx}"
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().numpy()
                elif isinstance(arr, np.ndarray):
                    pass
                arrays[key] = arr
                meta.append({
                    'source': 'refiner',
                    'key': key,
                    'layer': entry.get('layer'),
                    'shape': entry.get('shape'),
                })
                idx += 1
                refiner_count += 1
            if refiner_count > 0:
                print(f"[SAVE] Saving {refiner_count} refiner attention maps for video {video_id}, frame {frame_idx}", flush=True)
            
            # Predictor cross-attention
            predictor_count = 0
            for entry in attn_dict.get('predictor_cross_attn', []):
                arr = entry.get('attention_weights')
                key = f"attn_{idx}"
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().numpy()
                elif isinstance(arr, np.ndarray):
                    pass
                arrays[key] = arr
                meta.append({
                    'source': 'predictor_cross_attn',
                    'key': key,
                    'layer': entry.get('layer'),
                    'layer_idx': entry.get('layer_idx'),
                    'shape': entry.get('shape'),
                    'spatial_shape': entry.get('spatial_shape'),
                    'feature_level': entry.get('feature_level'),
                    'spatial_length': entry.get('spatial_length'),
                    'num_queries': entry.get('num_queries'),
                })
                idx += 1
                predictor_count += 1
            if predictor_count > 0:
                print(f"[SAVE] Saving {predictor_count} predictor cross-attention maps for video {video_id}, frame {frame_idx}", flush=True)
            
            # Build filename based on what we're saving
            # If saving refiner (temporal) attention only, add "_temporal_refiner_attn" suffix
            if has_refiner:
                # Only refiner (temporal) attention - add "_temporal_refiner_attn" suffix
                if frame_idx is None:
                    frame_counter = self._save_counter.get(video_id, 0)
                    self._save_counter[video_id] = frame_counter + 1
                    filename = f"video_{video_id}_frame_{frame_counter}_temporal_refiner_attn"
                elif isinstance(frame_idx, (list, tuple)) and len(frame_idx) > 0:
                    frame_start = min(frame_idx)
                    frame_end = max(frame_idx)
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_temporal_refiner_attn"
                else:
                    filename = f"video_{video_id}_frame_{frame_idx}_temporal_refiner_attn"
            else:
                # Predictor cross-attention only
                if frame_idx is None:
                    frame_counter = self._save_counter.get(video_id, 0)
                    self._save_counter[video_id] = frame_counter + 1
                    filename = f"video_{video_id}_frame_{frame_counter}_predictor_cross_attn"
                elif isinstance(frame_idx, (list, tuple)) and len(frame_idx) > 0:
                    frame_start = min(frame_idx)
                    frame_end = max(frame_idx)
                    filename = f"video_{video_id}_frames{frame_start}-{frame_end}_predictor_cross_attn"
                else:
                    filename = f"video_{video_id}_frame_{frame_idx}_predictor_cross_attn"
            
            # Save immediately
            if arrays:
                npz_path = os.path.join(attn_dir, f"{filename}.npz")
                try:
                    np.savez_compressed(npz_path, **arrays)
                except Exception:
                    np.savez(npz_path, **arrays)
                
                meta_path = os.path.join(attn_dir, f"{filename}.meta.json")
                with PathManager.open(meta_path, "w") as mf:
                    mf.write(json.dumps(meta))
                    mf.flush()
            
            # Clear memory immediately
            del arrays, meta
        
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save attention maps immediately: {e}")
        
        # Always clear after saving to free RAM immediately
        self.clear_attention_maps()
    
    def get_attention_maps(self):
        """Get all captured attention maps (refiner, predictor, and tracker)."""
        return {
            'refiner_temporal': self.refiner_attention_maps,
            'predictor_cross_attn': self.predictor_attention_maps,
            'tracker': self.tracker_attention_maps
        }
    
    def save_refiner_id_mappings(self, video_id=None, incremental=False):
        """Save refiner_id → sequence_id mappings to disk.
        
        Args:
            video_id: If provided, only save maps for this specific video (incremental mode)
            incremental: If True, append to existing file instead of overwriting
        
        Saves:
            refiner_id_to_seq_id_maps: refiner_id (index) → sequence_id mappings per window
        """
        if not self._output_dir:
            return
        
        # Only save if we have refiner_id mappings
        if not self._refiner_id_to_seq_id_maps:
            return
        
        try:
            import os
            import json
            from detectron2.utils.file_io import PathManager
            
            attn_dir = os.path.join(self._output_dir, "attention_maps")
            os.makedirs(attn_dir, exist_ok=True)
            
            tracking_path = os.path.join(attn_dir, "refiner_id_mappings.json")
            
            # Load existing data if incremental mode
            existing_dict = {}
            if incremental and os.path.exists(tracking_path):
                try:
                    with PathManager.open(tracking_path, "r") as f:
                        existing_dict = json.load(f)
                except Exception:
                    pass  # If we can't read existing file, start fresh
            
            tracking_dict = existing_dict.copy() if incremental else {}
            
            # Save refiner_id → sequence_id mappings
            # Structure: {"refiner_id_to_seq_id_maps": {video_id: {window_key: [seq_id_0, seq_id_1, ...]}}}
            videos_to_save_refiner = []
            if self._refiner_id_to_seq_id_maps:
                if '_refiner_id_to_seq_id_maps' not in tracking_dict:
                    tracking_dict['_refiner_id_to_seq_id_maps'] = {}
                
                videos_to_save_refiner = [video_id] if video_id is not None else list(self._refiner_id_to_seq_id_maps.keys())
                for vid in videos_to_save_refiner:
                    if vid not in self._refiner_id_to_seq_id_maps:
                        continue
                    window_maps = self._refiner_id_to_seq_id_maps[vid]
                    if str(vid) not in tracking_dict['_refiner_id_to_seq_id_maps']:
                        tracking_dict['_refiner_id_to_seq_id_maps'][str(vid)] = {}
                    for window_key, seq_id_list in window_maps.items():
                        tracking_dict['_refiner_id_to_seq_id_maps'][str(vid)][window_key] = seq_id_list
            
            with PathManager.open(tracking_path, "w") as f:
                f.write(json.dumps(tracking_dict, indent=2))
                f.flush()
            
            num_refiner_maps = len(videos_to_save_refiner) if videos_to_save_refiner else (len(self._refiner_id_to_seq_id_maps) if self._refiner_id_to_seq_id_maps else 0)
            print(
                f"[SAVE] Saved refiner_id mappings for {num_refiner_maps} video(s) to {tracking_path}",
                flush=True
            )
            
            # If saving incrementally for a specific video, clear that video's data from memory
            if incremental and video_id is not None:
                if video_id in self._refiner_id_to_seq_id_maps:
                    del self._refiner_id_to_seq_id_maps[video_id]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to save refiner_id mappings: {e}")
    
    
    def _get_sequence_id_from_refiner_id(self, video_id, refiner_id):
        """Convert refiner_id (index) to sequence_id.
        
        Args:
            video_id: Video ID
            refiner_id: Index in refiner output (0, 1, 2, ...)
        
        Returns:
            sequence_id: The sequence ID corresponding to this refiner_id, or None
        
        IMPORTANT: The seq_id_list from common_inference represents the FINAL order
        after all windows are processed. We should use the most recent/last window's
        seq_id_list, as it should contain the final ordering.
        """
        if video_id not in self._refiner_id_to_seq_id_maps:
            return None
        
        video_maps = self._refiner_id_to_seq_id_maps[video_id]
        
        if not video_maps:
            return None
        
        # The seq_id_list from common_inference is built from video_ins_hub which
        # accumulates across all windows. The final seq_id_list (from the last window)
        # should represent the final ordering used in results.json.
        # We'll use the last window's seq_id_list (most recent) as it should be the final one.
        window_keys = sorted(video_maps.keys())
        if not window_keys:
            return None
        
        # Try the last window first (should be the final ordering)
        for window_key in reversed(window_keys):
            seq_id_list = video_maps[window_key]
            if isinstance(seq_id_list, list) and refiner_id < len(seq_id_list):
                return int(seq_id_list[refiner_id])
        
        # Fallback: try any window (shouldn't be needed, but for safety)
        for window_key, seq_id_list in video_maps.items():
            if isinstance(seq_id_list, list) and refiner_id < len(seq_id_list):
                return int(seq_id_list[refiner_id])
        
        # If not found in any window, return None
        return None
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        if hasattr(self, '_model_forward_hook') and self._model_forward_hook is not None:
            self._model_forward_hook.remove()
            self._model_forward_hook = None
        if hasattr(self, '_backbone_forward_hook') and self._backbone_forward_hook is not None:
            self._backbone_forward_hook.remove()
            self._backbone_forward_hook = None
        if hasattr(self, '_predictor_forward_hook') and self._predictor_forward_hook is not None:
            self._predictor_forward_hook.remove()
            self._predictor_forward_hook = None
        if hasattr(self, '_common_inference_hook') and self._common_inference_hook is not None:
            model, original_method = self._common_inference_hook
            model.common_inference = original_method
            self._common_inference_hook = None
        if hasattr(self, '_tracker_inference_hook') and self._tracker_inference_hook is not None:
            tracker, original_method = self._tracker_inference_hook
            tracker.inference = original_method
            self._tracker_inference_hook = None
        if hasattr(self, '_run_window_hook') and self._run_window_hook is not None:
            model, original_method = self._run_window_hook
            model.run_window_inference = original_method
            self._run_window_hook = None
        self.refiner_hooks.clear()
        self.predictor_hooks.clear()
        self.tracker_hooks.clear()