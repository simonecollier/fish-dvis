import contextlib
import io
import copy
import os
import sys
import json
import logging
from collections import OrderedDict

import numpy as np
# NumPy compatibility shim for deprecated aliases used in YTVIS eval
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
import pycocotools.mask as mask_util
import torch

import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

# Ensure project root and DVIS_Plus are importable when running from attention_analysis
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DVIS_PLUS_ROOT = os.path.join(_PROJECT_ROOT, "DVIS_Plus")
for p in (_PROJECT_ROOT, _DVIS_PLUS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the project evaluator and helpers
from mask2former_video.data_video.ytvis_eval import YTVISEvaluator


def _instances_to_coco_json_video_with_refiner(inputs, outputs, attention_extractor=None):
    """
    Convert model outputs to YTVIS-style JSON entries, adding refiner_id and predictor_query_id.
    Expects outputs to contain: pred_scores, pred_labels, pred_masks, pred_ids (aligned).
    
    Args:
        inputs: Input dictionary containing video_id and optionally frame_idx
        outputs: Model outputs containing predictions
        attention_extractor: Optional AttentionExtractor to look up predictor query IDs
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_id"]
    # Try to get frame_idx from inputs, but it might not be available for window-based inference
    frame_idx = inputs[0].get("frame_idx", None)
    
    # If frame_idx is not available, try to get it from attention extractor's video context
    if frame_idx is None and attention_extractor is not None:
        video_context = getattr(attention_extractor, '_current_video_context', None)
        if video_context is not None and video_context.get('video_id') == video_id:
            # Use frame_end from window context as a fallback
            frame_idx = video_context.get('frame_end')

    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]
    refiner_ids = outputs.get("pred_ids", None)

    ytvis_results = []
    for instance_id, (s, l, m) in enumerate(zip(scores, labels, masks)):
        segms = [
            mask_util.encode(np.array(_mask[:, :, None], order="F", dtype="uint8"))[0]
            for _mask in m
        ]
        for rle in segms:
            rle["counts"] = rle["counts"].decode("utf-8")

        # Build with desired key order; place refiner_id and predictor_query_id before segmentations
        res = OrderedDict()
        res["video_id"] = video_id
        res["score"] = s
        res["category_id"] = l
        if refiner_ids is not None and instance_id < len(refiner_ids):
            try:
                refiner_id = int(refiner_ids[instance_id])
                res["refiner_id"] = refiner_id
                
            except Exception:
                pass
        res["segmentations"] = segms
        ytvis_results.append(res)

    return ytvis_results


class TemporalYTVISEvaluator(YTVISEvaluator):
    """
    YTVIS evaluator that writes a sidecar results_temporal.json containing
    a replica of results.json augmented with refiner_id per prediction.
    Also captures image and patch dimensions during evaluation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attention_extractor = None
        self._image_dimensions = {}  # Store image dimensions per video: {video_id: {'image_height': int, 'image_width': int, 'H_patches': int, 'W_patches': int}}
        self._backbone_hook_handle = None

    def set_attention_extractor(self, extractor):
        self._attention_extractor = extractor
        # Hook into the backbone to capture patch dimensions
        self._setup_backbone_hook()

    def _setup_backbone_hook(self):
        """Set up a hook on the backbone to capture patch dimensions (H, W)."""
        if self._attention_extractor is None:
            return
        
        # Get the model from the extractor
        if not hasattr(self._attention_extractor, 'model'):
            return
        
        model = self._attention_extractor.model
        if not hasattr(model, 'backbone'):
            return
        
        backbone = model.backbone
        if not hasattr(backbone, 'vit_module'):
            return
        
        vit_module = backbone.vit_module
        
        # Hook into prepare_tokens_with_masks to capture H, W
        original_prepare_tokens = vit_module.prepare_tokens_with_masks
        
        def hooked_prepare_tokens(x, masks=None, return_HW=False):
            # Call original method
            if return_HW:
                result, H, W = original_prepare_tokens(x, masks, return_HW=True)
                # Capture dimensions
                self._capture_patch_dimensions(H, W, x)
                return result, H, W
            else:
                return original_prepare_tokens(x, masks, return_HW=False)
        
        # Replace the method
        vit_module.prepare_tokens_with_masks = hooked_prepare_tokens
        self._original_prepare_tokens = original_prepare_tokens

    def _capture_patch_dimensions(self, H_patches, W_patches, input_tensor):
        """
        Capture patch dimensions and compute image dimensions.
        
        Note: The input_tensor is the PADDED image (after size_divisibility padding to be multiple of 32).
        The patches are computed from this padded image, so:
        - padded_image_height, padded_image_width are the PADDED dimensions (what the model actually uses)
        - H_patches, W_patches are computed from the padded image
        - Patches correspond to regions in the padded image
        
        The model uses size_divisibility=32, so images are padded to be multiples of 32.
        Since patch_size=16 and 32 is a multiple of 16, padding doesn't affect patch boundaries.
        
        We also try to capture original image dimensions from the current inputs if available.
        """
        # Get video_id from current context if available
        video_id = None
        if self._attention_extractor is not None:
            context = getattr(self._attention_extractor, '_current_video_context', None)
            if context is not None:
                video_id = context.get('video_id')
        
        if video_id is None:
            return
        
        # Get padded image dimensions from input tensor
        # Input tensor shape: (B, C, H_img, W_img) - this is the PADDED image
        if input_tensor is not None and hasattr(input_tensor, 'shape') and len(input_tensor.shape) == 4:
            _, _, padded_image_height, padded_image_width = input_tensor.shape
            # Verify: image dimensions should be multiples of patch_size (16)
            # and size_divisibility (32)
            assert padded_image_height % 16 == 0, f"Image height {padded_image_height} is not a multiple of patch_size (16)"
            assert padded_image_width % 16 == 0, f"Image width {padded_image_width} is not a multiple of patch_size (16)"
            # Verify patch dimensions match
            assert H_patches == padded_image_height // 16, f"H_patches {H_patches} != padded_image_height // 16 {padded_image_height // 16}"
            assert W_patches == padded_image_width // 16, f"W_patches {W_patches} != padded_image_width // 16 {padded_image_width // 16}"
        else:
            # Fallback: compute from patch dimensions
            padded_image_height = H_patches * 16  # patch_size = 16
            padded_image_width = W_patches * 16
        
        # Try to get resized image dimensions (after ResizeShortestEdge, before padding)
        # from current inputs - this will be set in process() method if available
        resized_image_height = None
        resized_image_width = None
        if hasattr(self, '_current_inputs') and self._current_inputs is not None:
            try:
                if len(self._current_inputs) == 1:
                    input_dict = self._current_inputs[0]
                    # Get from image_size field (after ResizeShortestEdge, before padding)
                    if "image_size" in input_dict:
                        img_size = input_dict["image_size"]
                        if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
                            resized_image_height = int(img_size[0])
                            resized_image_width = int(img_size[1])
            except Exception:
                pass
        
        # Store dimensions (overwrite if already exists - use latest)
        # Note: padded_image_height/width are PADDED dimensions (what the model actually uses)
        dims_data = {
            'padded_image_height': int(padded_image_height),  # PADDED height (after size_divisibility padding)
            'padded_image_width': int(padded_image_width),    # PADDED width (after size_divisibility padding)
            'H_patches': int(H_patches),        # Patch grid height (from padded image)
            'W_patches': int(W_patches),        # Patch grid width (from padded image)
            'patch_size': 16,
            'size_divisibility': 32,            # Model pads to multiple of 32
        }
        
        # Add resized dimensions (after ResizeShortestEdge, before padding)
        # If not found in image_size, use padded dimensions as fallback
        if resized_image_height is None or resized_image_width is None:
            # Use padded dimensions as fallback (no padding in this case)
            resized_image_height = padded_image_height
            resized_image_width = padded_image_width
        
        dims_data['original_image_height'] = int(resized_image_height)  # After resize, before padding
        dims_data['original_image_width'] = int(resized_image_width)   # After resize, before padding
        # Compute padding offsets
        pad_height = padded_image_height - resized_image_height
        pad_width = padded_image_width - resized_image_width
        dims_data['pad_height'] = pad_height
        dims_data['pad_width'] = pad_width
        dims_data['pad_top'] = 0  # Padding is typically at bottom/right
        dims_data['pad_left'] = 0
        dims_data['pad_bottom'] = pad_height
        dims_data['pad_right'] = pad_width
        
        dims_data['note'] = 'padded_image_height/width are PADDED dimensions (after size_divisibility padding). original_image_height/width are RESIZED dimensions (after ResizeShortestEdge, before padding). Use original_image_height/width for mapping to resized image.'
        
        # Check if dimensions already exist and warn if they differ
        if video_id in self._image_dimensions:
            existing = self._image_dimensions[video_id]
            if (existing.get('padded_image_height') != dims_data['padded_image_height'] or 
                existing.get('padded_image_width') != dims_data['padded_image_width']):
                logging.getLogger(__name__).warning(
                    f"Video {video_id} padded dimensions changed: "
                    f"was {existing.get('padded_image_height')}x{existing.get('padded_image_width')}, "
                    f"now {dims_data['padded_image_height']}x{dims_data['padded_image_width']}"
                )
        
        self._image_dimensions[video_id] = dims_data
        
        # Save dimensions immediately to disk (append/update the JSON file)
        try:
            if self._output_dir:
                from detectron2.utils.file_io import PathManager
                import json
                dimensions_path = os.path.join(self._output_dir, "image_dimensions.json")
                
                # Load existing dimensions if file exists
                existing_dims = {}
                if os.path.exists(dimensions_path):
                    try:
                        with PathManager.open(dimensions_path, "r") as f:
                            existing_dims = json.load(f)
                    except Exception:
                        existing_dims = {}
                
                # Update with new dimensions
                existing_dims[str(video_id)] = dims_data
                
                # Save immediately
                with PathManager.open(dimensions_path, "w") as fdim:
                    fdim.write(json.dumps(existing_dims, indent=2))
                    fdim.flush()
        except Exception as e:
            # Don't fail if we can't save immediately - will save at end of evaluation
            logging.getLogger(__name__).debug(f"Could not save dimensions immediately: {e}")

    def reset(self):
        super().reset()
        self._predictions_temporal = []
        self._image_dimensions = {}

    def process(self, inputs, outputs):
        # Standard predictions for metrics and results.json
        super().process(inputs, outputs)
        # Augmented predictions for results_temporal.json
        try:
            aug = _instances_to_coco_json_video_with_refiner(
                inputs, outputs, 
                attention_extractor=self._attention_extractor
            )
            self._predictions_temporal.extend(aug)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"TemporalYTVISEvaluator process() failed to add refiner ids: {e}")
        
        # Store current inputs so hook can access original dimensions
        self._current_inputs = inputs
        
        # Try to capture image dimensions from inputs if not already captured
        # Note: This captures PADDED dimensions (after size_divisibility padding)
        # Original dimensions are captured from 'height' and 'width' fields in inputs
        try:
            if len(inputs) == 1:
                video_id = inputs[0].get("video_id")
                input_dict = inputs[0]
                
                # Get resized dimensions (after ResizeShortestEdge, before padding)
                # These are the "original" dimensions we want for plotting
                resized_height = None
                resized_width = None
                
                # First, try to get from image_size field (this is after augmentation but before padding)
                if "image_size" in input_dict:
                    img_size = input_dict["image_size"]
                    if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
                        # image_size is (H, W) after ResizeShortestEdge but before padding
                        resized_height = img_size[0]
                        resized_width = img_size[1]
                
                # If not found, we'll compute from padded dimensions after we get them
                
                # Get padded dimensions from image tensor
                if "image" in input_dict:
                    image = input_dict["image"]
                    if hasattr(image, 'shape') and len(image.shape) == 4:
                        # Shape: (B, C, H, W) - this is the PADDED image
                        _, _, padded_height, padded_width = image.shape
                        # Verify dimensions are multiples of patch_size (16)
                        if padded_height % 16 == 0 and padded_width % 16 == 0:
                            # Compute patch dimensions
                            H_patches = padded_height // 16
                            W_patches = padded_width // 16
                            
                            dims_data = {
                                'padded_image_height': int(padded_height),
                                'padded_image_width': int(padded_width),
                                'H_patches': int(H_patches),
                                'W_patches': int(W_patches),
                                'patch_size': 16,
                                'size_divisibility': 32,
                            }
                            
                            # Add resized dimensions (after ResizeShortestEdge, before padding)
                            # If not found in image_size, compute from padded dimensions
                            # (assuming padding is at bottom/right, which is typical)
                            if resized_height is None or resized_width is None:
                                # Compute resized dimensions from padded dimensions
                                # Padding is typically added to make dimensions multiples of size_divisibility (32)
                                # So we can compute the resized dimensions by finding the largest
                                # dimensions <= padded that are multiples of 16 (patch_size)
                                # But actually, we can't know exactly without knowing the padding
                                # So we'll use padded dimensions as a fallback
                                resized_height = padded_height
                                resized_width = padded_width
                            
                            dims_data['original_image_height'] = int(resized_height)  # After resize, before padding
                            dims_data['original_image_width'] = int(resized_width)   # After resize, before padding
                            pad_height = padded_height - resized_height
                            pad_width = padded_width - resized_width
                            dims_data['pad_height'] = pad_height
                            dims_data['pad_width'] = pad_width
                            dims_data['pad_top'] = 0  # Padding is typically at bottom/right
                            dims_data['pad_left'] = 0
                            dims_data['pad_bottom'] = pad_height
                            dims_data['pad_right'] = pad_width
                            
                            dims_data['note'] = 'padded_image_height/width are PADDED dimensions (after size_divisibility padding). Patches correspond to padded image regions. Use original_image_height/width for mapping to original image.'
                            
                            if video_id is not None and video_id not in self._image_dimensions:
                                self._image_dimensions[video_id] = dims_data
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to capture image dimensions from inputs: {e}")

        # Capture attention maps if extractor is provided
        try:
            if self._attention_extractor is not None and len(inputs) == 1:
                video_id = inputs[0].get("video_id")
                frame_idx = inputs[0].get("frame_idx", None)
                
                # If saving directly from hooks, set video context before forward pass
                # (Note: this is called AFTER forward pass, so hooks already saved)
                # But we still need to handle refiner attention if it's not saved from hooks
                if hasattr(self._attention_extractor, 'save_immediately_from_hook') and \
                   self._attention_extractor.save_immediately_from_hook:
                    # Backbone attention is already saved from hooks with video context
                    # But we should set context for next forward pass if needed
                    # For now, hooks save with fallback IDs if context not set
                    # We can improve this by setting context before forward pass in the future
                    pass
                
                # Save immediately after each forward pass to prevent RAM accumulation
                # Force save to ensure we don't accumulate (for refiner attention or if hook saving disabled)
                if hasattr(self._attention_extractor, 'save_attention_maps_immediately'):
                    self._attention_extractor.save_attention_maps_immediately(
                        video_id=video_id, 
                        frame_idx=frame_idx, 
                        force_save=True  # Force save every time process() is called
                    )
                else:
                    # Fallback: just clear if save method doesn't exist
                    if hasattr(self._attention_extractor, 'clear_attention_maps'):
                        self._attention_extractor.clear_attention_maps()
                
                # Save refiner_id mappings incrementally after each video completes
                # This prevents memory accumulation of tracking data
                if hasattr(self._attention_extractor, 'save_immediately_from_hook') and \
                   self._attention_extractor.save_immediately_from_hook and \
                   hasattr(self._attention_extractor, 'save_refiner_id_mappings') and \
                   video_id is not None:
                    # Save refiner_id mappings for this video incrementally
                    # This will append to the existing file and clear the video's data from memory
                    try:
                        self._attention_extractor.save_refiner_id_mappings(
                            video_id=video_id,
                            incremental=True
                        )
                    except Exception as e:
                        logging.getLogger(__name__).debug(f"Failed to save refiner_id mappings incrementally: {e}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed capturing attention maps: {e}")

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(sum(predictions, []))

            temporal_predictions = comm.gather(self._predictions_temporal, dst=0)
            temporal_predictions = list(sum(temporal_predictions, []))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            temporal_predictions = self._predictions_temporal

        if len(predictions) == 0:
            logging.getLogger(__name__).warning("[TemporalYTVISEvaluator] Did not receive valid predictions.")
            return {}

        # Let the base class handle standard saving and metrics
        self._results = OrderedDict()
        self._eval_predictions(predictions)

        # Apply the same category ID unmapping to temporal predictions
        # (convert from contiguous IDs 0-4 to dataset IDs 1-5)
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            if min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1:
                reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
                for result in temporal_predictions:
                    category_id = result["category_id"]
                    if category_id < num_classes:
                        result["category_id"] = reverse_id_mapping[category_id]

        # Write the augmented replica
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            temporal_path = os.path.join(self._output_dir, "results_temporal.json")
            with PathManager.open(temporal_path, "w") as f:
                f.write(json.dumps(temporal_predictions))
                f.flush()

            # Note: Attention maps are now saved immediately per video in process()
            # This section is kept for backward compatibility but should be empty
            # as we save incrementally to avoid RAM accumulation
            try:
                if self._attention_extractor is not None and hasattr(self._attention_extractor, 'video_attention_data'):
                    # Check if there are any videos that weren't saved (shouldn't happen)
                    if self._attention_extractor.video_attention_data:
                        logging.getLogger(__name__).warning(
                            f"Found {len(self._attention_extractor.video_attention_data)} videos with unsaved attention maps. "
                            "This should not happen with incremental saving."
                        )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Error checking attention maps: {e}"
                )

            # Also write a compact summary of all refiner ids (overall and per video)
            try:
                all_ids = []
                per_video = {}
                for item in temporal_predictions:
                    vid = item.get("video_id")
                    rid = item.get("refiner_id")
                    if rid is None:
                        continue
                    all_ids.append(rid)
                    if vid not in per_video:
                        per_video[vid] = []
                    per_video[vid].append(rid)

                all_ids_unique = sorted({int(x) for x in all_ids}) if len(all_ids) else []
                per_video_unique = {
                    str(k): sorted({int(x) for x in v}) for k, v in per_video.items()
                }
                summary = {
                    "refiner_ids_all": all_ids_unique,
                    "refiner_ids_per_video": per_video_unique,
                }
                temporal_summary_path = os.path.join(self._output_dir, "results_temporal_summary.json")
                with PathManager.open(temporal_summary_path, "w") as fsum:
                    fsum.write(json.dumps(summary))
                    fsum.flush()
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to write results_temporal_summary.json: {e}"
                )
            
            # Write image and patch dimensions
            try:
                if self._image_dimensions:
                    dimensions_path = os.path.join(self._output_dir, "image_dimensions.json")
                    # Convert keys to strings for JSON serialization
                    dimensions_dict = {str(k): v for k, v in self._image_dimensions.items()}
                    with PathManager.open(dimensions_path, "w") as fdim:
                        fdim.write(json.dumps(dimensions_dict, indent=2))
                        fdim.flush()
                    logging.getLogger(__name__).info(
                        f"Saved image dimensions for {len(self._image_dimensions)} videos to {dimensions_path}"
                    )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to write image_dimensions.json: {e}"
                )
            
            # Save refiner_id mappings (refiner_id → sequence_id mapping)
            try:
                if self._attention_extractor is not None and hasattr(self._attention_extractor, 'save_refiner_id_mappings'):
                    self._attention_extractor.save_refiner_id_mappings()
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Failed to save refiner_id mappings: {e}"
                )

        # Return the same metrics dict as base
        return copy.deepcopy(self._results)


