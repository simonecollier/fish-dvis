# DVIS-DAQ Attention Map Extraction

This repository contains scripts to extract **ALL** attention maps from a trained DVIS-DAQ model for video instance segmentation.

## Overview

The DVIS-DAQ model has multiple attention mechanisms across different components. This script extracts attention maps from:

1. **DINOv2 ViT-L Backbone** - Self-attention in vision transformer layers
2. **Multi-Scale Deformable Attention** - Pixel decoder attention
3. **Transformer Decoder** - 10 layers of self-attention and cross-attention
4. **VideoInstanceCutter/Tracker** - 6 layers of tracking attention
5. **ReID Branch** - Re-identification attention
6. **Temporal Refiner** - 6 layers of temporal attention

## Important Note: Dataset Format

**The model expects data in the same Detectron2 YTVIS format used during training, not raw video files.** This ensures compatibility and accurate attention map extraction.

## File Structure

```
/home/simone/fish-dvis/attention_analysis/
├── __init__.py                      # Python package initialization
├── extract_attention_maps.py        # Main attention extraction class
├── run_attention_extraction.py      # Simple command-line interface
├── requirements.txt                  # Python dependencies
└── README_attention_extraction.md   # This file
```

## Installation

1. **Navigate to the attention analysis directory:**
   ```bash
   cd /home/simone/fish-dvis/attention_analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure DVIS-DAQ codebase is available:**
   The script expects the DVIS-DAQ codebase to be located at:
   ```
   /home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ
   ```

4. **Ensure dataset is accessible:**
   The script expects your fishway dataset at:
   ```
   /data/fishway_ytvis/val.json
   /data/fishway_ytvis/all_videos_mask
   ```

## Environment Setup

**Important**: The attention extraction scripts need the same environment setup as your training script. You have two options:

### **Option 1: Use the Wrapper Script (Recommended)**
```bash
cd /home/simone/fish-dvis/attention_analysis

# List available videos
./run_attention_extraction_env.sh \
    --list-videos \
    --model /home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth

# Extract attention maps
./run_attention_extraction_env.sh \
    --video-id VIDEO_ID_FROM_DATASET \
    --model /home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth
```

### **Option 2: Manual Environment Setup**
```bash
# Activate virtual environment
source /home/simone/.venv/bin/activate

# Set environment variables
export DETECTRON2_DATASETS=/data
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ

# Add attention_analysis to Python path
export PYTHONPATH=$PYTHONPATH:/home/simone/fish-dvis/attention_analysis

# Now run the script
cd /home/simone/fish-dvis/attention_analysis
python run_attention_extraction.py --list-videos --model /path/to/model.pth
```

## Usage

### Method 1: Command Line Interface (Recommended)

#### **List Available Videos**
```bash
cd /home/simone/fish-dvis/attention_analysis

# See what video IDs are available in your dataset
./run_attention_extraction_env.sh \
    --list-videos \
    --model /home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth
```

#### **Just Print Shapes (Default - No Saving)**
```bash
# Extract attention maps for a specific video ID (just print shapes)
./run_attention_extraction_env.sh \
    --video-id VIDEO_ID_FROM_DATASET \
    --model /home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth
```

#### **Save Attention Maps to Disk**
```bash
# Extract and save all attention maps for a specific video ID
./run_attention_extraction_env.sh \
    --video-id VIDEO_ID_FROM_DATASET \
    --model /home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth \
    --save \
    --output /store/simone/attention/
```

#### **Customize Output Options**
```bash
# Save attention maps without printing shapes
./run_attention_extraction_env.sh \
    --video-id VIDEO_ID_FROM_DATASET \
    --model /home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth \
    --save \
    --output /custom/output/path/ \
    --no-print-shapes
```

### Method 2: Python Script

#### **List Available Videos**
```python
import sys
sys.path.append('/home/simone/fish-dvis/attention_analysis')

from extract_attention_maps import AttentionExtractor

# Initialize extractor
extractor = AttentionExtractor(
    model_path="/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth",
    output_dir="/store/simone/attention/"
)

# List available video IDs
available_videos = extractor.list_available_videos()
print(f"Available videos: {available_videos}")
```

#### **Just Print Shapes (No Saving)**
```python
# Extract attention maps for a specific video ID (just print shapes, don't save)
attention_maps = extractor.extract_attention_for_video(
    video_id="VIDEO_ID_FROM_DATASET",
    save_attention=False,  # Don't save to disk
    print_shapes=True      # Print shapes to console
)

# Get summary
summary = extractor.get_attention_summary()
print(f"Extracted {len(summary)} attention maps")
```

#### **Save Attention Maps to Disk**
```python
# Extract attention maps and save to disk
attention_maps = extractor.extract_attention_for_video(
    video_id="VIDEO_ID_FROM_DATASET",
    save_attention=True,   # Save to disk
    print_shapes=True      # Also print shapes
)
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video-id` | Video ID from the dataset (e.g., from val.json) | **Required** (unless using --list-videos) |
| `--model` | Path to trained model checkpoint | **Required** |
| `--output` | Output directory for attention maps | `/store/simone/attention/` |
| `--save` | Save attention maps to disk | `False` (only print shapes) |
| `--no-print-shapes` | Do not print attention map shapes | `False` (print shapes) |
| `--list-videos` | List available video IDs from dataset | `False` |

## Workflow

### **Step 1: Discover Available Videos**
```bash
cd /home/simone/fish-dvis/attention_analysis

./run_attention_extraction_env.sh \
    --list-videos \
    --model /path/to/model.pth
```

**Example Output:**
```
Available video IDs from dataset:
    1. video_001
    2. video_002
    3. video_003
    ...
Total videos available: 150
```

### **Step 2: Extract Attention Maps**
```bash
# Just print shapes (no saving)
./run_attention_extraction_env.sh \
    --video-id 1 \
    --model /home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth

# Save attention maps to disk
./run_attention_extraction_env.sh \
    --video-id video_001 \
    --model /path/to/model.pth \
    --save \
    --output /store/simone/attention/
```

## Output Modes

### **Mode 1: Shape-Only Output (Default)**
- **No files saved** to disk
- **Shapes printed** to console in organized format
- **Statistical summary** provided
- **Memory efficient** - attention maps only stored temporarily

**Example Output:**
```
ATTENTION MAPS SHAPES SUMMARY
================================================================================

Backbone (DINOv2 ViT-L):
-----------------------
  backbone_layer_0: [1, 16, 196, 196]
  backbone_layer_1: [1, 16, 196, 196]
  ...

Transformer Decoder:
-------------------
  decoder_self_attn_layer_0: [1, 8, 200, 200]
  decoder_cross_attn_layer_0: [1, 8, 200, 1024]
  ...

TOTAL ATTENTION MAPS: 89
================================================================================
```

### **Mode 2: Save to Disk**
- **Attention maps saved** as `.pt` files
- **Metadata saved** for each attention map
- **Summary file** created
- **Organized by video ID** in output directory

## Output Structure (When Saving)

The script saves attention maps to the specified output directory with the following structure:

```
/store/simone/attention/
└── video_001/                          # Video ID from dataset
    ├── backbone_layer_0.pt             # Backbone attention layer 0
    ├── backbone_layer_0_metadata.pt    # Metadata for layer 0
    ├── backbone_layer_1.pt             # Backbone attention layer 1
    ├── backbone_layer_1_metadata.pt    # Metadata for layer 1
    ├── ...
    ├── decoder_self_attn_layer_0.pt    # Decoder self-attention layer 0
    ├── decoder_cross_attn_layer_0.pt   # Decoder cross-attention layer 0
    ├── ...
    ├── tracker_self_attn_layer_0.pt    # Tracker self-attention layer 0
    ├── tracker_cross_attn_layer_0.pt   # Tracker cross-attention layer 0
    ├── slot_attention_layer_0.pt       # Slot attention layer 0
    ├── ...
    ├── refiner_long_temp_attn_layer_0.pt  # Refiner long temporal attention layer 0
    ├── refiner_obj_attn_layer_0.pt        # Refiner object attention layer 0
    ├── refiner_cross_attn_layer_0.pt      # Refiner cross-attention layer 0
    ├── ...
    ├── attention_summary.pt                # Summary of all attention maps
    └── reid_attention_layer_0.pt          # ReID attention layer 0 (if enabled)
```

## Attention Map Details

### 1. Backbone Attention (DINOv2 ViT-L)
- **Location**: `fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/meta_architecture.py` → `model.backbone`
- **Shape**: `[1, 16, num_patches, num_patches]` (16 heads, patch-to-patch attention)
- **Layers**: 24 transformer layers

### 2. Pixel Decoder Attention
- **Location**: `fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/meta_architecture.py` → `model.sem_seg_head.pixel_decoder`
- **Shape**: `[1, 8, H*W, num_reference_points]` (deformable attention)
- **Feature levels**: 3 different scales (res3, res4, res5)

### 3. Transformer Decoder Attention
- **Location**: `fish-dvis/DVIS_Plus/DVIS_DAQ/mask2former_video/modeling/transformer_decoder/video_mask2former_transformer_decoder.py`
- **Self-attention**: `[1, 8, 200, 200]` (query-to-query attention)
- **Cross-attention**: `[1, 8, 200, H*W]` (query-to-feature attention)
- **Layers**: 10 decoder layers

### 4. Tracker Attention (VideoInstanceCutter)
- **Location**: `fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/track_module.py` → `model.tracker`
- **Self-attention**: `[1, 8, num_track_queries, num_track_queries]`
- **Cross-attention**: `[1, 8, num_track_queries, 200]`
- **Slot attention**: `[1, 8, 5, feature_dim]` (5 slots)
- **Layers**: 6 tracker layers

### 5. ReID Branch Attention
- **Location**: `fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/meta_architecture.py` → `model.sem_seg_head.predictor.reid_branch`
- **Shape**: `[1, 200, 256]` (query-to-reid-feature attention)
- **Layers**: 3 ReID layers

### 6. Temporal Refiner Attention
- **Location**: `fish-dvis/DVIS_Plus/DVIS_DAQ/dvis_daq/refiner.py` → `model.refiner`
- **Long temporal self-attention**: `[1, 8, 31*200, 31*200]` (long-range temporal)
- **Object self-attention**: `[1, 8, 200, 200]` (cross-frame object consistency)
- **Cross-attention**: `[1, 8, 200, 31*feature_dim]` (object-to-temporal-feature)
- **Layers**: 6 refiner layers

## Expected Attention Map Count

Based on your model configuration:
- **Backbone**: 24 layers × 1 attention type = 24 maps
- **Pixel Decoder**: 6 layers × 1 attention type = 6 maps  
- **Decoder**: 10 layers × 2 attention types = 20 maps
- **Tracker**: 6 layers × 3 attention types = 18 maps
- **ReID**: 3 layers × 1 attention type = 3 maps
- **Refiner**: 6 layers × 3 attention types = 18 maps

**Total expected**: ~89 attention maps

## Use Cases

### **Quick Analysis (Shape-Only Mode)**
- **Fast exploration** of model architecture
- **Memory efficient** for large models
- **No disk space** required
- **Perfect for** understanding attention map dimensions

### **Full Analysis (Save Mode)**
- **Complete attention maps** saved for detailed analysis
- **Metadata preserved** for each attention map
- **Offline analysis** possible
- **Perfect for** research and publication

## Why Dataset Format?

### **Compatibility**
- **Same preprocessing** as training
- **Same data format** expected by model
- **Same annotations** available
- **No format conversion** needed

### **Accuracy**
- **Exact match** with training conditions
- **Proper normalization** applied
- **Correct frame ordering** maintained
- **Annotation consistency** preserved

## Customization

### **Dataset Paths**
If your dataset is located elsewhere, update the paths in `extract_attention_maps.py`:
```python
register_ytvis_instances(
    "ytvis_fishway_val",
    {},
    "/your/path/to/val.json",           # Update this path
    "/your/path/to/all_videos_mask"     # Update this path
)
```

### **Memory Considerations**
Attention maps can be large. For high-resolution videos with long sequences, consider:
- Using **shape-only mode** for initial exploration
- Processing shorter video clips
- Using gradient checkpointing
- Saving attention maps incrementally

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure DVIS-DAQ codebase is in Python path
2. **Model loading**: Check checkpoint format and model architecture compatibility
3. **Dataset access**: Verify dataset paths and permissions
4. **Memory issues**: Use shape-only mode or reduce video resolution

### Debug Mode
The script includes extensive logging. Check console output for detailed information about:
- Hook registration
- Dataset loading
- Attention map extraction
- File saving progress (if enabled)

## Example Output

### Shape-Only Mode
```
Starting attention extraction...
Video ID: video_001
Model: /home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth
Save to disk: False
Print shapes: True

Registered backbone attention hook for layer 0
Registered backbone attention hook for layer 1
...
Registered refiner long temporal attention hook for layer 0

Extraction completed successfully!
Total attention maps extracted: 89

ATTENTION MAPS SHAPES SUMMARY
================================================================================

Backbone (DINOv2 ViT-L):
-----------------------
  backbone_layer_0: [1, 16, 196, 196]
  backbone_layer_1: [1, 16, 196, 196]
  ...

TOTAL ATTENTION MAPS: 89
================================================================================
```

### Save Mode
```
Starting attention extraction...
Video ID: video_001
Model: /home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth
Save to disk: True
Print shapes: True
Output directory: /store/simone/attention/

Extraction completed successfully!
Total attention maps extracted: 89
Attention maps saved to: /store/simone/attention/
```

## Citation

If you use this attention extraction script in your research, please cite the DVIS-DAQ paper and acknowledge the attention analysis methodology.
