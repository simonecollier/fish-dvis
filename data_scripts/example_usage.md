# Example Usage: Automatic Small Hole and Component Removal

## Overview
The `02_mask_outlier_check.py` script now includes functionality to automatically remove small holes (≤5 pixels) and small disconnected components (≤5 pixels) from masks and create a cleaned version of the dataset.

## Usage Examples

### 1. Create a cleaned dataset with small holes and components removed
```bash
python 02_mask_outlier_check.py \
  --json_file /data/fishway_ytvis/all_videos.json \
  --output_dir /data/fishway_ytvis/mask_validation_results \
  --create_cleaned \
  --max_hole_area 5 \
  --max_component_area 5
```

This will:
- Load the original dataset
- Detect and remove holes with area ≤ 5 pixels
- Detect and remove disconnected components with area ≤ 5 pixels
- Save the cleaned dataset as `all_videos_cleaned.json`
- Print statistics about the cleaning process

### 2. Use custom thresholds for hole and component removal
```bash
python 02_mask_outlier_check.py \
  --json_file /data/fishway_ytvis/all_videos.json \
  --cleaned_json_file /data/fishway_ytvis/my_cleaned_dataset.json \
  --output_dir /data/fishway_ytvis/mask_validation_results \
  --create_cleaned \
  --max_hole_area 3 \
  --max_component_area 10
```

### 3. Run mask review on the cleaned dataset
```bash
python 02_review_masks.py \
  --json_file /data/fishway_ytvis/all_videos.json \
  --cleaned_json_file /data/fishway_ytvis/mask_validation_results/all_videos_cleaned.json \
  --image_root /data/fishway_ytvis/all_videos \
  --output_dir /data/fishway_ytvis/mask_validation_results
```

## Benefits
1. **Reduces manual review workload**: Small holes and disconnected components are automatically fixed
2. **Preserves original data**: Original JSON file is never modified
3. **Configurable thresholds**: Adjust `--max_hole_area` and `--max_component_area` as needed
4. **Comprehensive statistics tracking**: See how many holes and components were removed
5. **Seamless integration**: Works with existing review workflow

## Output Files
- `all_videos_cleaned.json`: Dataset with small holes and components removed
- `mask_issues.json`: Issues found in the cleaned dataset
- `mask_validation_summary.json`: Summary statistics

## Statistics Example
```
=== CLEANING SUMMARY ===
Total masks processed: 15000
Masks with small holes removed: 234
Total small holes removed: 567
Masks with small components removed: 189
Total small components removed: 423
Cleaned dataset saved to: /data/fishway_ytvis/mask_validation_results/all_videos_cleaned.json
```

## What Gets Removed
- **Small holes**: Internal holes ≤ 5 pixels (configurable)
- **Small components**: Disconnected mask parts ≤ 5 pixels (configurable)
- **Preserved**: Large holes and components that may be legitimate features 