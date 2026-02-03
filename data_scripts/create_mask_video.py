import os
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm

try:
    from pycocotools import mask as mask_util
except ImportError:
    print("pycocotools not found. Please install it with: pip install pycocotools")
    exit()

def apply_degradation(mask, degradation_type, factor):
    """
    Applies a degradation effect to the mask.
    """
    if degradation_type == 'morph':
        if factor == 1.0:
            return mask
        kernel = np.ones((3, 3), np.uint8)
        if factor < 1.0:
            # More erosion for smaller factors
            it = int(1 + (1.0 - factor) * 10)
            return cv2.erode(mask, kernel, iterations=it)
        else:  # factor > 1.0
            # More dilation for larger factors
            it = int(1 + (factor - 1.0) * 10)
            return cv2.dilate(mask, kernel, iterations=it)

    elif degradation_type == 'blur':
        if factor <= 0:
            return mask
        # Ensure kernel size is odd and at least 1x1
        kernel_size = int(factor) * 2 + 1
        blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        # Binarize the blurred mask to keep it as a mask, but with softer edges
        _, binarized_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)
        return binarized_mask

    elif degradation_type == 'noise':
        if factor <= 0:
            return mask
        # factor should be between 0 and 1
        factor = np.clip(factor, 0, 1)
        noise = np.random.rand(*mask.shape)
        noisy_mask = mask.copy()
        # Apply pepper noise
        noisy_mask[noise < (factor / 2)] = 0
        # Apply salt noise
        noisy_mask[noise > (1 - factor / 2)] = 255
        return noisy_mask
    
    return mask


def create_masked_frames(
    annotations_file,
    output_dir,
    degradation_type='none',
    degradation_factor=1.0
):
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Group annotations by video_id
    annotations_by_video = {}
    if 'annotations' not in data:
        print("Error: 'annotations' key not found in JSON file.")
        return
        
    for ann in data.get('annotations', []):
        video_id = ann['video_id']
        if video_id not in annotations_by_video:
            annotations_by_video[video_id] = []
        annotations_by_video[video_id].append(ann)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(data['videos'])} videos...")
    for video_info in tqdm(data['videos'], desc="Processing videos"):
        video_id = video_info['id']
        if video_id not in annotations_by_video:
            continue

        if not video_info.get('file_names'):
            print(f"Warning: Video ID {video_id} has no 'file_names' list. Skipping.")
            continue

        video_annotations = annotations_by_video[video_id]
        width = video_info['width']
        height = video_info['height']
        
        # Use the directory of the first frame as the output subdirectory name
        video_dir_name = os.path.dirname(video_info['file_names'][0])
        video_output_dir = os.path.join(output_dir, video_dir_name)
        os.makedirs(video_output_dir, exist_ok=True)

        for frame_idx, frame_file_path in enumerate(tqdm(video_info['file_names'], desc=f"Video {video_id}", leave=False)):
            combined_mask = np.zeros((height, width), dtype=np.uint8)

            for ann in video_annotations:
                # The 'segmentations' list can be shorter than the number of frames
                if frame_idx < len(ann['segmentations']):
                    segmentation = ann['segmentations'][frame_idx]
                    if segmentation:
                        # The segmentation object is the RLE object
                        instance_mask = mask_util.decode(segmentation).astype(np.uint8) * 255
                        
                        if degradation_type != 'none':
                            instance_mask = apply_degradation(instance_mask, degradation_type, degradation_factor)

                        combined_mask = np.maximum(combined_mask, instance_mask)
            
            # Create a 3-channel image for writing
            output_frame = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

            # Use original frame basename for output, but with .jpg extension
            frame_basename = os.path.basename(frame_file_path)
            output_filename = os.path.splitext(frame_basename)[0] + '.jpg'
            output_path = os.path.join(video_output_dir, output_filename)
            cv2.imwrite(output_path, output_frame)
            
    print("Processing complete.")


if __name__ == "__main__":
    # Example usage: set your parameters here
    create_masked_frames(
        annotations_file="/data/fishway_ytvis/all_videos.json",
        output_dir="/data/fishway_ytvis/all_videos_mask",
        degradation_type='none',  # or 'morph', 'blur', 'noise'
        degradation_factor=0    # adjust as needed
    ) 