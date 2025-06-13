import os
import json
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from pycocotools import mask as mask_util

def convert_rle_to_compressed(rle, height, width):
    """Helper function to convert uncompressed RLE to compressed RLE"""
    if isinstance(rle["counts"], list):
        # Convert list-based RLE to string-based RLE
        rle = mask_util.frPyObjects(rle, height, width)
        if isinstance(rle, list):
            rle = mask_util.merge(rle)
    return rle

def load_ytvis_json(json_file, image_root, dataset_name):
    print(f"Loading {dataset_name} from {json_file}")

    with open(json_file, 'r') as f:
        ytvis = json.load(f)

    # Validate categories
    categories = ytvis.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {json_file}")
    
    print(f"Found {len(categories)} categories:")
    for cat in categories:
        print(f"  ID {cat['id']}: {cat['name']}")

    dataset_dicts = []

    for video in ytvis["videos"]:
        length = video["length"]
        height = video["height"]
        width = video["width"]
        file_names = [os.path.join(image_root, fn) for fn in video["file_names"]]

        video_dict = {
            "file_names": file_names,
            "height": height,
            "width": width,
            "length": length,
            "video_id": video["id"],
            "annotations": [[] for _ in range(length)]  # <== per-frame
        }

        # Find all annotations for this video
        for ann in [a for a in ytvis["annotations"] if a["video_id"] == video["id"]]:
            track_id = ann["id"]
            category_id = ann["category_id"]
            segs = ann["segmentations"]
            bboxes = ann.get("bboxes", [None] * length)
            areas = ann.get("areas", [None] * length)

            for frame_idx in range(length):
                seg = segs[frame_idx]
                bbox = bboxes[frame_idx]

                if seg is None or bbox is None:
                    continue

                # Handle segmentation based on its format
                if isinstance(seg, dict):
                    # RLE format
                    if "counts" in seg and "size" in seg:
                        try:
                            seg = convert_rle_to_compressed(seg, height, width)
                        except Exception as e:
                            print(f"Error processing RLE for video {video['id']}, frame {frame_idx}: {e}")
                            continue
                    else:
                        print(f"Invalid RLE format in video {video['id']}, frame {frame_idx}")
                        continue
                else:
                    raise ValueError(f"Unsupported mask format: {type(seg)}. Expected RLE dict.")

                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h

                ann_per_frame = {
                    "id": track_id,
                    "category_id": category_id,
                    "segmentation": seg,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "iscrowd": 0,
                }
                video_dict["annotations"][frame_idx].append(ann_per_frame)

        dataset_dicts.append(video_dict)

    # Add validation stats
    total_annotations = sum(len(v["annotations"]) for v in dataset_dicts)
    print(f"Loaded {len(dataset_dicts)} videos with {total_annotations} total annotations")

    return dataset_dicts

def register_ytvis_instances(name, metadata, json_file, image_root):
    """
    Register a YTVIS dataset in COCO's json annotation format.
    
    Args:
        name (str): name of the dataset (e.g., "ytvis_train")
        metadata (dict): dataset metadata containing categories, etc.
        json_file (str): path to the json instance annotation file
        image_root (str): directory which contains all the images
    """
    # Clear any existing dataset with this name
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name))
    
    # Get category information from the JSON file
    with open(json_file) as f:
        cats = json.load(f)["categories"]
    
    # Register metadata
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="ytvis",
        thing_classes=[c["name"] for c in sorted(cats, key=lambda x: x["id"])],
        thing_dataset_id_to_contiguous_id={c["id"]: i for i, c in enumerate(sorted(cats, key=lambda x: x["id"]))},
        **metadata
    )
