import os
from .ytvis_loader import register_ytvis_instances

def register_all_ytvis_fishway(root="/data/fishway_ytvis"):
    """
    Register all YTVIS-format Fishway datasets
    """
    SPLITS = [
        ("ytvis_fishway_train", "train", "train.json"),
        ("ytvis_fishway_val", "val", "val.json"),
    ]
    
    for name, image_root, json_file in SPLITS:
        json_path = os.path.join(root, json_file)
        image_path = os.path.join(root, image_root)
        
        # Debug print to verify paths
        print(f"Registering {name}:")
        print(f"  JSON: {json_path}")
        print(f"  Images: {image_path}")
        
        register_ytvis_instances(
            name,
            {
                "min_size": 1,
            },
            json_path,
            image_path
        )