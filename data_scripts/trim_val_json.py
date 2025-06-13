import json

VAL_JSON_PATH = "/data/fishway_ytvis/val_long.json"  # Change this if your file is elsewhere
OUTPUT_JSON_PATH = "/data/fishway_ytvis/val.json"  # Output file

MAX_FRAMES = 30

def truncate_list(lst, n):
    """Truncate a list to at most n elements."""
    if isinstance(lst, list):
        return lst[:n]
    return lst

def main():
    with open(VAL_JSON_PATH, "r") as f:
        data = json.load(f)

    # Truncate videos
    for video in data["videos"]:
        if len(video["file_names"]) > MAX_FRAMES:
            video["file_names"] = video["file_names"][:MAX_FRAMES]
            video["length"] = MAX_FRAMES
        else:
            video["length"] = len(video["file_names"])

    # Truncate annotations
    for ann in data["annotations"]:
        # Truncate segmentations, bboxes, areas if present and are lists
        for key in ["segmentations", "bboxes", "areas"]:
            if key in ann and isinstance(ann[key], list):
                if len(ann[key]) > MAX_FRAMES:
                    ann[key] = ann[key][:MAX_FRAMES]
        # Update length if present
        if "length" in ann:
            ann["length"] = min(ann.get("length", MAX_FRAMES), MAX_FRAMES)

    # Save new json
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved truncated val.json to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()