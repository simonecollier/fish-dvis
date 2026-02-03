import os
import json
import cv2
from tqdm import tqdm

def create_sorted_videos(json_path, image_root, output_base_dir):
    """
    Create videos from images sorted by species.
    
    Args:
        json_path: Path to all_videos.json
        image_root: Path to directory containing video image folders
        output_base_dir: Base directory where species-sorted videos will be saved
    """
    # Load JSON data
    print(f"Loading JSON from {json_path}...")
    with open(json_path) as f:
        data = json.load(f)
    
    # Create category mapping
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    
    # Create video info mapping
    video_info = {v["id"]: v for v in data["videos"]}
    
    # Extract species information from annotations
    species_by_video = {}
    if "annotations" in data:
        for ann in data["annotations"]:
            video_id = ann["video_id"]
            category_id = ann["category_id"]
            species_name = categories[category_id]
            if video_id not in species_by_video:
                species_by_video[video_id] = []
            species_by_video[video_id].append(species_name)
    
    # Remove duplicates - keep list of unique species per video
    for video_id in species_by_video:
        species_by_video[video_id] = list(set(species_by_video[video_id]))
    
    # Group videos by species (a video can appear in multiple species groups)
    videos_by_species = {}
    for video_id, species_list in species_by_video.items():
        for species in species_list:
            if species not in videos_by_species:
                videos_by_species[species] = []
            videos_by_species[species].append(video_id)
    
    print(f"Found {len(videos_by_species)} species:")
    for species, video_ids in videos_by_species.items():
        print(f"  {species}: {len(video_ids)} videos")
    
    # Process each species
    for species, video_ids in videos_by_species.items():
        # Create output directory for this species (lowercase, replace spaces with underscores)
        species_dir = species.lower().replace(" ", "_")
        output_dir = os.path.join(output_base_dir, species_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove duplicates from video_ids list (in case a video appears multiple times)
        video_ids = list(set(video_ids))
        
        print(f"\nProcessing {species} videos ({len(video_ids)} videos)...")
        
        # Process each video
        for video_id in tqdm(video_ids, desc=f"Processing {species}"):
            if video_id not in video_info:
                print(f"Warning: Video id {video_id} not found in video_info, skipping.")
                continue
            
            video = video_info[video_id]
            file_names = video["file_names"]
            height = video["height"]
            width = video["width"]
            
            # Get folder name from first file
            folder_name = os.path.dirname(file_names[0])
            
            # Create output video path
            output_video_path = os.path.join(output_dir, f"{folder_name}.mp4")
            
            # Skip if video already exists
            if os.path.exists(output_video_path):
                continue
            
            fps = 10
            out = None
            
            # Process each frame
            for fn in file_names:
                img_path = os.path.join(image_root, fn)
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Warning: could not read image {img_path}")
                    continue
                
                # Initialize video writer on first frame
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
                
                # Write frame without any overlays
                out.write(frame)
            
            if out:
                out.release()
    
    print(f"\nProcessing complete. Videos saved to: {output_base_dir}")

if __name__ == "__main__":
    json_path = "/home/simone/shared-data/fishway_ytvis/all_videos.json"
    image_root = "/home/simone/shared-data/fishway_ytvis/all_videos"
    output_base_dir = "/home/simone/shared-data/fishway_ytvis/recreated_videos"
    
    create_sorted_videos(json_path, image_root, output_base_dir)

