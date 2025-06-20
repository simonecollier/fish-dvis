import json
import csv

VAL_JSON_PATH = '/data/fishway_ytvis/val.json'
FISHWAY_METADATA_PATH = 'home/simone/fish-dvis/data_scripts/fishway_metadata.csv'
RESULTS_JSON_PATH = 'fish-dvis/dvis-model-outputs/trained_models/dvis_daq_vitl_offline_80vids_10k/inference/results.json'
OUTPUT_JSON_PATH = '/data/labeled/simone/fishway_ytvis.json'
# --- Load val.json ---
with open(VAL_JSON_PATH, 'r') as f:
    val_data = json.load(f)

# --- Load fishway_metadata.csv ---
video_to_species = {}
with open(FISHWAY_METADATA_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        folder = row['Fish Computer Folder'].strip()
        species = row['Species'].strip()
        video_to_species[folder] = species

# --- Build category mapping (species to category_id) ---
# You may want to use the categories from val.json or define your own
species_to_catid = {}
categories = []
if 'categories' in val_data:    
    for cat in val_data['categories']:
        species_to_catid[cat['name']] = cat['id']
        categories.append(cat)
else:
    # If not present, build from unique species
    for i, species in enumerate(sorted(set(video_to_species.values())), 1):
        species_to_catid[species] = i
        categories.append({'id': i, 'name': species})

# --- Map video_id to folder name ---
videoid_to_folder = {}
for vid in val_data['videos']:
    # Try to extract folder from the first file_name
    first_file = vid['file_names'][0]
    folder = first_file.split('/')[0]
    videoid_to_folder[vid['id']] = folder

# --- Load results.json ---
with open('fish-dvis/dvis-model-outputs/trained_models/dvis_daq_vitl_offline_80vids/inference/results.json', 'r') as f:
    results = json.load(f)

# --- Build annotations ---
annotations = []
ann_id = 1
for pred in results:
    video_id = pred['video_id']
    folder = videoid_to_folder.get(video_id)
    if not folder:
        continue  # skip if not found

    # Get correct species/category
    species = video_to_species.get(folder)
    if not species:
        continue  # skip if not found

    category_id = species_to_catid[species]

    # For each segmentation (one per frame/instance)
    for seg in pred['segmentations']:
        annotation = {
            'id': ann_id,
            'video_id': video_id,
            'category_id': category_id,
            'score': pred['score'],
            'segmentation': seg,
            # Add other fields as needed (e.g., frame_id, track_id, etc.)
        }
        annotations.append(annotation)
        ann_id += 1

# --- Build output JSON ---
output = {
    'videos': val_data['videos'],
    'categories': categories,
    'annotations': annotations
}

with open(OUTPUT_JSON_PATH, 'w') as f:
    json.dump(output, f, indent=2)