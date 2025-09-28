#!/bin/bash

# Fishway Dataset Creation Pipeline
# This script runs the four-stage pipeline to create YTVIS format datasets

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METADATA_CSV="$SCRIPT_DIR/fishway_metadata.csv"
BASE_DATA_DIR="/data/labeled"
OUTPUT_DIR="/data/fishway_ytvis"
ALL_VIDEOS_JSON="$OUTPUT_DIR/all_videos.json"
TRAIN_JSON="$OUTPUT_DIR/train.json"
VAL_JSON="$OUTPUT_DIR/val.json"
MASK_VALIDATION_DIR="/store/simone/mask_validation_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if [ ! -f "$METADATA_CSV" ]; then
        error "Metadata CSV not found: $METADATA_CSV"
        exit 1
    fi
    
    if [ ! -d "$BASE_DATA_DIR" ]; then
        error "Base data directory not found: $BASE_DATA_DIR"
        exit 1
    fi
    
    # Check if Python scripts exist
    for script in "01_convert_coco_to_ytvis.py" "02_validate_masks.py" "03_create_train_val_jsons.py" "04_validate_ytvis.py"; do
        if [ ! -f "$SCRIPT_DIR/$script" ]; then
            error "Script not found: $SCRIPT_DIR/$script"
            exit 1
        fi
    done
    
    success "Prerequisites check passed"
}

# Stage 1: Convert COCO to YTVIS
stage1_convert_coco_to_ytvis() {
    log "Starting Stage 1: COCO to YTVIS conversion"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    cd "$SCRIPT_DIR"
    python 01_convert_coco_to_ytvis.py
    
    if [ $? -eq 0 ]; then
        success "Stage 1 completed successfully"
    else
        error "Stage 1 failed"
        exit 1
    fi
}

# Stage 2: Validate masks
stage2_validate_masks() {
    log "Starting Stage 2: Mask validation and review"
    
    # Create mask validation directory
    mkdir -p "$MASK_VALIDATION_DIR"
    
    cd "$SCRIPT_DIR"
    python 02_validate_masks.py
    
    if [ $? -eq 0 ]; then
        success "Stage 2 completed successfully"
    else
        error "Stage 2 failed"
        exit 1
    fi
}

# Stage 3: Create train/val splits
stage3_create_splits() {
    log "Starting Stage 3: Create train/val splits"
    
    cd "$SCRIPT_DIR"
    python 03_create_train_val_jsons.py
    
    if [ $? -eq 0 ]; then
        success "Stage 3 completed successfully"
    else
        error "Stage 3 failed"
        exit 1
    fi
}

# Stage 4: Validate datasets
stage4_validate() {
    log "Starting Stage 4: Validate datasets"
    
    cd "$SCRIPT_DIR"
    python 04_validate_ytvis.py
    
    if [ $? -eq 0 ]; then
        success "Stage 4 completed successfully"
    else
        error "Stage 4 failed"
        exit 1
    fi
}

# Main execution
main() {
    log "Starting Fishway Dataset Creation Pipeline"
    log "Configuration:"
    log "  Metadata CSV: $METADATA_CSV"
    log "  Base data dir: $BASE_DATA_DIR"
    log "  Output dir: $OUTPUT_DIR"
    log "  All videos JSON: $ALL_VIDEOS_JSON"
    log "  Train JSON: $TRAIN_JSON"
    log "  Val JSON: $VAL_JSON"
    log "  Mask validation dir: $MASK_VALIDATION_DIR"
    echo
    
    check_prerequisites
    echo
    
    stage1_convert_coco_to_ytvis
    echo
    
    stage2_validate_masks
    echo
    
    stage3_create_splits
    echo
    
    stage4_validate
    echo
    
    success "Pipeline completed successfully!"
    log "Output files:"
    log "  - All videos: $ALL_VIDEOS_JSON"
    log "  - Train set: $TRAIN_JSON"
    log "  - Val set: $VAL_JSON"
    log "  - Images: $OUTPUT_DIR/all_videos/"
    log "  - Mask validation results: $MASK_VALIDATION_DIR/"
}

# Run main function
main "$@" 