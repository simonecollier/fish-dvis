#!/usr/bin/env python3
"""
Script to evaluate all checkpoints for a given model.
This script will:
1. Find all checkpoint files in a model directory
2. Run evaluation for each checkpoint
3. Save results in separate subdirectories

Note: This script only performs evaluation. Use visualization_scripts/run_analysis_all_checkpoint_results.sh
to analyze the results and create comparison plots.
"""

import os
import glob
import subprocess
import argparse
import re
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_checkpoints(model_dir: str) -> List[str]:
    """
    Find all checkpoint files in the model directory.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        List of checkpoint file paths, sorted by iteration number
    """
    checkpoint_pattern = os.path.join(model_dir, "model_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    # Sort by iteration number
    def extract_iteration(checkpoint_path):
        filename = os.path.basename(checkpoint_path)
        match = re.search(r'model_(\d+)\.pth', filename)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=extract_iteration)
    return checkpoints

def run_evaluation(checkpoint_path: str, config_file: str, output_dir: str, 
                   num_gpus: int = 1, debug: bool = False) -> bool:
    """
    Run evaluation for a single checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config_file: Path to the config file
        output_dir: Output directory for results
        num_gpus: Number of GPUs to use
        debug: Whether to enable debug mode
        
    Returns:
        True if evaluation succeeded, False otherwise
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "/home/simone/fish-dvis/training_scripts/train_net_video.py",
        "--num-gpus", str(num_gpus),
        "--config-file", config_file,
        "--eval-only",
        "MODEL.WEIGHTS", checkpoint_path,
        "OUTPUT_DIR", output_dir
    ]
    
    if debug:
        cmd.append("--debug")
    
    logger.info(f"Running evaluation for checkpoint: {os.path.basename(checkpoint_path)}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env['DETECTRON2_DATASETS'] = '/data'
        env['PYTHONPATH'] = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ'
        
        # Run evaluation
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            logger.info(f"Evaluation completed successfully for {os.path.basename(checkpoint_path)}")
            return True
        else:
            logger.error(f"Evaluation failed for {os.path.basename(checkpoint_path)}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Evaluation timed out for {os.path.basename(checkpoint_path)}")
        return False
    except Exception as e:
        logger.error(f"Exception during evaluation: {e}")
        return False





def main():
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints for a model')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to the model directory containing checkpoints')
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to the config file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Base output directory for all evaluation results (defaults to model_dir/checkpoint_evaluations)')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip evaluation if results already exist')
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "checkpoint_evaluations")
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.model_dir)
    logger.info(f"Found {len(checkpoints)} checkpoints in {args.model_dir}")
    
    if not checkpoints:
        logger.error("No checkpoints found!")
        return
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate each checkpoint
    successful_evaluations = 0
    
    for i, checkpoint_path in enumerate(checkpoints):
        checkpoint_name = os.path.basename(checkpoint_path)
        checkpoint_iteration = re.search(r'model_(\d+)\.pth', checkpoint_name)
        iteration_num = checkpoint_iteration.group(1) if checkpoint_iteration else str(i)
        
        # Create output directory for this checkpoint
        checkpoint_output_dir = os.path.join(args.output_dir, f"checkpoint_{iteration_num}")
        
        # Check if results already exist
        if args.skip_existing and os.path.exists(os.path.join(checkpoint_output_dir, "inference", "results.json")):
            logger.info(f"Skipping {checkpoint_name} - results already exist")
            successful_evaluations += 1
            continue
        
        logger.info(f"Evaluating checkpoint {i+1}/{len(checkpoints)}: {checkpoint_name}")
        
        # Run evaluation
        success = run_evaluation(
            checkpoint_path=checkpoint_path,
            config_file=args.config_file,
            output_dir=checkpoint_output_dir,
            num_gpus=args.num_gpus,
            debug=args.debug
        )
        
        if success:
            successful_evaluations += 1
        else:
            logger.error(f"Failed to evaluate {checkpoint_name}")
    
    logger.info(f"Successfully evaluated {successful_evaluations}/{len(checkpoints)} checkpoints")
    
    if successful_evaluations > 0:
        logger.info(f"Evaluation results saved to {args.output_dir}")
        logger.info("Use visualization_scripts/run_analysis_all_checkpoint_results.sh to analyze the results")
    else:
        logger.error("No successful evaluations completed")

if __name__ == "__main__":
    main() 