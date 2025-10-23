#!/usr/bin/env python3
"""
Script to evaluate all checkpoints for a model.
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

def cleanup_gpu_memory():
    """
    Clean up GPU memory after each evaluation.
    This helps prevent memory accumulation across checkpoints.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.debug("GPU memory cleaned up")
    except ImportError:
        logger.debug("PyTorch not available, skipping GPU cleanup")
    except Exception as e:
        logger.debug(f"GPU cleanup failed: {e}")

def get_gpu_memory_info():
    """
    Get current GPU memory usage information.
    
    Returns:
        Dictionary with memory information
    """
    try:
        import torch
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_cached = torch.cuda.memory_cached() / 1024**3        # GB
            
            return {
                'allocated_gb': round(memory_allocated, 2),
                'reserved_gb': round(memory_reserved, 2),
                'cached_gb': round(memory_cached, 2)
            }
    except ImportError:
        return None
    except Exception as e:
        logger.debug(f"Failed to get GPU memory info: {e}")
        return None

def find_checkpoints(model_dir: str) -> List[str]:
    """
    Find all checkpoint files in the model directory.
    Excludes model_final.pth as it's redundant with the highest iteration checkpoint.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        List of checkpoint file paths, sorted by iteration number
    """
    checkpoint_pattern = os.path.join(model_dir, "model_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    # Filter out model_final.pth as it's redundant
    checkpoints = [cp for cp in checkpoints if not cp.endswith('model_final.pth')]
    
    # Sort by iteration number
    def extract_iteration(checkpoint_path):
        filename = os.path.basename(checkpoint_path)
        match = re.search(r'model_(\d+)\.pth', filename)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=extract_iteration)
    return checkpoints

def run_evaluation(checkpoint_path: str, config_file: str, output_dir: str, 
                   debug: bool = False, device: int = 0) -> bool:
    """
    Run evaluation for a single checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config_file: Path to the config file
        output_dir: Output directory for results
        debug: Whether to enable debug mode
        device: GPU device ID to use
        
    Returns:
        True if evaluation succeeded, False otherwise
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "/home/simone/fish-dvis/training_scripts/train_net_video.py",
        "--num-gpus", "1",
        "--config-file", config_file,
        "--eval-only",
        "MODEL.WEIGHTS", checkpoint_path,
        "OUTPUT_DIR", output_dir
    ]
    
    if debug:
        cmd.append("--debug")
    
    logger.info(f"Running evaluation for checkpoint: {os.path.basename(checkpoint_path)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using 1 GPU")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        
        # For sequential processing, use the CUDA_VISIBLE_DEVICES set by the shell script
        current_visible = env.get('CUDA_VISIBLE_DEVICES', '')
        if current_visible:
            logger.info(f"Using CUDA_VISIBLE_DEVICES={current_visible} for evaluation")
        else:
            # Fallback: set device explicitly if not already set
            env['CUDA_VISIBLE_DEVICES'] = str(device)
            logger.info(f"Setting CUDA_VISIBLE_DEVICES={device} for evaluation")
        env['DETECTRON2_DATASETS'] = '/data'
        env['PYTHONPATH'] = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ'
        
        # Run evaluation
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            logger.info(f"Evaluation completed successfully for {os.path.basename(checkpoint_path)}")
            # Clean up GPU memory after successful evaluation
            cleanup_gpu_memory()
            return True
        else:
            logger.error(f"Evaluation failed for {os.path.basename(checkpoint_path)}")
            # Log both stderr and stdout to aid debugging (some libs write traces to stdout)
            if result.stderr:
                logger.error(f"stderr:\n{result.stderr}")
            if result.stdout:
                logger.error(f"stdout:\n{result.stdout}")

            # Persist full outputs for post-mortem analysis
            try:
                base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
                safe_name = base_name.replace('.', '_')
                stderr_path = os.path.join(output_dir, f"{safe_name}_stderr.log")
                stdout_path = os.path.join(output_dir, f"{safe_name}_stdout.log")
                with open(stderr_path, 'w') as f:
                    f.write(result.stderr or "")
                with open(stdout_path, 'w') as f:
                    f.write(result.stdout or "")
                logger.info(f"Saved subprocess logs to {stderr_path} and {stdout_path}")
            except Exception as log_e:
                logger.debug(f"Failed to save subprocess logs: {log_e}")
            # Clean up GPU memory even after failed evaluation
            cleanup_gpu_memory()
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Evaluation timed out for {os.path.basename(checkpoint_path)}")
        cleanup_gpu_memory()
        return False
    except Exception as e:
        logger.error(f"Exception during evaluation: {e}")
        cleanup_gpu_memory()
        return False

def evaluate_single_checkpoint(i, checkpoint_path, checkpoints, args):
    """Evaluate a single checkpoint and return 1 if successful, 0 if failed."""
    checkpoint_name = os.path.basename(checkpoint_path)
    checkpoint_iteration = re.search(r'model_(\d+)\.pth', checkpoint_name)
    iteration_num = checkpoint_iteration.group(1) if checkpoint_iteration else str(i)
    
    # Create output directory for this checkpoint
    checkpoint_output_dir = os.path.join(args.output_dir, f"checkpoint_{iteration_num}")
    
    # Check if results already exist
    if args.skip_existing and os.path.exists(os.path.join(checkpoint_output_dir, "inference", "results.json")):
        logger.info(f"Skipping {checkpoint_name} - results already exist")
        return 1
    
    # Log memory before evaluation
    memory_before = None
    if args.monitor_memory:
        memory_before = get_gpu_memory_info()
        if memory_before:
            logger.info(f"Memory before {checkpoint_name} - Allocated: {memory_before['allocated_gb']}GB, Reserved: {memory_before['reserved_gb']}GB")
    
    logger.info(f"Evaluating checkpoint {i+1}/{len(checkpoints)}: {checkpoint_name}")
    
    # Run evaluation on the specified device
    success = run_evaluation(
        checkpoint_path=checkpoint_path,
        config_file=args.config_file,
        output_dir=checkpoint_output_dir,
        debug=args.debug,
        device=0  # Always use device 0 since CUDA_VISIBLE_DEVICES is set to the target device
    )
    
    # Log memory after evaluation
    if args.monitor_memory:
        memory_after = get_gpu_memory_info()
        if memory_after and memory_before:
            allocated_diff = memory_after['allocated_gb'] - memory_before['allocated_gb']
            reserved_diff = memory_after['reserved_gb'] - memory_before['reserved_gb']
            logger.info(f"Memory after {checkpoint_name} - Allocated: {memory_after['allocated_gb']:.2f}GB (++{allocated_diff:.2f}), Reserved: {memory_after['reserved_gb']:.2f}GB (++{reserved_diff:.2f})")
    
    if success:
        logger.info(f"Successfully evaluated {checkpoint_name}")
        return 1
    else:
        logger.error(f"Failed to evaluate {checkpoint_name}")
        return 0

# Parallel processing function removed - only sequential processing is used

def main():
    parser = argparse.ArgumentParser(description='Evaluate all checkpoints for a model')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to the model directory containing checkpoints')
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to the config file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Base output directory for all evaluation results (defaults to model_dir/checkpoint_evaluations)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip evaluation if results already exist')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID to use (default: 0)')
    parser.add_argument('--monitor-memory', action='store_true',
                       help='Monitor GPU memory usage during evaluation')
    parser.add_argument('--checkpoint-range', type=str, default=None,
                       help='Evaluate only checkpoints in range (e.g., "1-10", "5-15")')
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, "checkpoint_evaluations")
    
    # Find all checkpoints
    all_checkpoints = find_checkpoints(args.model_dir)
    logger.info(f"Found {len(all_checkpoints)} checkpoints in {args.model_dir}")
    
    if not all_checkpoints:
        logger.error("No checkpoints found!")
        return
    
    # Filter checkpoints by range if specified
    if args.checkpoint_range:
        try:
            start_idx, end_idx = map(int, args.checkpoint_range.split('-'))
            # Convert to 0-based indexing
            start_idx = max(0, start_idx - 1)
            end_idx = min(len(all_checkpoints), end_idx)
            checkpoints = all_checkpoints[start_idx:end_idx]
            logger.info(f"Filtered checkpoints to range {start_idx+1}-{end_idx}: {len(checkpoints)} checkpoints")
            
            if not checkpoints:
                logger.error(f"No checkpoints in specified range {args.checkpoint_range}")
                return
        except ValueError:
            logger.error(f"Invalid checkpoint range format: {args.checkpoint_range}. Use format '1-10'")
            return
    else:
        checkpoints = all_checkpoints
        logger.info(f"Evaluating all {len(checkpoints)} checkpoints")
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy the config file to the output directory for reproducibility
    config_copy_path = os.path.join(args.output_dir, "config.yaml")
    if os.path.exists(args.config_file):
        import shutil
        shutil.copy2(args.config_file, config_copy_path)
        logger.info(f"Copied config file to {config_copy_path}")
    
    # Evaluate each checkpoint
    successful_evaluations = 0
    
    # Log initial GPU memory state
    if args.monitor_memory:
        initial_memory = get_gpu_memory_info()
        if initial_memory:
            logger.info(f"Initial GPU memory - Allocated: {initial_memory['allocated_gb']}GB, Reserved: {initial_memory['reserved_gb']}GB")
    
    # Run sequential evaluation
    logger.info(f"Running sequential evaluation on GPU {args.device}")
    for i, checkpoint_path in enumerate(checkpoints):
        successful_evaluations += evaluate_single_checkpoint(i, checkpoint_path, checkpoints, args)
    
    
    # Final memory cleanup
    cleanup_gpu_memory()
    if args.monitor_memory:
        final_memory = get_gpu_memory_info()
        if final_memory:
            logger.info(f"Final GPU memory - Allocated: {final_memory['allocated_gb']}GB, Reserved: {final_memory['reserved_gb']}GB")
    
    logger.info(f"Successfully evaluated {successful_evaluations}/{len(checkpoints)} checkpoints")
    
    if successful_evaluations > 0:
        logger.info(f"Evaluation results saved to {args.output_dir}")
        logger.info("Use visualization_scripts/run_analysis_all_checkpoint_results.sh to analyze the results")
        
        # Create a summary file with evaluation info
        summary_file = os.path.join(args.output_dir, "evaluation_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Summary\n")
            f.write(f"==================\n")
            f.write(f"Model directory: {args.model_dir}\n")
            f.write(f"Config file: {args.config_file}\n")
            f.write(f"Output directory: {args.output_dir}\n")
            f.write(f"Total checkpoints found: {len(checkpoints)}\n")
            f.write(f"Successfully evaluated: {successful_evaluations}\n")
            f.write(f"Failed evaluations: {len(checkpoints) - successful_evaluations}\n")
            f.write(f"Evaluation completed: {os.popen('date').read().strip()}\n")
        
        logger.info(f"Evaluation summary saved to {summary_file}")
    else:
        logger.error("No successful evaluations completed")

if __name__ == "__main__":
    main() 