#!/usr/bin/env python3
"""
Batch wrapper to create multiple scrambled versions of validation JSON files.

This script creates 10 scrambled versions (seeds 1-10) for each fold (1-6)
of the validation data.

Usage:
    python batch_scramble_folds.py [--folds FOLDS] [--seeds SEEDS] [--base-dir DIR] [--verify]
    
Example:
    python batch_scramble_folds.py --folds 1,2,3,4,5,6 --seeds 1,2,3,4,5,6,7,8,9,10
"""

import subprocess
import argparse
import sys
from pathlib import Path


def run_scramble(input_json, output_json, seed, verify=False):
    """
    Run the scramble_val.py script with given parameters.
    
    Args:
        input_json: Path to input JSON file
        output_json: Path to output scrambled JSON file
        seed: Random seed for scrambling
        verify: Whether to verify scrambling after completion
        
    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / "scramble_val.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        str(input_json),
        str(output_json),
        "--seed", str(seed)
    ]
    
    if verify:
        cmd.append("--verify")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Successfully created: {output_json}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating {output_json}:")
        print(f"  {e.stderr}")
        return False


def batch_scramble_folds(base_dir, folds, seeds, verify=False):
    """
    Create scrambled versions for multiple folds and seeds.
    
    Args:
        base_dir: Base directory containing the validation JSON files
        folds: List of fold numbers (e.g., [1, 2, 3, 4, 5, 6])
        seeds: List of seed values (e.g., [1, 2, ..., 10])
        verify: Whether to verify each scrambling
    """
    base_path = Path(base_dir)
    
    total_tasks = len(folds) * len(seeds)
    completed = 0
    failed = 0
    
    print(f"Starting batch scrambling:")
    print(f"  Folds: {folds}")
    print(f"  Seeds: {seeds}")
    print(f"  Total tasks: {total_tasks}")
    print()
    
    for fold in folds:
        # Construct input file path
        input_file = base_path / f"val_fold{fold}_all_frames.json"
        
        if not input_file.exists():
            print(f"⚠ Warning: Input file not found: {input_file}")
            print(f"  Skipping fold {fold}")
            continue
        
        print(f"Processing fold {fold}...")
        
        for seed in seeds:
            # Construct output file path
            output_file = base_path / f"val_fold{fold}_all_frames_scrambled_seed{seed}.json"
            
            print(f"  Seed {seed}: {input_file.name} -> {output_file.name}")
            
            if run_scramble(input_file, output_file, seed, verify):
                completed += 1
            else:
                failed += 1
            
            print()  # Blank line between tasks
    
    print("=" * 60)
    print(f"Batch scrambling completed!")
    print(f"  Successful: {completed}/{total_tasks}")
    print(f"  Failed: {failed}/{total_tasks}")
    
    if failed > 0:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Batch create scrambled versions of validation JSON files for multiple folds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 10 scrambles (seeds 1-10) for all folds (1-6)
  python batch_scramble_folds.py
  
  # Create scrambles for specific folds
  python batch_scramble_folds.py --folds 1,2,3
  
  # Create more scrambles with different seeds
  python batch_scramble_folds.py --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  
  # Verify each scrambling
  python batch_scramble_folds.py --verify
        """
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/home/simone/shared-data/fishway_ytvis',
        help='Base directory containing validation JSON files (default: /home/simone/shared-data/fishway_ytvis)'
    )
    
    parser.add_argument(
        '--folds',
        type=str,
        default='1,2,3,4,5,6',
        help='Comma-separated list of fold numbers (default: 1,2,3,4,5,6)'
    )
    
    parser.add_argument(
        '--seeds',
        type=str,
        default='1,2,3,4,5,6,7,8,9,10',
        help='Comma-separated list of seed values (default: 1,2,3,4,5,6,7,8,9,10)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify each scrambling after completion (slower but more thorough)'
    )
    
    args = parser.parse_args()
    
    # Parse folds and seeds
    try:
        folds = [int(f.strip()) for f in args.folds.split(',')]
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError as e:
        print(f"Error: Invalid format for folds or seeds. Use comma-separated integers.")
        print(f"  Example: --folds 1,2,3 --seeds 1,2,3,4,5")
        return 1
    
    # Check base directory exists
    base_path = Path(args.base_dir)
    if not base_path.exists():
        print(f"Error: Base directory does not exist: {base_path}")
        return 1
    
    # Run batch scrambling
    return batch_scramble_folds(args.base_dir, folds, seeds, args.verify)


if __name__ == "__main__":
    exit(main())

