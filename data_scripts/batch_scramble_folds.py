#!/usr/bin/env python3
"""
Batch wrapper to create multiple scrambled versions of validation JSON files.

This script creates 100 scrambled versions (seeds 1-100) for a single validation file.
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


def batch_scramble_single_file(input_file, output_base_dir, seeds, dir_pattern=None, verify=False):
    """
    Create scrambled versions for a single file with multiple seeds.
    
    Args:
        input_file: Path to input JSON file
        output_base_dir: Base directory for output files
        seeds: List of seed values (e.g., [1, 2, ..., 100])
        dir_pattern: Pattern for output directory name, with {seed} placeholder (e.g., "eval_443_all_frames_{seed}")
        verify: Whether to verify each scrambling
    """
    input_path = Path(input_file)
    output_base = Path(output_base_dir)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Create output base directory if it doesn't exist
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Default directory pattern if not provided
    if dir_pattern is None:
        dir_pattern = "eval_6059_edit91_seed{seed}"
    
    total_tasks = len(seeds)
    completed = 0
    failed = 0
    
    print(f"Starting batch scrambling:")
    print(f"  Input file: {input_path}")
    print(f"  Output base: {output_base}")
    print(f"  Directory pattern: {dir_pattern}")
    print(f"  Seeds: {seeds[0]} to {seeds[-1]} ({total_tasks} total)")
    print()
    
    for seed in seeds:
        # Construct output directory and file path
        dir_name = dir_pattern.format(seed=seed)
        output_dir = output_base / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "val.json"
        
        print(f"  Seed {seed}: {input_path.name} -> {output_file}")
        
        if run_scramble(input_path, output_file, seed, verify):
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
        description='Batch create scrambled versions of a validation JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default='/home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_edit91.json',
        help='Input JSON file to scramble (default: /home/simone/shared-data/fishway_ytvis/val_fold6_all_frames_edit91.json)'
    )
    
    parser.add_argument(
        '--output-base-dir',
        type=str,
        default='/home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6',
        help='Base directory for output files (default: /home/simone/store/simone/dvis-model-outputs/top_fold_results/camera/scrambled_fold6)'
    )
    
    parser.add_argument(
        '--seeds',
        type=str,
        default=','.join(map(str, range(1, 101))),
        help='Comma-separated list of seed values (default: 1-100)'
    )
    
    parser.add_argument(
        '--dir-pattern',
        type=str,
        default=None,
        help='Pattern for output directory name with {seed} placeholder (e.g., "eval_443_all_frames_{seed}")'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify each scrambling after completion (slower but more thorough)'
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    except ValueError as e:
        print(f"Error: Invalid format for seeds. Use comma-separated integers.")
        print(f"  Example: --seeds 1,2,3,4,5")
        return 1
    
    # Run batch scrambling
    return batch_scramble_single_file(args.input_file, args.output_base_dir, seeds, args.dir_pattern, args.verify)


if __name__ == "__main__":
    exit(main())

