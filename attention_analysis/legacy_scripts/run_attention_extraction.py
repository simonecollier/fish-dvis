#!/usr/bin/env python3
"""
Simple script to run attention extraction for DVIS-DAQ model

Usage:
    python run_attention_extraction.py --video-id VIDEO_ID --model /path/to/model.pth
    python run_attention_extraction.py --list-videos --model /path/to/model.pth
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Set environment variables like the training script
os.environ['DETECTRON2_DATASETS'] = '/data'
os.environ['PYTHONPATH'] = '/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ'

# Add the DVIS-DAQ codebase to Python path
sys.path.append('/home/simone/fish-dvis/DVIS_Plus/DVIS_DAQ')

# Add the current attention_analysis directory to Python path
sys.path.append('/home/simone/fish-dvis/attention_analysis')

from extract_attention_maps import AttentionExtractor

def main():
    parser = argparse.ArgumentParser(description='Extract attention maps from DVIS-DAQ model')
    parser.add_argument('--video-id', type=str, help='Video ID from the dataset (e.g., from val.json)')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='/store/simone/attention/', 
                       help='Output directory for attention maps (only used if --save is specified)')
    parser.add_argument('--save', action='store_true', 
                       help='Save attention maps to disk (default: only print shapes)')
    parser.add_argument('--no-print-shapes', action='store_true',
                       help='Do not print attention map shapes (default: print shapes)')
    parser.add_argument('--list-videos', action='store_true',
                       help='List available video IDs from the dataset')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Create output directory only if saving
    if args.save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = args.output
        print("Note: Attention maps will NOT be saved to disk (use --save to enable)")
    
    try:
        # Initialize extractor
        extractor = AttentionExtractor(args.model, str(output_dir))
        
        # List available videos if requested
        if args.list_videos:
            print("Available video IDs from dataset:")
            available_videos = extractor.list_available_videos()
            for i, video_id in enumerate(available_videos):
                print(f"  {i+1:3d}. {video_id}")
            print(f"\nTotal videos available: {len(available_videos)}")
            
            # Log to extractor
            extractor.log_print("Available video IDs from dataset:")
            for i, video_id in enumerate(available_videos):
                extractor.log_print(f"  {i+1:3d}. {video_id}")
            extractor.log_print(f"\nTotal videos available: {len(available_videos)}")
            
            # Save this information to file as well (overwrite, don't append)
            model_dir = Path(args.model).parent
            report_path = model_dir / "attention_maps_report.txt"
            
            # Write fresh report file
            with open(report_path, 'w') as f:
                f.write("DVIS-DAQ VIDEO LISTING REPORT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                
                f.write("Available video IDs from dataset:\n")
                for i, video_id in enumerate(available_videos):
                    f.write(f"  {i+1:3d}. {video_id}\n")
                f.write(f"\nTotal videos available: {len(available_videos)}\n")
            
            print(f"Video list saved to: {report_path}")
            return
        
        # Check if video-id is provided
        if not args.video_id:
            print("Error: Please provide --video-id or use --list-videos to see available options")
            print("\nExample usage:")
            print("  python run_attention_extraction.py --list-videos --model /path/to/model.pth")
            print("  python run_attention_extraction.py --video-id VIDEO_ID --model /path/to/model.pth")
            sys.exit(1)
        
        print(f"Starting attention extraction...")
        print(f"Video ID: {args.video_id}")
        print(f"Model: {args.model}")
        print(f"Save to disk: {args.save}")
        print(f"Print shapes: {not args.no_print_shapes}")
        
        # Log these messages to the extractor's log buffer
        extractor.log_print(f"Starting attention extraction...")
        extractor.log_print(f"Video ID: {args.video_id}")
        extractor.log_print(f"Model: {args.model}")
        extractor.log_print(f"Save to disk: {args.save}")
        extractor.log_print(f"Print shapes: {not args.no_print_shapes}")
        
        # Extract attention maps
        attention_maps = extractor.extract_attention_for_video(
            args.video_id, 
            save_attention=args.save,
            print_shapes=not args.no_print_shapes
        )
        
        # Print summary
        summary = extractor.get_attention_summary()
        print(f"\nExtraction completed successfully!")
        print(f"Total attention maps extracted: {len(summary)}")
        
        # Log to extractor
        extractor.log_print(f"\nExtraction completed successfully!")
        extractor.log_print(f"Total attention maps extracted: {len(summary)}")
        
        if args.save:
            print(f"Attention maps saved to: {output_dir}")
            extractor.log_print(f"Attention maps saved to: {output_dir}")
        
        # Print detailed information if shapes were not printed during extraction
        if args.no_print_shapes:
            print("\n" + "="*80)
            print("ATTENTION MAPS DETAILED SUMMARY")
            print("="*80)
            
            # Group attention maps by component
            components = {
                'Backbone (DINOv2 ViT-L)': [],
                'Pixel Decoder': [],
                'Transformer Decoder': [],
                'Tracker (VideoInstanceCutter)': [],
                'ReID Branch': [],
                'Temporal Refiner': []
            }
            
            for attn_name, attn_info in summary.items():
                if 'backbone' in attn_name:
                    components['Backbone (DINOv2 ViT-L)'].append((attn_name, attn_info))
                elif 'pixel_decoder' in attn_name:
                    components['Pixel Decoder'].append((attn_name, attn_info))
                elif 'decoder' in attn_name:
                    components['Transformer Decoder'].append((attn_name, attn_info))
                elif 'tracker' in attn_name or 'slot' in attn_name:
                    components['Tracker (VideoInstanceCutter)'].append((attn_name, attn_info))
                elif 'reid' in attn_name:
                    components['ReID Branch'].append((attn_name, attn_info))
                elif 'refiner' in attn_name:
                    components['Temporal Refiner'].append((attn_name, attn_info))
            
            # Print organized summary
            total_maps = 0
            for component_name, maps in components.items():
                if maps:
                    print(f"\n{component_name}:")
                    print("-" * len(component_name))
                    for attn_name, attn_info in maps:
                        shape_str = str(attn_info['shape'])
                        print(f"  {attn_name}: {shape_str}")
                        total_maps += 1
            
            print(f"\n" + "="*80)
            print(f"TOTAL ATTENTION MAPS: {total_maps}")
            print("="*80)
        
        # Print statistical information with dimension interpretations
        print("\n" + "="*80)
        print("ATTENTION MAPS STATISTICAL SUMMARY")
        print("="*80)
        
        for attn_name, attn_info in summary.items():
            # Get dimension interpretation for this attention map
            dims_origin = extractor._get_dimension_interpretation(attn_name, attn_info['shape'])
            
            print(f"\n{attn_name}:")
            print(f"  Shape: {attn_info['shape']}")
            print(f"  Dims_origin: {dims_origin}")
            print(f"  Range: [{attn_info['min_value']:.4f}, {attn_info['max_value']:.4f}]")
            print(f"  Mean: {attn_info['mean_value']:.4f}")
            print(f"  Std: {attn_info['std_value']:.4f}")
        
        # Save complete report to file
        print("\n" + "="*80)
        print("SAVING COMPLETE REPORT TO FILE")
        print("="*80)
        
        # Start with a fresh report file
        model_dir = Path(args.model).parent
        report_path = model_dir / "attention_maps_report.txt"
        
        # Clear the file first
        with open(report_path, 'w') as f:
            f.write("DVIS-DAQ ATTENTION MAPS EXTRACTION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video ID: {args.video_id}\n")
            f.write("\n")
        
        # Save detailed summary to file
        detailed_lines = []
        detailed_lines.append("\n" + "="*80)
        detailed_lines.append("ATTENTION MAPS DETAILED SUMMARY WITH DIMENSION INTERPRETATIONS")
        detailed_lines.append("="*80)
        
        for attn_name, attn_info in summary.items():
            dims_origin = extractor._get_dimension_interpretation(attn_name, attn_info['shape'])
            
            detailed_lines.append(f"\n{attn_name}:")
            detailed_lines.append(f"  Shape: {attn_info['shape']}")
            detailed_lines.append(f"  Dims_origin: {dims_origin}")
            detailed_lines.append(f"  Range: [{attn_info['min_value']:.4f}, {attn_info['max_value']:.4f}]")
            detailed_lines.append(f"  Mean: {attn_info['mean_value']:.4f}")
            detailed_lines.append(f"  Std: {attn_info['std_value']:.4f}")
        
        # Save to file
        extractor._save_detailed_summary_to_file(detailed_lines, len(summary))
        
        # Save complete execution log
        extractor.save_complete_log_to_file()
        
        print(f"Complete report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error during attention extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
