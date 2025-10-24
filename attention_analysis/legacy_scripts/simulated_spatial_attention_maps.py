#!/usr/bin/env python3
"""
Script to create simulated spatial attention maps by downsampling images and overlaying numbered grids or heatmaps.

This script takes a JPEG image, downsamples it to a specified size while maintaining
aspect ratio, and either overlays a grid where each cell is 16x16 pixels with numbered labels,
or creates a spatial attention heatmap with configurable target cells and transparency.
"""

import argparse
import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def calculate_new_dimensions(original_width, original_height, max_size=512):
    """
    Calculate new dimensions while maintaining aspect ratio.
    
    Args:
        original_width (int): Original image width
        original_height (int): Original image height
        max_size (int): Maximum size for the longest side
        
    Returns:
        tuple: (new_width, new_height)
    """
    if original_width > original_height:
        # Width is the longest side
        new_width = max_size
        new_height = int((original_height * max_size) / original_width)
    else:
        # Height is the longest side
        new_height = max_size
        new_width = int((original_width * max_size) / original_height)
    
    return new_width, new_height


def create_grid_overlay(image, cell_size=16):
    """
    Create a grid overlay on the image with numbered cells.
    
    Args:
        image (PIL.Image): The image to overlay the grid on
        cell_size (int): Size of each grid cell in pixels
        
    Returns:
        PIL.Image: Image with grid overlay
    """
    # Create a copy of the image to draw on
    grid_image = image.copy()
    draw = ImageDraw.Draw(grid_image)
    
    width, height = image.size
    
    # Calculate number of grid cells
    cols = width // cell_size
    rows = height // cell_size
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 5)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Draw grid lines and numbers
    cell_number = 0
    
    for row in range(rows + 1):
        y = row * cell_size
        # Draw horizontal line
        draw.line([(0, y), (width, y)], fill="red", width=0)
        
        for col in range(cols + 1):
            x = col * cell_size
            # Draw vertical line
            draw.line([(x, 0), (x, height)], fill="red", width=0)
            
            # Add cell number (only for complete cells)
            if row < rows and col < cols:
                cell_number += 1
                # Calculate center of the cell
                center_x = x + cell_size // 2
                center_y = y + cell_size // 2
                
                # Draw cell number
                text = str(cell_number)
                if font:
                    # Get text bounding box for centering
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    text_x = center_x - text_width // 2
                    text_y = center_y - text_height // 2
                    
                    draw.text((text_x, text_y), text, fill="yellow", font=font)
                else:
                    # Use smaller offset for default font
                    draw.text((center_x - 3, center_y - 3), text, fill="yellow")
    
    return grid_image


def create_heatmap_overlay(image, cell_size=16, target_cells=None, high_value_range=(0.9, 1.0), 
                          low_value_range=(0.0, 0.5), transparency=0.7):
    """
    Create a transparent heatmap overlay for the image.
    
    Args:
        image (PIL.Image): The base image
        cell_size (int): Size of each grid cell in pixels
        target_cells (list): List of cell numbers to have high values (1-indexed)
        high_value_range (tuple): Range for high values (min, max)
        low_value_range (tuple): Range for low values (min, max)
        transparency (float): Transparency level (0.0 = fully transparent, 1.0 = opaque)
        
    Returns:
        PIL.Image: Transparent heatmap overlay
    """
    width, height = image.size
    
    # Calculate number of grid cells
    cols = width // cell_size
    rows = height // cell_size
    total_cells = cols * rows
    
    # Create a transparent overlay
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Generate heatmap values for each cell
    cell_number = 0
    for row in range(rows):
        for col in range(cols):
            cell_number += 1
            
            # Determine if this cell should have high values
            if target_cells and cell_number in target_cells:
                # High value cell
                value = random.uniform(high_value_range[0], high_value_range[1])
            else:
                # Low value cell
                value = random.uniform(low_value_range[0], low_value_range[1])
            
            # Convert value to color (yellow to red heatmap)
            # Higher values = more red, lower values = more yellow
            if value < 0.5:
                # Yellow to orange transition (low values)
                # Interpolate between yellow (255, 255, 0) and orange (255, 165, 0)
                t = value * 2  # Scale to 0-1 for low values
                red = 255
                green = int(255 - (255 - 165) * t)  # 255 to 165
                blue = 0
            else:
                # Orange to red transition (high values)
                # Interpolate between orange (255, 165, 0) and red (255, 0, 0)
                t = (value - 0.5) * 2  # Scale to 0-1 for high values
                red = 255
                green = int(165 - 165 * t)  # 165 to 0
                blue = 0
            
            alpha = int(255 * transparency)
            
            # Calculate cell boundaries
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = min(x1 + cell_size, width)
            y2 = min(y1 + cell_size, height)
            
            # Fill the cell with the heatmap color
            cell_color = (red, green, blue, alpha)
            overlay.paste(cell_color, (x1, y1, x2, y2))
    
    return overlay


def process_image(input_path, output_path, max_size=512, cell_size=16, mode='grid', 
                 target_cells=None, high_value_range=(0.9, 1.0), low_value_range=(0.0, 0.5), 
                 transparency=0.7):
    """
    Process the input image: downsample and add grid overlay or heatmap.
    
    Args:
        input_path (str): Path to input JPEG image
        output_path (str): Path to save output image
        max_size (int): Maximum size for the longest side
        cell_size (int): Size of each grid cell in pixels
        mode (str): 'grid' for grid overlay, 'heatmap' for heatmap overlay
        target_cells (list): List of cell numbers to have high values (1-indexed)
        high_value_range (tuple): Range for high values (min, max)
        low_value_range (tuple): Range for low values (min, max)
        transparency (float): Transparency level for heatmap (0.0-1.0)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return False
    
    try:
        # Load the image
        print(f"Loading image: {input_path}")
        image = Image.open(input_path)
        
        # Convert to RGB if necessary (handles different color modes)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_width, original_height = image.size
        print(f"Original dimensions: {original_width}x{original_height}")
        
        # Calculate new dimensions
        new_width, new_height = calculate_new_dimensions(original_width, original_height, max_size)
        print(f"New dimensions: {new_width}x{new_height}")
        
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        if mode == 'grid':
            # Create grid overlay
            print(f"Creating grid overlay with {cell_size}x{cell_size} cells...")
            result_image = create_grid_overlay(resized_image, cell_size)
            
            # Save the result
            print(f"Saving grid image to: {output_path}")
            result_image.save(output_path, "JPEG", quality=95)
            
        elif mode == 'heatmap':
            # Create spatial attention heatmap overlay
            print(f"Creating spatial attention heatmap with {cell_size}x{cell_size} cells...")
            if target_cells:
                print(f"Target cells with high attention values: {target_cells}")
            print(f"Transparency: {transparency}")
            
            heatmap_overlay = create_heatmap_overlay(
                resized_image, cell_size, target_cells, 
                high_value_range, low_value_range, transparency
            )
            
            # Composite the heatmap onto the original image
            result_image = Image.alpha_composite(resized_image.convert('RGBA'), heatmap_overlay)
            
            # Save the result as JPEG (converted back from RGBA)
            print(f"Saving spatial attention heatmap composite to: {output_path}")
            result_image.convert('RGB').save(output_path, "JPEG", quality=95)
            
        else:
            print(f"Error: Unknown mode '{mode}'. Use 'grid' or 'heatmap'.")
            return False
        
        # Print grid information
        cols = new_width // cell_size
        rows = new_height // cell_size
        total_cells = cols * rows
        print(f"Grid: {cols} columns x {rows} rows = {total_cells} cells")
        
        return True
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False


def main():
    """Main function to handle command line arguments and process the image."""
    parser = argparse.ArgumentParser(
        description="Create simulated spatial attention maps with grid overlay or heatmap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create grid overlay
  python simulated_spatial_attention_maps.py input.jpg output.jpg
  python simulated_spatial_attention_maps.py input.jpg output.jpg --max-size 1024
  python simulated_spatial_attention_maps.py input.jpg output.jpg --cell-size 32
  
  # Create spatial attention heatmap
  python simulated_spatial_attention_maps.py input.jpg output.jpg --mode heatmap
  python simulated_spatial_attention_maps.py input.jpg output.jpg --mode heatmap --target-cells 5 10 15
  python simulated_spatial_attention_maps.py input.jpg output.jpg --mode heatmap --transparency 0.5
        """
    )
    
    parser.add_argument("input", help="Path to input JPEG image", nargs='?',
                        default="/data/fishway_ytvis/all_videos/Ganaraska__Ganaraska 2022__09082022-09122022__22  09  08  11  03__1081/00022.jpg")
    parser.add_argument("output", help="Path to output image", nargs='?',
                        default="/home/simone/fish-dvis/visualization_scripts/Ganaraska__Ganaraska 2022__09082022-09122022__22  09  08  11  03__1081_00022_heatmap.jpg")
    parser.add_argument("--max-size", type=int, default=512,
                       help="Maximum size for the longest side (default: 512)")
    parser.add_argument("--cell-size", type=int, default=16,
                       help="Size of each grid cell in pixels (default: 16)")
    parser.add_argument("--mode", choices=['grid', 'heatmap'], default='heatmap',
                       help="Output mode: 'grid' for grid overlay, 'heatmap' for spatial attention heatmap (default: heatmap)")
    parser.add_argument("--target-cells", type=int, nargs='+', 
                       default=[390, 391, 397, 398, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
                                454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,
                                486, 487, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499,
                                522, 524, 525, 526, 527, 528, 529, 530, 531, 532,
                                558, 559, 560, 561, 562, 563, 564, 565, 566, 
                                592, 593, 594, 595, 596, 597, 598],
                       help="Cell numbers to have high values (1-indexed, space-separated, default: 390 422 454 486 564 565 567 695 696 597)")
    parser.add_argument("--high-value-range", type=float, nargs=2, default=[0.9, 1.0],
                       metavar=('MIN', 'MAX'),
                       help="Range for high values (default: 0.9 1.0)")
    parser.add_argument("--low-value-range", type=float, nargs=2, default=[0.0, 0.6],
                       metavar=('MIN', 'MAX'),
                       help="Range for low values (default: 0.0 0.5)")
    parser.add_argument("--transparency", type=float, default=0.7,
                       help="Transparency level for spatial attention heatmap (0.0-1.0, default: 0.7)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.max_size <= 0:
        print("Error: max-size must be positive")
        sys.exit(1)
    
    if args.cell_size <= 0:
        print("Error: cell-size must be positive")
        sys.exit(1)
    
    if not (0.0 <= args.transparency <= 1.0):
        print("Error: transparency must be between 0.0 and 1.0")
        sys.exit(1)
    
    if len(args.high_value_range) != 2 or not (0.0 <= args.high_value_range[0] <= args.high_value_range[1] <= 1.0):
        print("Error: high-value-range must be two values between 0.0 and 1.0, with min <= max")
        sys.exit(1)
    
    if len(args.low_value_range) != 2 or not (0.0 <= args.low_value_range[0] <= args.low_value_range[1] <= 1.0):
        print("Error: low-value-range must be two values between 0.0 and 1.0, with min <= max")
        sys.exit(1)
    
    # Process the image
    success = process_image(
        args.input, args.output, args.max_size, args.cell_size, 
        args.mode, args.target_cells, tuple(args.high_value_range), 
        tuple(args.low_value_range), args.transparency
    )
    
    if success:
        print("Image processing completed successfully!")
    else:
        print("Image processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
