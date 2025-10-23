#!/usr/bin/env python3
"""
Script to create simulated temporal attention maps with different pattern types.

This script generates 150x150 pixel heatmaps with three different patterns:
- Diagonal: Higher values near the diagonal line (top-left to bottom-right)
- Vertical: Higher values near the 30th column with smooth falloff
- Noisy: Random heatmap values
"""

import argparse
import os
import sys
import random
import math
from PIL import Image, ImageDraw
import numpy as np


def create_diagonal_pattern(size=150, noise_level=0.1, max_value=1.0):
    """
    Create a diagonal pattern heatmap with smooth falloff from the diagonal line.
    
    Args:
        size (int): Size of the square heatmap
        noise_level (float): Amount of noise to add (0.0-1.0)
        max_value (float): Maximum value for the heatmap
        
    Returns:
        numpy.ndarray: 2D array of heatmap values
    """
    heatmap = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            # Calculate distance from diagonal line (y = x)
            # Diagonal line goes from (0,0) to (size-1, size-1)
            diagonal_distance = abs(i - j) / math.sqrt(2)
            
            # Create smooth falloff - closer to diagonal = higher value
            max_distance = size / math.sqrt(2)  # Maximum possible distance
            normalized_distance = diagonal_distance / max_distance
            
            # Exponential falloff for smooth transition
            base_value = max_value * math.exp(-3 * normalized_distance)
            
            # Add noise
            noise = random.uniform(-noise_level, noise_level)
            value = max(0, min(max_value, base_value + noise))
            
            heatmap[i, j] = value
    
    return heatmap


def create_vertical_pattern(size=150, target_column=30, noise_level=0.1, max_value=1.0):
    """
    Create a vertical pattern heatmap with higher values near the target column.
    
    Args:
        size (int): Size of the square heatmap
        target_column (int): Column with highest values (0-indexed)
        noise_level (float): Amount of noise to add (0.0-1.0)
        max_value (float): Maximum value for the heatmap
        
    Returns:
        numpy.ndarray: 2D array of heatmap values
    """
    heatmap = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            # Calculate distance from target column
            column_distance = abs(j - target_column)
            
            # Create smooth falloff - closer to target column = higher value
            max_distance = max(target_column, size - 1 - target_column)
            normalized_distance = column_distance / max_distance
            
            # Exponential falloff for smooth transition
            base_value = max_value * math.exp(-3 * normalized_distance)
            
            # Add noise
            noise = random.uniform(-noise_level, noise_level)
            value = max(0, min(max_value, base_value + noise))
            
            heatmap[i, j] = value
    
    return heatmap


def create_noisy_pattern(size=150, max_value=1.0):
    """
    Create a random noisy pattern heatmap.
    
    Args:
        size (int): Size of the square heatmap
        max_value (float): Maximum value for the heatmap
        
    Returns:
        numpy.ndarray: 2D array of heatmap values
    """
    heatmap = np.random.random((size, size)) * max_value
    return heatmap


def heatmap_to_image(heatmap, colormap='yellow-red'):
    """
    Convert heatmap values to an RGB image.
    
    Args:
        heatmap (numpy.ndarray): 2D array of heatmap values (0.0-1.0)
        colormap (str): Colormap to use ('yellow-red')
        
    Returns:
        PIL.Image: RGB image of the heatmap
    """
    size = heatmap.shape[0]
    image = Image.new('RGB', (size, size))
    
    for i in range(size):
        for j in range(size):
            value = heatmap[i, j]
            
            if colormap == 'yellow-red':
                # Yellow to red gradient
                if value < 0.5:
                    # Yellow to orange transition (low values)
                    t = value * 2  # Scale to 0-1 for low values
                    red = 255
                    green = int(255 - (255 - 165) * t)  # 255 to 165
                    blue = 0
                else:
                    # Orange to red transition (high values)
                    t = (value - 0.5) * 2  # Scale to 0-1 for high values
                    red = 255
                    green = int(165 - 165 * t)  # 165 to 0
                    blue = 0
                
                color = (red, green, blue)
            else:
                # Default grayscale
                intensity = int(255 * value)
                color = (intensity, intensity, intensity)
            
            image.putpixel((j, i), color)  # Note: (x, y) = (j, i)
    
    return image


def create_temporal_attention_map(pattern='diagonal', size=150, output_path='temporal_attention_map.jpg',
                                noise_level=0.1, target_column=30, max_value=1.0):
    """
    Create a temporal attention map with the specified pattern.
    
    Args:
        pattern (str): Pattern type ('diagonal', 'vertical', 'noisy')
        size (int): Size of the square heatmap
        output_path (str): Path to save the output image
        noise_level (float): Amount of noise to add (0.0-1.0)
        target_column (int): Column with highest values for vertical pattern
        max_value (float): Maximum value for the heatmap
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Creating {pattern} temporal attention map ({size}x{size})...")
        
        # Generate heatmap based on pattern
        if pattern == 'diagonal':
            heatmap = create_diagonal_pattern(size, noise_level, max_value)
            print(f"Diagonal pattern with noise level: {noise_level}")
        elif pattern == 'vertical':
            heatmap = create_vertical_pattern(size, target_column, noise_level, max_value)
            print(f"Vertical pattern with target column: {target_column}, noise level: {noise_level}")
        elif pattern == 'noisy':
            heatmap = create_noisy_pattern(size, max_value)
            print(f"Noisy pattern with random values")
        else:
            print(f"Error: Unknown pattern '{pattern}'. Use 'diagonal', 'vertical', or 'noisy'.")
            return False
        
        # Convert heatmap to image
        image = heatmap_to_image(heatmap)
        
        # Save the result
        print(f"Saving temporal attention map to: {output_path}")
        image.save(output_path, "JPEG", quality=95)
        
        # Print statistics
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        mean_val = np.mean(heatmap)
        print(f"Heatmap statistics - Min: {min_val:.3f}, Max: {max_val:.3f}, Mean: {mean_val:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error creating temporal attention map: {str(e)}")
        return False


def main():
    """Main function to handle command line arguments and create the temporal attention map."""
    parser = argparse.ArgumentParser(
        description="Create simulated temporal attention maps with different patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create diagonal pattern
  python simulated_temporal_attention_maps.py --pattern diagonal
  
  # Create vertical pattern with custom target column
  python simulated_temporal_attention_maps.py --pattern vertical --target-column 50
  
  # Create noisy pattern with custom size
  python simulated_temporal_attention_maps.py --pattern noisy --size 200
  
  # Create diagonal pattern with high noise
  python simulated_temporal_attention_maps.py --pattern diagonal --noise-level 0.3
        """
    )
    
    parser.add_argument("--pattern", choices=['diagonal', 'vertical', 'noisy'], default='noisy',
                       help="Pattern type for the temporal attention map (default: diagonal)")
    parser.add_argument("--size", type=int, default=150,
                       help="Size of the square heatmap in pixels (default: 150)")
    parser.add_argument("--output", type=str, default='temporal_attention_map.jpg',
                       help="Output file path (default: temporal_attention_map.jpg)")
    parser.add_argument("--noise-level", type=float, default=0.1,
                       help="Amount of noise to add (0.0-1.0, default: 0.1)")
    parser.add_argument("--target-column", type=int, default=30,
                       help="Target column for vertical pattern (0-indexed, default: 30)")
    parser.add_argument("--max-value", type=float, default=1.0,
                       help="Maximum value for the heatmap (default: 1.0)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.size <= 0:
        print("Error: size must be positive")
        sys.exit(1)
    
    if not (0.0 <= args.noise_level <= 1.0):
        print("Error: noise-level must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not (0 <= args.target_column < args.size):
        print(f"Error: target-column must be between 0 and {args.size-1}")
        sys.exit(1)
    
    if not (0.0 < args.max_value <= 1.0):
        print("Error: max-value must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Create the temporal attention map
    success = create_temporal_attention_map(
        args.pattern, args.size, args.output, 
        args.noise_level, args.target_column, args.max_value
    )
    
    if success:
        print("Temporal attention map created successfully!")
    else:
        print("Failed to create temporal attention map!")
        sys.exit(1)


if __name__ == "__main__":
    main()
