import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

# Define the fish species names (classes 0-4)
fish_species = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]

def plot_training_loss(model_dir):
    """
    Plot training loss curves by fish species.
    
    Args:
        model_dir: Path to the model directory containing metrics.json
    """
    # Path to the metrics file
    metrics_path = os.path.join(model_dir, "metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"Warning: metrics.json not found in {model_dir}")
        return False

    # Initialize data structures
    iterations = []
    total_losses = []
    loss_ce = []
    loss_dice = []
    loss_mask = []

    # Class-specific losses
    loss_ce_by_class = {i: [] for i in range(5)}
    loss_dice_by_class = {i: [] for i in range(5)}
    loss_mask_by_class = {i: [] for i in range(5)}

    # Read the metrics file
    with open(metrics_path, "r") as f:
        for line in f:
            data = json.loads(line)
            
            # Only process lines with iteration data (skip evaluation lines)
            if "iteration" in data and "total_loss" in data:
                iterations.append(data["iteration"])
                total_losses.append(data["total_loss"])
                
                # Overall losses
                if "loss_ce" in data:
                    loss_ce.append(data["loss_ce"])
                if "loss_dice" in data:
                    loss_dice.append(data["loss_dice"])
                if "loss_mask" in data:
                    loss_mask.append(data["loss_mask"])
                
                # Class-specific losses
                for i in range(5):
                    if f"loss_ce_{i}" in data:
                        loss_ce_by_class[i].append(data[f"loss_ce_{i}"])
                    if f"loss_dice_{i}" in data:
                        loss_dice_by_class[i].append(data[f"loss_dice_{i}"])
                    if f"loss_mask_{i}" in data:
                        loss_mask_by_class[i].append(data[f"loss_mask_{i}"])

    # Create subplots for different loss types
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Loss Metrics by Fish Species', fontsize=16, fontweight='bold')

    # 1. Total Loss
    ax1 = axes[0, 0]
    ax1.plot(iterations, total_losses, 'k-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cross-Entropy Loss by Species
    ax2 = axes[0, 1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(5):
        if len(loss_ce_by_class[i]) > 0:
            ax2.plot(iterations[:len(loss_ce_by_class[i])], loss_ce_by_class[i], 
                    color=colors[i], linewidth=2, label=fish_species[i])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_title('Cross-Entropy Loss by Species')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Dice Loss by Species
    ax3 = axes[1, 0]
    for i in range(5):
        if len(loss_dice_by_class[i]) > 0:
            ax3.plot(iterations[:len(loss_dice_by_class[i])], loss_dice_by_class[i], 
                    color=colors[i], linewidth=2, label=fish_species[i])
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Dice Loss')
    ax3.set_title('Dice Loss by Species')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Mask Loss by Species
    ax4 = axes[1, 1]
    for i in range(5):
        if len(loss_mask_by_class[i]) > 0:
            ax4.plot(iterations[:len(loss_mask_by_class[i])], loss_mask_by_class[i], 
                    color=colors[i], linewidth=2, label=fish_species[i])
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Mask Loss')
    ax4.set_title('Mask Loss by Species')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_loss_curves_by_species.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()  # Close to avoid showing plot when called from analysis script

    # Create a summary plot showing only the most important metric (Cross-Entropy) for each species
    plt.figure(figsize=(12, 8))

    for i in range(5):
        if len(loss_ce_by_class[i]) > 0:
            plt.plot(iterations[:len(loss_ce_by_class[i])], loss_ce_by_class[i], 
                    color=colors[i], linewidth=3, label=fish_species[i])

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Cross-Entropy Loss by Fish Species (Main Training Metric)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "ce_loss_by_species.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()  # Close to avoid showing plot when called from analysis script

    print(f"Training loss plots saved with {len(iterations)} iterations of training data")
    print(f"Fish species: {fish_species}")
    print("\nInterpretation:")
    print("- Cross-Entropy Loss: Measures classification accuracy for each species")
    print("- Dice Loss: Measures segmentation quality (how well the model outlines fish)")
    print("- Mask Loss: Measures mask prediction accuracy")
    print("- Lower values indicate better performance")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Plot training loss curves by fish species')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to the model directory containing metrics.json')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory {args.model_dir} does not exist")
        sys.exit(1)
    
    success = plot_training_loss(args.model_dir)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()