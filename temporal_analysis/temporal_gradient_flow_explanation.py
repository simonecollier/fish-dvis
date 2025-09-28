import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

def create_temporal_gradient_flow_diagram():
    """
    Create a visual diagram showing how frame order affects temporal gradients in DVIS-DAQ.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original order diagram
    ax1.set_title('Original Frame Order: [Frame1 → Frame2 → Frame3]', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Frame boxes
    frame1_box = FancyBboxPatch((1, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='blue', linewidth=2)
    frame2_box = FancyBboxPatch((4, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', edgecolor='green', linewidth=2)
    frame3_box = FancyBboxPatch((7, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                               facecolor='lightcoral', edgecolor='red', linewidth=2)
    
    ax1.add_patch(frame1_box)
    ax1.add_patch(frame2_box)
    ax1.add_patch(frame3_box)
    
    # Tracker outputs
    out1_box = FancyBboxPatch((1, 5), 2, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor='blue', alpha=0.7)
    out2_box = FancyBboxPatch((4, 5), 2, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightgreen', edgecolor='green', alpha=0.7)
    out3_box = FancyBboxPatch((7, 5), 2, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightcoral', edgecolor='red', alpha=0.7)
    
    ax1.add_patch(out1_box)
    ax1.add_patch(out2_box)
    ax1.add_patch(out3_box)
    
    # Loss boxes
    loss1_box = FancyBboxPatch((1, 3), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='yellow', edgecolor='orange', linewidth=2)
    loss2_box = FancyBboxPatch((4, 3), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='yellow', edgecolor='orange', linewidth=2)
    loss3_box = FancyBboxPatch((7, 3), 2, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='yellow', edgecolor='orange', linewidth=2)
    
    ax1.add_patch(loss1_box)
    ax1.add_patch(loss2_box)
    ax1.add_patch(loss3_box)
    
    # Temporal dependencies (arrows)
    ax1.arrow(2, 6.5, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax1.arrow(5, 6.5, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Gradient flow (dashed arrows)
    ax1.arrow(2, 5.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='red', ec='red', 
             linewidth=2, linestyle='--', alpha=0.7)
    ax1.arrow(5, 5.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='red', ec='red', 
             linewidth=2, linestyle='--', alpha=0.7)
    ax1.arrow(8, 5.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='red', ec='red', 
             linewidth=2, linestyle='--', alpha=0.7)
    
    # Cross-dependency gradient (from loss3 to out2)
    ax1.arrow(8, 3.5, -3, 1.5, head_width=0.1, head_length=0.1, fc='purple', ec='purple', 
             linewidth=2, linestyle=':', alpha=0.8)
    
    # Labels
    ax1.text(2, 7.75, 'Frame 1', ha='center', va='center', fontweight='bold')
    ax1.text(5, 7.75, 'Frame 2', ha='center', va='center', fontweight='bold')
    ax1.text(8, 7.75, 'Frame 3', ha='center', va='center', fontweight='bold')
    
    ax1.text(2, 5.5, 'Output 1', ha='center', va='center', fontsize=10)
    ax1.text(5, 5.5, 'Output 2', ha='center', va='center', fontsize=10)
    ax1.text(8, 5.5, 'Output 3', ha='center', va='center', fontsize=10)
    
    ax1.text(2, 3.4, 'Loss 1', ha='center', va='center', fontsize=10)
    ax1.text(5, 3.4, 'Loss 2', ha='center', va='center', fontsize=10)
    ax1.text(8, 3.4, 'Loss 3', ha='center', va='center', fontsize=10)
    
    # Gradient equations
    ax1.text(1, 1.5, r'$\frac{\partial L_3}{\partial Frame_2} = \frac{\partial L_3}{\partial Output_3} \cdot \frac{\partial Output_3}{\partial Output_2} \cdot \frac{\partial Output_2}{\partial Frame_2}$', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    # Shuffled order diagram
    ax2.set_title('Shuffled Frame Order: [Frame3 → Frame1 → Frame2]', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Frame boxes (shuffled)
    frame3_box_shuffled = FancyBboxPatch((1, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                                        facecolor='lightcoral', edgecolor='red', linewidth=2)
    frame1_box_shuffled = FancyBboxPatch((4, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                                        facecolor='lightblue', edgecolor='blue', linewidth=2)
    frame2_box_shuffled = FancyBboxPatch((7, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                                        facecolor='lightgreen', edgecolor='green', linewidth=2)
    
    ax2.add_patch(frame3_box_shuffled)
    ax2.add_patch(frame1_box_shuffled)
    ax2.add_patch(frame2_box_shuffled)
    
    # Tracker outputs (shuffled)
    out3_box_shuffled = FancyBboxPatch((1, 5), 2, 1, boxstyle="round,pad=0.1", 
                                      facecolor='lightcoral', edgecolor='red', alpha=0.7)
    out1_box_shuffled = FancyBboxPatch((4, 5), 2, 1, boxstyle="round,pad=0.1", 
                                      facecolor='lightblue', edgecolor='blue', alpha=0.7)
    out2_box_shuffled = FancyBboxPatch((7, 5), 2, 1, boxstyle="round,pad=0.1", 
                                      facecolor='lightgreen', edgecolor='green', alpha=0.7)
    
    ax2.add_patch(out3_box_shuffled)
    ax2.add_patch(out1_box_shuffled)
    ax2.add_patch(out2_box_shuffled)
    
    # Loss boxes (shuffled)
    loss3_box_shuffled = FancyBboxPatch((1, 3), 2, 0.8, boxstyle="round,pad=0.1", 
                                       facecolor='yellow', edgecolor='orange', linewidth=2)
    loss1_box_shuffled = FancyBboxPatch((4, 3), 2, 0.8, boxstyle="round,pad=0.1", 
                                       facecolor='yellow', edgecolor='orange', linewidth=2)
    loss2_box_shuffled = FancyBboxPatch((7, 3), 2, 0.8, boxstyle="round,pad=0.1", 
                                       facecolor='yellow', edgecolor='orange', linewidth=2)
    
    ax2.add_patch(loss3_box_shuffled)
    ax2.add_patch(loss1_box_shuffled)
    ax2.add_patch(loss2_box_shuffled)
    
    # Temporal dependencies (arrows) - different pattern
    ax2.arrow(2, 6.5, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax2.arrow(5, 6.5, 2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Gradient flow (dashed arrows)
    ax2.arrow(2, 5.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='red', ec='red', 
             linewidth=2, linestyle='--', alpha=0.7)
    ax2.arrow(5, 5.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='red', ec='red', 
             linewidth=2, linestyle='--', alpha=0.7)
    ax2.arrow(8, 5.5, 0, -1.5, head_width=0.1, head_length=0.1, fc='red', ec='red', 
             linewidth=2, linestyle='--', alpha=0.7)
    
    # Cross-dependency gradient (from loss2 to out1) - DIFFERENT!
    ax2.arrow(8, 3.5, -3, 1.5, head_width=0.1, head_length=0.1, fc='purple', ec='purple', 
             linewidth=2, linestyle=':', alpha=0.8)
    
    # Labels (shuffled)
    ax2.text(2, 7.75, 'Frame 3', ha='center', va='center', fontweight='bold')
    ax2.text(5, 7.75, 'Frame 1', ha='center', va='center', fontweight='bold')
    ax2.text(8, 7.75, 'Frame 2', ha='center', va='center', fontweight='bold')
    
    ax2.text(2, 5.5, 'Output 3', ha='center', va='center', fontsize=10)
    ax2.text(5, 5.5, 'Output 1', ha='center', va='center', fontsize=10)
    ax2.text(8, 5.5, 'Output 2', ha='center', va='center', fontsize=10)
    
    ax2.text(2, 3.4, 'Loss 3', ha='center', va='center', fontsize=10)
    ax2.text(5, 3.4, 'Loss 1', ha='center', va='center', fontsize=10)
    ax2.text(8, 3.4, 'Loss 2', ha='center', va='center', fontsize=10)
    
    # Different gradient equation
    ax2.text(1, 1.5, r'$\frac{\partial L_2}{\partial Frame_1} = \frac{\partial L_2}{\partial Output_2} \cdot \frac{\partial Output_2}{\partial Output_1} \cdot \frac{\partial Output_1}{\partial Frame_1}$', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    plt.tight_layout()
    plt.savefig('temporal_gradient_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def explain_temporal_dependencies():
    """
    Text explanation of how temporal dependencies work in DVIS-DAQ.
    """
    print("="*80)
    print("HOW FRAME ORDER AFFECTS TEMPORAL GRADIENTS IN DVIS-DAQ")
    print("="*80)
    
    print("\n1. SEQUENTIAL PROCESSING IN THE TRACKER:")
    print("   - Frame 1: processed independently (no temporal reference)")
    print("   - Frame 2: uses Frame 1's outputs as reference")
    print("   - Frame 3: uses Frame 2's outputs as reference")
    print("   - This creates a chain of temporal dependencies")
    
    print("\n2. LOSS FUNCTION STRUCTURE:")
    print("   - Loss is computed frame-by-frame: L = L1 + L2 + L3")
    print("   - Each Li depends on the tracker output for frame i")
    print("   - But tracker outputs depend on temporal order!")
    
    print("\n3. GRADIENT COMPUTATION:")
    print("   - ∂L/∂Frame1 = ∂L1/∂Output1 × ∂Output1/∂Frame1")
    print("   - ∂L/∂Frame2 = ∂L2/∂Output2 × ∂Output2/∂Frame2 +")
    print("                  ∂L3/∂Output3 × ∂Output3/∂Output2 × ∂Output2/∂Frame2")
    print("   - ∂L/∂Frame3 = ∂L3/∂Output3 × ∂Output3/∂Frame3")
    
    print("\n4. KEY INSIGHT:")
    print("   - Frame 2's gradient depends on Frame 3's loss")
    print("   - This is because Frame 3 uses Frame 2's outputs as reference")
    print("   - Shuffling frames changes these dependencies")
    
    print("\n5. WHY SHUFFLING MATTERS:")
    print("   - Original: [F1→F2→F3] → Frame2 gradient depends on L3")
    print("   - Shuffled: [F3→F1→F2] → Frame1 gradient depends on L2")
    print("   - Different temporal dependencies = different gradients")
    
    print("\n6. IMPLICATIONS:")
    print("   - If model uses temporal information: gradients will differ significantly")
    print("   - If model is appearance-based: gradients will be similar")
    print("   - This is why frame shuffling analysis works!")

def demonstrate_gradient_differences():
    """
    Demonstrate how gradient patterns differ between original and shuffled sequences.
    """
    print("\n" + "="*80)
    print("GRADIENT PATTERN DIFFERENCES")
    print("="*80)
    
    # Simulate gradient patterns
    np.random.seed(42)
    
    # Original sequence gradients (with temporal dependencies)
    original_gradients = np.array([0.8, 1.2, 0.9])  # Frame 2 has higher gradient due to L3 dependency
    
    # Shuffled sequence gradients (different temporal dependencies)
    shuffled_gradients = np.array([0.9, 0.7, 1.1])  # Different pattern due to changed dependencies
    
    # Calculate similarity
    similarity = np.corrcoef(original_gradients, shuffled_gradients)[0, 1]
    
    print(f"\nOriginal sequence gradients: {original_gradients}")
    print(f"Shuffled sequence gradients:  {shuffled_gradients}")
    print(f"Correlation: {similarity:.3f}")
    
    if similarity < 0.8:
        print("→ LOW CORRELATION: Model likely uses temporal information")
    else:
        print("→ HIGH CORRELATION: Model likely appearance-based")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original gradients
    ax1.bar(['Frame 1', 'Frame 2', 'Frame 3'], original_gradients, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax1.set_title('Original Sequence Gradients\n[F1→F2→F3]')
    ax1.set_ylabel('Gradient Magnitude')
    ax1.text(1, 1.3, 'High due to L3\ndependency', ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    # Shuffled gradients
    ax2.bar(['Frame 3', 'Frame 1', 'Frame 2'], shuffled_gradients, color=['lightcoral', 'lightblue', 'lightgreen'])
    ax2.set_title('Shuffled Sequence Gradients\n[F3→F1→F2]')
    ax2.set_ylabel('Gradient Magnitude')
    ax2.text(1, 1.2, 'High due to L2\ndependency', ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('gradient_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_temporal_gradient_flow_diagram()
    explain_temporal_dependencies()
    demonstrate_gradient_differences()
