import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

from utils.model_loader import DVISModelLoader


class DirectMotionAnalyzer:
    """
    Direct analysis of temporal components in DVIS-DAQ to determine if motion is used for classification.
    Based on our understanding that temporal information flows through the tracker before classification.
    """
    
    def __init__(self, config_path: str, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_loader = DVISModelLoader(config_path, model_path, device)
        self.model = self.model_loader.model
        self.model.eval()
        
        # Storage for analysis
        self.cross_attention_weights = []
        self.reference_embeddings = []
        self.classification_logits = []
        self.temporal_hooks = []
        
        # Register hooks for temporal analysis
        self.register_temporal_hooks()
        
    def register_temporal_hooks(self):
        """Register hooks to capture temporal information flow."""
        print("Registering temporal analysis hooks...")
        
        def cross_attention_hook(module, input, output):
            """Capture cross-attention weights between frames."""
            if hasattr(output, 'attn_weights'):
                self.cross_attention_weights.append(output.attn_weights.detach().cpu())
            elif isinstance(output, tuple) and len(output) > 1:
                if hasattr(output[1], 'shape'):
                    self.cross_attention_weights.append(output[1].detach().cpu())
        
        def reference_hook(module, input, output):
            """Capture reference embeddings from previous frames."""
            if hasattr(module, 'last_reference') and module.last_reference is not None:
                self.reference_embeddings.append(module.last_reference.detach().cpu())
        
        def classification_hook(module, input, output):
            """Capture classification logits."""
            if isinstance(output, dict) and 'pred_logits' in output:
                self.classification_logits.append(output['pred_logits'].detach().cpu())
        
        # Register hooks on temporal components
        for name, module in self.model.tracker.named_modules():
            if 'cross_attention' in name.lower():
                hook = module.register_forward_hook(cross_attention_hook)
                self.temporal_hooks.append(hook)
                print(f"Registered cross-attention hook on: {name}")
        
        # Register hook on tracker for references
        hook = self.model.tracker.register_forward_hook(reference_hook)
        self.temporal_hooks.append(hook)
        
        # Register hook on final output for classification
        hook = self.model.register_forward_hook(classification_hook)
        self.temporal_hooks.append(hook)
        
        print(f"Registered {len(self.temporal_hooks)} temporal analysis hooks")
    
    def clear_hooks(self):
        """Clear all registered hooks."""
        for hook in self.temporal_hooks:
            hook.remove()
        self.temporal_hooks.clear()
        self.cross_attention_weights.clear()
        self.reference_embeddings.clear()
        self.classification_logits.clear()
    
    def analyze_sequential_vs_parallel(self, frames: List[torch.Tensor]) -> Dict:
        """
        Compare sequential processing (with temporal dependencies) vs parallel processing.
        This directly tests if temporal order matters for classification.
        """
        print("Analyzing sequential vs parallel processing...")
        
        # Method 1: Sequential processing (normal)
        self.clear_hooks()
        self.register_temporal_hooks()
        
        with torch.no_grad():
            # Process frames sequentially (normal way)
            sequential_output = self.model([{"image": frames}])
        
        sequential_logits = self.classification_logits[-1] if self.classification_logits else None
        sequential_attention = self.cross_attention_weights.copy()
        sequential_references = self.reference_embeddings.copy()
        
        # Method 2: Parallel processing (no temporal dependencies)
        self.clear_hooks()
        self.register_temporal_hooks()
        
        # Modify the model to process frames in parallel
        # This requires temporarily disabling temporal dependencies
        original_forward = self.model.tracker.forward
        
        def parallel_forward(*args, **kwargs):
            """Modified forward that processes frames in parallel."""
            # This is a simplified version - would need more sophisticated modification
            return original_forward(*args, **kwargs)
        
        self.model.tracker.forward = parallel_forward
        
        with torch.no_grad():
            parallel_output = self.model([{"image": frames}])
        
        # Restore original forward
        self.model.tracker.forward = original_forward
        
        parallel_logits = self.classification_logits[-1] if self.classification_logits else None
        parallel_attention = self.cross_attention_weights.copy()
        parallel_references = self.reference_embeddings.copy()
        
        # Compare results
        comparison = {
            'sequential_logits': sequential_logits,
            'parallel_logits': parallel_logits,
            'logit_similarity': None,
            'attention_difference': None,
            'reference_difference': None
        }
        
        if sequential_logits is not None and parallel_logits is not None:
            # Compute similarity between logits
            logit_similarity = F.cosine_similarity(
                sequential_logits.flatten(), 
                parallel_logits.flatten(), 
                dim=0
            ).item()
            comparison['logit_similarity'] = logit_similarity
            
            print(f"Logit similarity (sequential vs parallel): {logit_similarity:.4f}")
            
            if logit_similarity > 0.95:
                print("→ High similarity suggests appearance-based classification")
            elif logit_similarity < 0.7:
                print("→ Low similarity suggests temporal classification")
            else:
                print("→ Moderate similarity suggests mixed approach")
        
        return comparison
    
    def analyze_cross_attention_patterns(self, frames: List[torch.Tensor]) -> Dict:
        """
        Analyze cross-attention patterns to see if temporal information flows to classification.
        """
        print("Analyzing cross-attention patterns...")
        
        self.clear_hooks()
        self.register_temporal_hooks()
        
        with torch.no_grad():
            output = self.model([{"image": frames}])
        
        if not self.cross_attention_weights:
            print("No cross-attention weights captured")
            return {}
        
        # Analyze attention patterns
        attention_analysis = {
            'attention_weights': self.cross_attention_weights,
            'temporal_attention_strength': [],
            'frame_to_frame_attention': [],
            'attention_evolution': []
        }
        
        # Compute temporal attention strength
        for i, attn_weights in enumerate(self.cross_attention_weights):
            if len(attn_weights.shape) >= 3:
                # Compute how much attention is paid to temporal relationships
                temporal_strength = attn_weights.mean().item()
                attention_analysis['temporal_attention_strength'].append(temporal_strength)
                
                # Analyze frame-to-frame attention
                if attn_weights.shape[-1] > 1:  # Multiple frames
                    frame_attention = attn_weights.mean(dim=(0, 1)).cpu().numpy()
                    attention_analysis['frame_to_frame_attention'].append(frame_attention)
        
        # Compute attention evolution over time
        if attention_analysis['temporal_attention_strength']:
            attention_analysis['attention_evolution'] = np.array(
                attention_analysis['temporal_attention_strength']
            )
        
        return attention_analysis
    
    def analyze_reference_embeddings(self, frames: List[torch.Tensor]) -> Dict:
        """
        Analyze how reference embeddings from previous frames affect classification.
        """
        print("Analyzing reference embeddings...")
        
        self.clear_hooks()
        self.register_temporal_hooks()
        
        with torch.no_grad():
            output = self.model([{"image": frames}])
        
        if not self.reference_embeddings:
            print("No reference embeddings captured")
            return {}
        
        reference_analysis = {
            'reference_embeddings': self.reference_embeddings,
            'reference_similarity': [],
            'reference_impact': []
        }
        
        # Analyze reference embedding evolution
        for i in range(1, len(self.reference_embeddings)):
            prev_ref = self.reference_embeddings[i-1]
            curr_ref = self.reference_embeddings[i]
            
            # Compute similarity between consecutive references
            similarity = F.cosine_similarity(
                prev_ref.flatten(), 
                curr_ref.flatten(), 
                dim=0
            ).item()
            reference_analysis['reference_similarity'].append(similarity)
        
        # Analyze impact of references on classification
        if self.classification_logits:
            logits = self.classification_logits[-1]
            # This would require more sophisticated analysis to correlate references with logits
        
        return reference_analysis
    
    def test_temporal_ablation(self, frames: List[torch.Tensor]) -> Dict:
        """
        Test what happens when we remove temporal dependencies.
        """
        print("Testing temporal ablation...")
        
        # Store original model state
        original_tracker_state = {}
        for name, param in self.model.tracker.named_parameters():
            original_tracker_state[name] = param.data.clone()
        
        # Method 1: Zero out cross-attention weights
        ablation_results = {}
        
        for name, module in self.model.tracker.named_modules():
            if 'cross_attention' in name.lower() and hasattr(module, 'multihead_attn'):
                # Temporarily zero out attention weights
                original_weight = module.multihead_attn.in_proj_weight.data.clone()
                module.multihead_attn.in_proj_weight.data.zero_()
                
                with torch.no_grad():
                    ablated_output = self.model([{"image": frames}])
                
                # Restore original weights
                module.multihead_attn.in_proj_weight.data = original_weight
                
                ablation_results[f'ablated_{name}'] = ablated_output
        
        # Restore original model state
        for name, param in self.model.tracker.named_parameters():
            param.data = original_tracker_state[name]
        
        return ablation_results
    
    def visualize_temporal_analysis(self, analysis_results: Dict, output_dir: str = "temporal_analysis_outputs"):
        """
        Visualize the temporal analysis results.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Cross-attention patterns
        if 'attention_evolution' in analysis_results:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            attention_evolution = analysis_results['attention_evolution']
            plt.plot(attention_evolution)
            plt.title('Cross-Attention Strength Over Time')
            plt.xlabel('Frame')
            plt.ylabel('Attention Strength')
            
            plt.subplot(2, 2, 2)
            if analysis_results.get('frame_to_frame_attention'):
                frame_attention = analysis_results['frame_to_frame_attention'][0]
                plt.imshow(frame_attention, cmap='viridis')
                plt.title('Frame-to-Frame Attention Matrix')
                plt.colorbar()
            
            plt.subplot(2, 2, 3)
            if 'reference_similarity' in analysis_results:
                ref_similarity = analysis_results['reference_similarity']
                plt.plot(ref_similarity)
                plt.title('Reference Embedding Similarity')
                plt.xlabel('Frame')
                plt.ylabel('Similarity')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'temporal_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Sequential vs Parallel comparison
        if 'logit_similarity' in analysis_results:
            plt.figure(figsize=(8, 6))
            similarity = analysis_results['logit_similarity']
            
            plt.bar(['Sequential vs Parallel'], [similarity])
            plt.title('Classification Similarity: Sequential vs Parallel Processing')
            plt.ylabel('Cosine Similarity')
            plt.ylim(0, 1)
            
            # Add interpretation
            if similarity > 0.95:
                interpretation = "Appearance-Based"
                color = 'green'
            elif similarity < 0.7:
                interpretation = "Temporal-Based"
                color = 'red'
            else:
                interpretation = "Mixed"
                color = 'orange'
            
            plt.text(0, similarity + 0.02, interpretation, ha='center', color=color, fontweight='bold')
            
            plt.savefig(os.path.join(output_dir, 'sequential_vs_parallel.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_comprehensive_analysis(self, frames: List[torch.Tensor]) -> Dict:
        """
        Run all temporal analysis methods and provide comprehensive results.
        """
        print("Running comprehensive temporal analysis...")
        
        results = {}
        
        # 1. Sequential vs Parallel analysis
        print("\n1. Testing sequential vs parallel processing...")
        results['sequential_vs_parallel'] = self.analyze_sequential_vs_parallel(frames)
        
        # 2. Cross-attention analysis
        print("\n2. Analyzing cross-attention patterns...")
        results['cross_attention'] = self.analyze_cross_attention_patterns(frames)
        
        # 3. Reference embedding analysis
        print("\n3. Analyzing reference embeddings...")
        results['reference_embeddings'] = self.analyze_reference_embeddings(frames)
        
        # 4. Temporal ablation test
        print("\n4. Testing temporal ablation...")
        results['temporal_ablation'] = self.test_temporal_ablation(frames)
        
        # 5. Generate visualizations
        print("\n5. Generating visualizations...")
        self.visualize_temporal_analysis(results)
        
        # 6. Provide interpretation
        print("\n6. Providing interpretation...")
        interpretation = self.interpret_results(results)
        results['interpretation'] = interpretation
        
        return results
    
    def interpret_results(self, results: Dict) -> str:
        """
        Provide interpretation of the analysis results.
        """
        interpretation = []
        
        # Interpret sequential vs parallel results
        if 'sequential_vs_parallel' in results:
            similarity = results['sequential_vs_parallel'].get('logit_similarity')
            if similarity is not None:
                if similarity > 0.95:
                    interpretation.append("HIGH APPEARANCE-BASED CLASSIFICATION: Sequential and parallel processing produce nearly identical results, indicating the model relies primarily on static visual features.")
                elif similarity < 0.7:
                    interpretation.append("HIGH TEMPORAL CLASSIFICATION: Sequential and parallel processing produce significantly different results, indicating the model relies heavily on temporal information.")
                else:
                    interpretation.append("MIXED CLASSIFICATION: Moderate difference between sequential and parallel processing, suggesting both appearance and temporal features are used.")
        
        # Interpret cross-attention results
        if 'cross_attention' in results and results['cross_attention'].get('temporal_attention_strength'):
            avg_attention = np.mean(results['cross_attention']['temporal_attention_strength'])
            if avg_attention > 0.1:
                interpretation.append(f"STRONG TEMPORAL ATTENTION: Average cross-attention strength of {avg_attention:.3f} indicates significant temporal information flow.")
            else:
                interpretation.append(f"WEAK TEMPORAL ATTENTION: Average cross-attention strength of {avg_attention:.3f} indicates minimal temporal information flow.")
        
        # Interpret reference embedding results
        if 'reference_embeddings' in results and results['reference_embeddings'].get('reference_similarity'):
            avg_similarity = np.mean(results['reference_embeddings']['reference_similarity'])
            if avg_similarity > 0.8:
                interpretation.append("STABLE TEMPORAL REFERENCES: High reference similarity indicates consistent temporal tracking.")
            else:
                interpretation.append("VARIABLE TEMPORAL REFERENCES: Low reference similarity indicates dynamic temporal relationships.")
        
        if not interpretation:
            interpretation.append("INSUFFICIENT DATA: Could not determine classification approach from available data.")
        
        return "\n".join(interpretation)


def main():
    """
    Example usage of the DirectMotionAnalyzer.
    """
    # Configuration
    config_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/config.yaml"
    model_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth"
    
    # Initialize analyzer
    analyzer = DirectMotionAnalyzer(config_path, model_path)
    
    # Load sample frames (you would replace this with your actual frame loading)
    # frames = load_frames_from_video(video_path)
    
    # For demonstration, create dummy frames
    dummy_frames = [torch.randn(3, 480, 640) for _ in range(5)]
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(dummy_frames)
    
    # Print results
    print("\n" + "="*50)
    print("TEMPORAL ANALYSIS RESULTS")
    print("="*50)
    print(results['interpretation'])
    
    # Clean up
    analyzer.clear_hooks()


if __name__ == "__main__":
    main()
