import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import json
from pathlib import Path

from utils.model_loader import DVISModelLoader


class SimpleMotionTest:
    """
    Simple and direct test to determine if DVIS-DAQ uses motion for classification.
    Tests the key hypothesis: if temporal order doesn't matter for classification,
    then shuffled frames should produce similar classification results.
    """
    
    def __init__(self, config_path: str, model_path: str, device: str = "cuda"):
        self.device = device
        self.model_loader = DVISModelLoader(config_path, model_path, device)
        self.model = self.model_loader.model
        self.model.eval()
        
    def test_frame_order_importance(self, frames: List[torch.Tensor], num_shuffles: int = 5) -> Dict:
        """
        Test if frame order matters for classification by comparing original vs shuffled sequences.
        
        Args:
            frames: List of frame tensors
            num_shuffles: Number of random shuffles to test
            
        Returns:
            Dictionary with comparison results
        """
        print(f"Testing frame order importance with {len(frames)} frames and {num_shuffles} shuffles...")
        
        # Get original classification
        with torch.no_grad():
            original_output = self.model([{"image": frames}])
            original_logits = original_output["pred_logits"]  # (b, t, q, c)
            original_probs = F.softmax(original_logits, dim=-1)
            original_predictions = torch.argmax(original_logits, dim=-1)  # (b, t, q)
        
        print(f"Original logits shape: {original_logits.shape}")
        print(f"Original predictions: {original_predictions.flatten().tolist()}")
        
        # Test shuffled sequences
        shuffled_results = []
        
        for shuffle_idx in range(num_shuffles):
            # Create shuffled frame sequence
            shuffled_frames = frames.copy()
            np.random.shuffle(shuffled_frames)
            
            with torch.no_grad():
                shuffled_output = self.model([{"image": shuffled_frames}])
                shuffled_logits = shuffled_output["pred_logits"]
                shuffled_probs = F.softmax(shuffled_logits, dim=-1)
                shuffled_predictions = torch.argmax(shuffled_logits, dim=-1)
            
            # Compare with original
            logit_similarity = F.cosine_similarity(
                original_logits.flatten(), 
                shuffled_logits.flatten(), 
                dim=0
            ).item()
            
            prob_similarity = F.cosine_similarity(
                original_probs.flatten(), 
                shuffled_probs.flatten(), 
                dim=0
            ).item()
            
            prediction_agreement = (original_predictions == shuffled_predictions).float().mean().item()
            
            shuffled_results.append({
                'shuffle_idx': shuffle_idx,
                'logit_similarity': logit_similarity,
                'prob_similarity': prob_similarity,
                'prediction_agreement': prediction_agreement,
                'shuffled_predictions': shuffled_predictions.flatten().tolist()
            })
            
            print(f"Shuffle {shuffle_idx + 1}:")
            print(f"  Logit similarity: {logit_similarity:.4f}")
            print(f"  Prob similarity: {prob_similarity:.4f}")
            print(f"  Prediction agreement: {prediction_agreement:.4f}")
        
        # Compute summary statistics
        avg_logit_similarity = np.mean([r['logit_similarity'] for r in shuffled_results])
        avg_prob_similarity = np.mean([r['prob_similarity'] for r in shuffled_results])
        avg_prediction_agreement = np.mean([r['prediction_agreement'] for r in shuffled_results])
        
        results = {
            'original_predictions': original_predictions.flatten().tolist(),
            'shuffled_results': shuffled_results,
            'summary': {
                'avg_logit_similarity': avg_logit_similarity,
                'avg_prob_similarity': avg_prob_similarity,
                'avg_prediction_agreement': avg_prediction_agreement,
                'interpretation': self._interpret_similarity(avg_logit_similarity, avg_prediction_agreement)
            }
        }
        
        return results
    
    def test_single_frame_vs_sequence(self, frames: List[torch.Tensor]) -> Dict:
        """
        Test if single frame classification is as good as sequence classification.
        This directly tests if temporal information is needed.
        """
        print("Testing single frame vs sequence classification...")
        
        # Test sequence classification
        with torch.no_grad():
            sequence_output = self.model([{"image": frames}])
            sequence_logits = sequence_output["pred_logits"]
            sequence_probs = F.softmax(sequence_logits, dim=-1)
            sequence_predictions = torch.argmax(sequence_logits, dim=-1)
        
        # Test single frame classification (use middle frame)
        middle_frame_idx = len(frames) // 2
        single_frame = [frames[middle_frame_idx]]
        
        with torch.no_grad():
            single_output = self.model([{"image": single_frame}])
            single_logits = single_output["pred_logits"]
            single_probs = F.softmax(single_logits, dim=-1)
            single_predictions = torch.argmax(single_logits, dim=-1)
        
        # Compare results
        logit_similarity = F.cosine_similarity(
            sequence_logits.flatten(), 
            single_logits.flatten(), 
            dim=0
        ).item()
        
        prob_similarity = F.cosine_similarity(
            sequence_probs.flatten(), 
            single_probs.flatten(), 
            dim=0
        ).item()
        
        results = {
            'sequence_predictions': sequence_predictions.flatten().tolist(),
            'single_frame_predictions': single_predictions.flatten().tolist(),
            'logit_similarity': logit_similarity,
            'prob_similarity': prob_similarity,
            'interpretation': self._interpret_single_vs_sequence(logit_similarity)
        }
        
        print(f"Sequence vs single frame logit similarity: {logit_similarity:.4f}")
        print(f"Sequence vs single frame prob similarity: {prob_similarity:.4f}")
        
        return results
    
    def test_temporal_consistency(self, frames: List[torch.Tensor]) -> Dict:
        """
        Test if classification is consistent across frames in the sequence.
        If temporal information is important, we'd expect some variation.
        """
        print("Testing temporal consistency...")
        
        with torch.no_grad():
            output = self.model([{"image": frames}])
            logits = output["pred_logits"]  # (b, t, q, c)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)  # (b, t, q)
        
        # Analyze consistency across time
        temporal_consistency = []
        for t in range(logits.shape[1] - 1):  # Compare consecutive frames
            frame_t_logits = logits[:, t, :, :]
            frame_t1_logits = logits[:, t + 1, :, :]
            
            similarity = F.cosine_similarity(
                frame_t_logits.flatten(), 
                frame_t1_logits.flatten(), 
                dim=0
            ).item()
            temporal_consistency.append(similarity)
        
        avg_consistency = np.mean(temporal_consistency)
        consistency_std = np.std(temporal_consistency)
        
        # Check if predictions change over time
        prediction_changes = []
        for t in range(predictions.shape[1] - 1):
            frame_t_preds = predictions[:, t, :]
            frame_t1_preds = predictions[:, t + 1, :]
            change_rate = (frame_t_preds != frame_t1_preds).float().mean().item()
            prediction_changes.append(change_rate)
        
        avg_prediction_change = np.mean(prediction_changes)
        
        results = {
            'temporal_consistency': temporal_consistency,
            'avg_consistency': avg_consistency,
            'consistency_std': consistency_std,
            'prediction_changes': prediction_changes,
            'avg_prediction_change': avg_prediction_change,
            'interpretation': self._interpret_consistency(avg_consistency, avg_prediction_change)
        }
        
        print(f"Average temporal consistency: {avg_consistency:.4f}")
        print(f"Average prediction change rate: {avg_prediction_change:.4f}")
        
        return results
    
    def _interpret_similarity(self, logit_similarity: float, prediction_agreement: float) -> str:
        """Interpret similarity results."""
        if logit_similarity > 0.95 and prediction_agreement > 0.9:
            return "STRONG APPEARANCE-BASED: Frame order has minimal impact on classification. Model relies primarily on static visual features."
        elif logit_similarity < 0.7 or prediction_agreement < 0.5:
            return "STRONG TEMPORAL-BASED: Frame order significantly affects classification. Model relies heavily on temporal information."
        else:
            return "MIXED APPROACH: Moderate impact of frame order. Model uses both appearance and temporal features."
    
    def _interpret_single_vs_sequence(self, similarity: float) -> str:
        """Interpret single frame vs sequence results."""
        if similarity > 0.9:
            return "SINGLE FRAME SUFFICIENT: Single frame classification is nearly as good as sequence classification. Temporal information is not essential."
        elif similarity < 0.6:
            return "SEQUENCE NECESSARY: Sequence classification is significantly better than single frame. Temporal information is important."
        else:
            return "MODERATE BENEFIT: Sequence provides some benefit over single frame, but not dramatic."
    
    def _interpret_consistency(self, consistency: float, change_rate: float) -> str:
        """Interpret temporal consistency results."""
        if consistency > 0.95 and change_rate < 0.1:
            return "HIGH CONSISTENCY: Classification is very stable across frames. Suggests appearance-based classification."
        elif consistency < 0.7 or change_rate > 0.3:
            return "LOW CONSISTENCY: Classification varies significantly across frames. Suggests temporal information is important."
        else:
            return "MODERATE CONSISTENCY: Some variation in classification across frames."
    
    def run_comprehensive_test(self, frames: List[torch.Tensor]) -> Dict:
        """
        Run all motion tests and provide comprehensive analysis.
        """
        print("="*60)
        print("COMPREHENSIVE MOTION ANALYSIS")
        print("="*60)
        
        results = {}
        
        # Test 1: Frame order importance
        print("\n1. Testing frame order importance...")
        results['frame_order'] = self.test_frame_order_importance(frames)
        
        # Test 2: Single frame vs sequence
        print("\n2. Testing single frame vs sequence...")
        results['single_vs_sequence'] = self.test_single_frame_vs_sequence(frames)
        
        # Test 3: Temporal consistency
        print("\n3. Testing temporal consistency...")
        results['temporal_consistency'] = self.test_temporal_consistency(frames)
        
        # Provide overall interpretation
        print("\n" + "="*60)
        print("OVERALL INTERPRETATION")
        print("="*60)
        
        overall_interpretation = self._provide_overall_interpretation(results)
        results['overall_interpretation'] = overall_interpretation
        
        print(overall_interpretation)
        
        return results
    
    def _provide_overall_interpretation(self, results: Dict) -> str:
        """Provide overall interpretation based on all tests."""
        interpretations = []
        
        # Frame order test
        frame_order_summary = results['frame_order']['summary']
        interpretations.append(f"Frame Order Test: {frame_order_summary['interpretation']}")
        
        # Single vs sequence test
        single_vs_seq = results['single_vs_sequence']
        interpretations.append(f"Single vs Sequence: {single_vs_seq['interpretation']}")
        
        # Temporal consistency test
        consistency = results['temporal_consistency']
        interpretations.append(f"Temporal Consistency: {consistency['interpretation']}")
        
        # Overall conclusion
        logit_similarity = frame_order_summary['avg_logit_similarity']
        single_similarity = single_vs_seq['logit_similarity']
        consistency_score = consistency['avg_consistency']
        
        if logit_similarity > 0.9 and single_similarity > 0.8 and consistency_score > 0.9:
            conclusion = "CONCLUSION: The model appears to use APPEARANCE-BASED classification. Temporal information is not essential for species identification."
        elif logit_similarity < 0.7 or single_similarity < 0.6 or consistency_score < 0.7:
            conclusion = "CONCLUSION: The model appears to use TEMPORAL-BASED classification. Motion patterns are important for species identification."
        else:
            conclusion = "CONCLUSION: The model uses a MIXED approach, combining both appearance and temporal features for classification."
        
        interpretations.append(conclusion)
        
        return "\n\n".join(interpretations)
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file."""
        # Convert tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_tensors(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def load_frames_from_video(video_path: str, num_frames: int = 5) -> List[torch.Tensor]:
    """
    Load frames from a video file. This is a placeholder - you'll need to implement
    actual video loading based on your data format.
    """
    # Placeholder - replace with actual video loading
    import cv2
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and normalize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
            frames.append(frame_tensor)
        frame_count += 1
    
    cap.release()
    return frames


def main():
    """
    Example usage of the SimpleMotionTest.
    """
    # Configuration
    config_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/config.yaml"
    model_path = "/home/simone/store/simone/dvis-model-outputs/trained_models/model3_unmasked/model_final.pth"
    
    # Initialize tester
    tester = SimpleMotionTest(config_path, model_path)
    
    # Load frames (replace with your actual frame loading)
    # frames = load_frames_from_video("path/to/your/video.mp4", num_frames=5)
    
    # For demonstration, create dummy frames
    print("Creating dummy frames for demonstration...")
    frames = [torch.randn(3, 480, 640) for _ in range(5)]
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(frames)
    
    # Save results
    output_path = "motion_analysis_results.json"
    tester.save_results(results, output_path)
    
    print(f"\nAnalysis complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
