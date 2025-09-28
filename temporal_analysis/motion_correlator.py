#!/usr/bin/env python3
"""
Motion correlation analysis for DVIS-DAQ temporal analysis
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
import scipy.stats

class MotionCorrelator:
    """Correlate motion patterns with temporal gradients"""
    
    def __init__(self, config):
        """
        Initialize motion correlator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.optical_flow_method = config.optical_flow_method
        self.cache_optical_flow = config.cache_optical_flow
        self.flow_cache = {}
        
    def compute_optical_flow(self, video_frames: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between consecutive frames
        
        Args:
            video_frames: Video frames array (T, H, W, C) or (T, C, H, W)
            
        Returns:
            Motion magnitude array (T-1,)
        """
        # Ensure frames are in (T, H, W, C) format
        if video_frames.shape[1] == 3:  # (T, C, H, W)
            video_frames = np.transpose(video_frames, (0, 2, 3, 1))
        
        T, H, W, C = video_frames.shape
        
        # Convert to grayscale for optical flow
        gray_frames = []
        for frame in video_frames:
            if C == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame.squeeze()
            gray_frames.append(gray)
        
        gray_frames = np.array(gray_frames)
        
        # Compute optical flow
        motion_magnitudes = []
        
        for i in range(T - 1):
            prev_frame = gray_frames[i]
            curr_frame = gray_frames[i + 1]
            
            # Compute optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, 
                flags=0
            )
            
            # Compute motion magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_magnitude = np.mean(magnitude)
            motion_magnitudes.append(mean_magnitude)
        
        return np.array(motion_magnitudes)
    
    def compute_motion_statistics(self, motion_magnitudes: np.ndarray) -> Dict[str, float]:
        """
        Compute motion statistics
        
        Args:
            motion_magnitudes: Motion magnitude array
            
        Returns:
            Dictionary of motion statistics
        """
        stats = {
            "mean_motion": float(np.mean(motion_magnitudes)),
            "std_motion": float(np.std(motion_magnitudes)),
            "max_motion": float(np.max(motion_magnitudes)),
            "min_motion": float(np.min(motion_magnitudes)),
            "motion_range": float(np.max(motion_magnitudes) - np.min(motion_magnitudes)),
            "motion_variance": float(np.var(motion_magnitudes)),
            "motion_skewness": float(self.compute_skewness(motion_magnitudes)),
            "motion_kurtosis": float(self.compute_kurtosis(motion_magnitudes))
        }
        
        return stats
    
    def compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        skewness = np.mean(((data - mean) / std) ** 3)
        return float(skewness)
    
    def compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return float(kurtosis)
    
    def compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two arrays"""
        try:
            # Discretize for mutual information computation
            x_bins = np.histogram(x, bins=10)[0]
            y_bins = np.histogram(y, bins=10)[0]
            
            # Normalize bins
            x_bins = x_bins / np.sum(x_bins)
            y_bins = y_bins / np.sum(y_bins)
            
            # Compute mutual information
            return float(mutual_info_score(x_bins, y_bins))
        except:
            return 0.0
    
    def compute_temporal_consistency(self, gradient_magnitudes: np.ndarray) -> float:
        """Compute temporal consistency of gradient magnitudes"""
        try:
            if len(gradient_magnitudes) < 2:
                return 0.0
            
            # Compute autocorrelation
            autocorr = np.corrcoef(gradient_magnitudes[:-1], gradient_magnitudes[1:])[0, 1]
            return float(autocorr) if not np.isnan(autocorr) else 0.0
        except:
            return 0.0
    
    def correlate_motion_gradients(self, 
                                 motion_magnitudes: np.ndarray, 
                                 temporal_gradients: torch.Tensor) -> Dict[str, float]:
        """
        Correlate motion magnitudes with temporal gradients
        
        Args:
            motion_magnitudes: Motion magnitude array (T-1,)
            temporal_gradients: Temporal gradients tensor (T, C, H, W)
            
        Returns:
            Dictionary containing correlation metrics
        """
        # Ensure temporal gradients are on CPU for numpy operations
        if isinstance(temporal_gradients, torch.Tensor):
            temporal_gradients = temporal_gradients.detach().cpu()
        
        # Compute gradient magnitudes across spatial dimensions
        gradient_magnitudes = torch.norm(temporal_gradients, dim=(1, 2, 3)).numpy()
        
        # Ensure we have the same number of frames
        T = len(gradient_magnitudes)
        if len(motion_magnitudes) != T - 1:
            # Pad or truncate motion magnitudes to match
            if len(motion_magnitudes) > T - 1:
                motion_magnitudes = motion_magnitudes[:T-1]
            else:
                # Pad with zeros
                padding = np.zeros(T - 1 - len(motion_magnitudes))
                motion_magnitudes = np.concatenate([motion_magnitudes, padding])
        
        # Compute correlation coefficients
        pearson_corr = np.corrcoef(motion_magnitudes, gradient_magnitudes[1:])[0, 1]
        spearman_corr = scipy.stats.spearmanr(motion_magnitudes, gradient_magnitudes[1:])[0]
        
        # Compute mutual information
        try:
            mi_score = self.compute_mutual_information(motion_magnitudes, gradient_magnitudes[1:])
        except:
            mi_score = 0.0
        
        # Compute temporal consistency
        temporal_consistency = self.compute_temporal_consistency(gradient_magnitudes)
        
        return {
            "pearson_correlation": pearson_corr if not np.isnan(pearson_corr) else 0.0,
            "spearman_correlation": spearman_corr if not np.isnan(spearman_corr) else 0.0,
            "mutual_information": mi_score,
            "temporal_consistency": temporal_consistency,
            "motion_magnitudes": motion_magnitudes,
            "gradient_magnitudes": gradient_magnitudes
        }
    
    def analyze_motion_gradient_temporal_patterns(self,
                                                motion_magnitudes: np.ndarray,
                                                temporal_gradients: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze temporal patterns in motion-gradient relationship
        
        Args:
            motion_magnitudes: Motion magnitude array
            temporal_gradients: Temporal gradient importance
            
        Returns:
            Dictionary of temporal pattern analysis
        """
        # Ensure same length
        min_length = min(len(motion_magnitudes), len(temporal_gradients))
        motion_magnitudes = motion_magnitudes[:min_length]
        temporal_gradients = temporal_gradients[:min_length].cpu().numpy()
        
        # Find motion peaks
        motion_peaks = self.find_peaks(motion_magnitudes, threshold=0.5)
        gradient_peaks = self.find_peaks(temporal_gradients, threshold=0.5)
        
        # Analyze peak alignment
        peak_alignment = self.analyze_peak_alignment(motion_peaks, gradient_peaks, min_length)
        
        # Analyze temporal lag
        temporal_lag = self.analyze_temporal_lag(motion_magnitudes, temporal_gradients)
        
        # Analyze motion-gradient phase relationship
        phase_analysis = self.analyze_phase_relationship(motion_magnitudes, temporal_gradients)
        
        return {
            "motion_peaks": motion_peaks,
            "gradient_peaks": gradient_peaks,
            "peak_alignment": peak_alignment,
            "temporal_lag": temporal_lag,
            "phase_analysis": phase_analysis
        }
    
    def find_peaks(self, data: np.ndarray, threshold: float = 0.5) -> List[int]:
        """Find peaks in data above threshold"""
        peaks = []
        max_val = np.max(data)
        threshold_val = threshold * max_val
        
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and 
                data[i] > data[i+1] and 
                data[i] > threshold_val):
                peaks.append(i)
        
        return peaks
    
    def analyze_peak_alignment(self, 
                             motion_peaks: List[int], 
                             gradient_peaks: List[int],
                             total_length: int) -> Dict[str, float]:
        """Analyze alignment between motion and gradient peaks"""
        if not motion_peaks or not gradient_peaks:
            return {
                "peak_overlap_ratio": 0.0,
                "average_peak_distance": float('inf'),
                "peak_correlation": 0.0
            }
        
        # Find overlapping peaks (within 2 frames)
        overlapping_peaks = 0
        peak_distances = []
        
        for m_peak in motion_peaks:
            for g_peak in gradient_peaks:
                distance = abs(m_peak - g_peak)
                if distance <= 2:  # Within 2 frames
                    overlapping_peaks += 1
                    peak_distances.append(distance)
        
        # Compute metrics
        overlap_ratio = overlapping_peaks / min(len(motion_peaks), len(gradient_peaks))
        avg_distance = np.mean(peak_distances) if peak_distances else float('inf')
        
        # Create binary peak arrays for correlation
        motion_peak_array = np.zeros(total_length)
        gradient_peak_array = np.zeros(total_length)
        
        for peak in motion_peaks:
            if peak < total_length:
                motion_peak_array[peak] = 1
        
        for peak in gradient_peaks:
            if peak < total_length:
                gradient_peak_array[peak] = 1
        
        peak_correlation = np.corrcoef(motion_peak_array, gradient_peak_array)[0, 1]
        if np.isnan(peak_correlation):
            peak_correlation = 0.0
        
        return {
            "peak_overlap_ratio": float(overlap_ratio),
            "average_peak_distance": float(avg_distance),
            "peak_correlation": float(peak_correlation)
        }
    
    def analyze_temporal_lag(self, 
                           motion_magnitudes: np.ndarray,
                           temporal_gradients: np.ndarray,
                           max_lag: int = 5) -> Dict[str, float]:
        """Analyze temporal lag between motion and gradients"""
        max_correlation = 0.0
        optimal_lag = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Motion leads gradients
                motion_shifted = motion_magnitudes[-lag:]
                gradients_shifted = temporal_gradients[:len(motion_shifted)]
            else:
                # Gradients lead motion
                motion_shifted = motion_magnitudes[:-lag] if lag > 0 else motion_magnitudes
                gradients_shifted = temporal_gradients[lag:len(motion_shifted) + lag]
            
            if len(motion_shifted) > 1 and len(gradients_shifted) > 1:
                correlation = np.corrcoef(motion_shifted, gradients_shifted)[0, 1]
                if not np.isnan(correlation) and abs(correlation) > abs(max_correlation):
                    max_correlation = correlation
                    optimal_lag = lag
        
        return {
            "optimal_lag": int(optimal_lag),
            "max_lag_correlation": float(max_correlation)
        }
    
    def analyze_phase_relationship(self,
                                 motion_magnitudes: np.ndarray,
                                 temporal_gradients: np.ndarray) -> Dict[str, float]:
        """Analyze phase relationship between motion and gradients"""
        # Compute phase using FFT
        motion_fft = np.fft.fft(motion_magnitudes)
        gradient_fft = np.fft.fft(temporal_gradients)
        
        # Compute phase difference
        phase_diff = np.angle(motion_fft) - np.angle(gradient_fft)
        
        # Compute phase coherence
        phase_coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        # Compute phase correlation
        phase_correlation = np.corrcoef(np.angle(motion_fft), np.angle(gradient_fft))[0, 1]
        if np.isnan(phase_correlation):
            phase_correlation = 0.0
        
        return {
            "phase_coherence": float(phase_coherence),
            "phase_correlation": float(phase_correlation),
            "mean_phase_difference": float(np.mean(phase_diff))
        }
    
    def classify_motion_dependency(self, 
                                 correlation_results: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify motion dependency based on correlation results
        
        Args:
            correlation_results: Results from correlate_motion_gradients
            
        Returns:
            Classification results
        """
        pearson_corr = correlation_results["pearson_correlation"]
        spearman_corr = correlation_results["spearman_correlation"]
        mutual_info = correlation_results["mutual_information"]
        
        # Classification thresholds
        high_corr_threshold = self.config.motion_correlation_threshold
        low_corr_threshold = self.config.motion_low_correlation_threshold
        
        # Determine motion dependency level
        if pearson_corr > high_corr_threshold and spearman_corr > high_corr_threshold:
            dependency_level = "high"
            classification = "motion_dependent"
        elif pearson_corr > low_corr_threshold or spearman_corr > low_corr_threshold:
            dependency_level = "moderate"
            classification = "partially_motion_dependent"
        else:
            dependency_level = "low"
            classification = "static_feature_dependent"
        
        # Confidence score
        confidence = (abs(pearson_corr) + abs(spearman_corr) + mutual_info) / 3
        
        return {
            "dependency_level": dependency_level,
            "classification": classification,
            "confidence": float(confidence),
            "pearson_significance": pearson_corr > high_corr_threshold,
            "spearman_significance": spearman_corr > high_corr_threshold,
            "mutual_info_significance": mutual_info > 0.1  # Threshold for mutual info
        }
    
    def save_motion_analysis(self, 
                           analysis_results: Dict[str, Any], 
                           output_path: Path):
        """Save motion analysis results"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save correlation results
        if "correlation_results" in analysis_results:
            corr_path = output_path / "motion_gradient_correlation.json"
            with open(corr_path, 'w') as f:
                json.dump(analysis_results["correlation_results"], f, indent=2)
        
        # Save motion statistics
        if "motion_statistics" in analysis_results:
            stats_path = output_path / "motion_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(analysis_results["motion_statistics"], f, indent=2)
        
        # Save temporal patterns
        if "temporal_patterns" in analysis_results:
            pattern_path = output_path / "temporal_patterns.json"
            with open(pattern_path, 'w') as f:
                json.dump(analysis_results["temporal_patterns"], f, indent=2)
        
        # Save classification results
        if "classification" in analysis_results:
            class_path = output_path / "motion_dependency_classification.json"
            with open(class_path, 'w') as f:
                json.dump(analysis_results["classification"], f, indent=2)
        
        print(f"Motion analysis saved to {output_path}")
