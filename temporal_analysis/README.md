# DVIS-DAQ Temporal Motion Analysis

This repository implements gradient-based temporal analysis for the DVIS-DAQ (Decoupled Video Instance Segmentation with Dynamic Adaptive Querying) model to investigate motion awareness in fish species classification.

## Overview

This analysis determines whether your DVIS-DAQ model uses temporal motion information for species classification by:

1. **Extracting temporal gradients** from key model components
2. **Correlating motion patterns** with gradient importance
3. **Analyzing temporal perturbations** to measure motion dependency
4. **Identifying species-specific temporal signatures**

## Scientific Background

### Research Question
Does the DVIS-DAQ model use temporal motion patterns to distinguish between fish species, or does it rely primarily on static visual features?

### Key Methods
- **Gradient-based Temporal Analysis**: Extends Grad-CAM to temporal dimensions
- **Motion-Gradient Correlation**: Correlates optical flow with temporal gradients
- **Temporal Perturbation Studies**: Measures performance under temporal modifications
- **Species-Specific Analysis**: Identifies unique temporal signatures per species

### Expected Insights
- **High motion-gradient correlation**: Model uses motion for classification
- **Species-specific temporal patterns**: Different species have distinct motion signatures
- **Temporal perturbation sensitivity**: Model relies on temporal order

## Architecture Analysis

### DVIS-DAQ Temporal Components

#### 1. **VideoInstanceCutter (Track Module)**
- **Location**: `track_module.py`
- **Key Layers**:
  - `transformer_cross_attention_layers`: Cross-frame attention
  - `transformer_self_attention_layers`: Temporal consistency
  - `slot_cross_attention_layers`: Background modeling
- **Temporal Processing**: Processes frame embeddings (B, C, T, Q)

#### 2. **TemporalRefiner**
- **Location**: `refiner.py`
- **Key Layers**:
  - `transformer_time_self_attention_layers`: Long-term temporal attention
  - `conv_short_aggregate_layers`: Short-term temporal convolution
- **Temporal Processing**: Refines instance embeddings across time

#### 3. **Video Mask2Former Transformer Decoder**
- **Location**: `video_mask2former_transformer_decoder.py`
- **Key Components**:
  - `SelfAttentionLayer`: Temporal self-attention
  - `CrossAttentionLayer`: Cross-frame attention
  - `PositionEmbeddingSine3D`: 3D positional encoding

### Gradient Extraction Strategy

#### Primary Targets
1. **Frame Embeddings**: `frame_embeds` (B, C, T, Q) - Direct temporal features
2. **Track Queries**: `track_queries` - Temporal object representations
3. **Temporal Attention**: `transformer_time_self_attention_layers` - Attention weights

#### Gradient Flow Paths
```
Input Video → Backbone → Frame Embeddings → Track Module → Classification
                                    ↓
                              Temporal Refiner → Classification
```

## Implementation Details

### Core Components

#### 1. **TemporalGradientExtractor**
- Hooks into temporal layers to capture gradients
- Computes gradients with respect to temporal input
- Aggregates gradients across spatial dimensions

#### 2. **MotionCorrelator**
- Computes optical flow between consecutive frames
- Correlates motion magnitude with gradient importance
- Analyzes species-specific motion patterns

#### 3. **TemporalPerturbationAnalyzer**
- Applies temporal perturbations (shuffle, drop, reverse)
- Measures gradient changes under perturbations
- Quantifies temporal dependency

#### 4. **SpeciesTemporalAnalyzer**
- Extracts species-specific temporal signatures
- Compares temporal patterns across species
- Identifies motion-based classification indicators

### Key Metrics

#### 1. **Motion-Gradient Correlation**
- **Formula**: `corr(motion_magnitude, temporal_gradient_importance)`
- **Interpretation**: 
  - High correlation (>0.7): Motion-dependent classification
  - Low correlation (<0.3): Static feature classification

#### 2. **Temporal Perturbation Degradation**
- **Formula**: `(original_performance - perturbed_performance) / original_performance`
- **Interpretation**:
  - High degradation (>20%): Temporal order dependency
  - Low degradation (<10%): Static feature reliance

#### 3. **Species Temporal Signature Distance**
- **Formula**: `||species_A_temporal_pattern - species_B_temporal_pattern||`
- **Interpretation**:
  - Large distance: Distinct motion patterns
  - Small distance: Similar motion patterns

## Usage

### Basic Analysis
```python
from temporal_analyzer import DVIS_TemporalAnalyzer

# Initialize analyzer
analyzer = DVIS_TemporalAnalyzer(
    model_path="/path/to/model_final.pth",
    config_path="/path/to/config.yaml",
    data_path="/path/to/video/data"
)

# Run comprehensive analysis
results = analyzer.analyze_temporal_motion_awareness()
```

### Species-Specific Analysis
```python
# Analyze specific species
species_results = analyzer.analyze_species_temporal_patterns(
    species_list=["Atlantic Salmon", "Chinook", "Brown Trout"]
)
```

### Temporal Perturbation Study
```python
# Test temporal perturbations
perturbation_results = analyzer.analyze_temporal_perturbations(
    perturbation_types=["shuffle", "drop", "reverse"]
)
```

## Output Interpretation

### Motion-Aware Classification Indicators

#### Strong Motion Dependency
- Motion-gradient correlation > 0.7
- Temporal perturbation degradation > 20%
- Species-specific temporal signatures
- High gradient importance during motion events

#### Static Feature Classification
- Motion-gradient correlation < 0.3
- Temporal perturbation degradation < 10%
- Similar temporal patterns across species
- High gradient importance on static frames

### Species-Specific Insights

#### Distinct Motion Patterns
- Different species show unique temporal gradient patterns
- Motion events correlate with species-specific behaviors
- Temporal perturbations affect species differently

#### Behavioral Correlations
- Swimming speed patterns in temporal gradients
- Direction change detection in attention weights
- Species-specific motion signatures

## Technical Implementation

### Memory Management
- **GPU Memory**: Optimized for 24GB GPU
- **Batch Processing**: Configurable batch sizes
- **Gradient Checkpointing**: For long video sequences

### Performance Optimization
- **Frame Sampling**: Configurable temporal sampling
- **Caching**: Intermediate result caching
- **Parallel Processing**: Multi-video analysis

### Extensibility
- **Modular Design**: Easy to add new analysis methods
- **Configurable Hooks**: Flexible gradient extraction points
- **Custom Metrics**: Extensible metric computation

## Scientific Applications

### 1. **Fish Behavior Research**
- Understanding species-specific swimming patterns
- Identifying behavioral differences between species
- Correlating motion patterns with ecological factors

### 2. **Model Interpretability**
- Understanding what features the model learns
- Validating model behavior against biological knowledge
- Improving model transparency

### 3. **Conservation Biology**
- Identifying behavioral indicators for species health
- Understanding habitat-specific behaviors
- Supporting population monitoring efforts

### 4. **Computer Vision**
- Improving video classification models
- Understanding temporal feature learning
- Developing motion-aware architectures

## References

1. **Grad-CAM**: Selvaraju, R. R., et al. (2017). Grad-cam: Visual explanations from deep networks.
2. **Optical Flow**: Farnebäck, G. (2003). Two-frame motion estimation based on polynomial expansion.
3. **Temporal Perturbations**: Ribeiro, M. T., et al. (2016). "Why should I trust you?" Explaining predictions.
4. **Fish Behavior**: Webb, P. W. (1984). Body form, locomotion and foraging in aquatic vertebrates.

## File Structure

```
dvis_temporal_analysis/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── config.py                          # Configuration settings
├── temporal_analyzer.py               # Main analysis class
├── gradient_extractor.py              # Gradient computation
├── motion_correlator.py               # Motion-gradient correlation
├── perturbation_analyzer.py           # Temporal perturbation analysis
├── species_analyzer.py                # Species-specific analysis
├── utils/
│   ├── model_loader.py               # DVIS-DAQ model loading
│   ├── data_loader.py                # Video data loading
│   ├── metrics.py                    # Analysis metrics
│   └── visualization.py              # Plotting utilities
├── examples/
│   ├── basic_analysis.py             # Basic usage example
│   ├── species_comparison.py         # Species-specific analysis
│   └── perturbation_study.py         # Temporal perturbation analysis
└── results/                          # Output directory
    ├── temporal_heatmaps/
    ├── correlation_plots/
    └── analysis_reports/
```

## Contributing

To extend this analysis:

1. **Add New Metrics**: Implement in `metrics.py`
2. **New Analysis Methods**: Add to main analyzer class
3. **Additional Visualizations**: Extend `visualization.py`
4. **Model Support**: Adapt `model_loader.py` for new architectures

## License

This project is licensed under the MIT License.
