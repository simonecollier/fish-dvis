# Memory-Efficient Temporal Gradient Analysis

This directory contains memory-efficient implementations for analyzing temporal gradients in DVIS-DAQ models using 31-frame windows, even with limited GPU memory.

## Problem Statement

Your DVIS-DAQ model was trained on 31-frame windows (`SAMPLING_FRAME_NUM: 31`), but analyzing temporal gradients on full 31-frame sequences can exhaust GPU memory. This implementation provides several strategies to handle this constraint while maintaining accurate temporal analysis.

## Key Features

### 1. **Chunked Processing**
- Process 31-frame videos in smaller chunks (2-8 frames at a time)
- Maintains temporal context while reducing memory usage
- Configurable chunk sizes based on available GPU memory

### 2. **Gradient Checkpointing**
- Uses PyTorch's gradient checkpointing to trade computation for memory
- Reduces memory usage by ~50% with minimal accuracy loss
- Automatically enabled for memory-constrained setups

### 3. **Intermediate Result Caching**
- Saves gradient computations to disk between chunks
- Enables resuming analysis if interrupted
- Reduces redundant computations

### 4. **Memory Monitoring**
- Real-time GPU and system memory monitoring
- Automatic memory cleanup when usage exceeds thresholds
- Configurable memory limits and cleanup frequency

### 5. **Adaptive Configuration**
- Auto-detects available GPU memory
- Adjusts settings based on memory constraints
- Supports CPU offloading for very limited memory

## Memory Requirements

| GPU Memory | Chunk Size | Max Frames/Video | Max Videos/Species | Performance |
|------------|------------|------------------|-------------------|-------------|
| 24GB+      | 8 frames   | 31 frames        | 10 videos         | Full speed  |
| 16GB       | 6 frames   | 25 frames        | 8 videos          | Good speed  |
| 8GB        | 4 frames   | 20 frames        | 5 videos          | Moderate    |
| <8GB       | 2 frames   | 15 frames        | 3 videos          | CPU offload |

## Quick Start

### 1. Basic Usage (Auto-detect GPU memory)

```bash
cd fish-dvis/temporal_analysis
python run_memory_efficient_analysis.py
```

### 2. Specify GPU Memory

```bash
python run_memory_efficient_analysis.py --gpu-memory 16
```

### 3. Custom Configuration

```bash
python run_memory_efficient_analysis.py \
    --gpu-memory 8 \
    --max-videos 3 \
    --max-frames 20 \
    --chunk-size 4 \
    --output-dir my_analysis_results
```

### 4. Force CPU Processing (for very limited memory)

```bash
python run_memory_efficient_analysis.py --force-cpu
```

## Advanced Usage

### Custom Configuration

```python
from memory_efficient_config import create_memory_efficient_config
from memory_efficient_temporal_analyzer import MemoryEfficientTemporalAnalyzer

# Create configuration for 16GB GPU
config = create_memory_efficient_config(gpu_memory_gb=16.0)

# Customize settings
config.max_videos_per_species = 5
config.chunk_size = 6
config.save_intermediate_results = True

# Run analysis
analyzer = MemoryEfficientTemporalAnalyzer(config)
results = analyzer.run_memory_efficient_analysis()
```

### Memory Monitoring

```python
# Monitor memory usage during analysis
analyzer.monitor_memory_usage()

# Force memory cleanup
analyzer.force_memory_cleanup()

# Check memory constraints
usage_estimate = config.get_memory_usage_estimate()
print(f"Estimated memory utilization: {usage_estimate['memory_utilization']:.1%}")
```

## Implementation Details

### Memory-Efficient Gradient Extraction

The `MemoryEfficientGradientExtractor` class implements:

1. **Chunked Processing**: Breaks 31-frame sequences into smaller chunks
2. **Gradient Checkpointing**: Uses `torch.utils.checkpoint.checkpoint` for memory efficiency
3. **Intermediate Caching**: Saves results to temporary files
4. **Memory Cleanup**: Aggressive garbage collection between chunks

### Key Methods

```python
# Extract gradients for a chunk of frames
chunk_gradients = extractor.extract_chunk_gradients(chunk_frames, start_idx, total_frames)

# Process full video in chunks
complete_gradients = extractor.process_video_chunks(video_frames, video_id)

# Memory-efficient extraction
temporal_gradients = extractor.extract_temporal_gradients_memory_efficient(video_frames, video_id)
```

### Configuration Optimization

The `MemoryEfficientConfig` class automatically optimizes:

- **Chunk size** based on available GPU memory
- **Maximum frames per video** to fit in memory
- **Gradient checkpointing** settings
- **Memory cleanup frequency**
- **CPU offloading** for very limited memory

## Output Structure

```
temporal_analysis_results/
├── reports/
│   └── temporal_analysis_results.json    # Final aggregated results
├── cache/
│   ├── video_0000_results.pkl           # Intermediate results
│   ├── video_0001_results.pkl
│   └── ...
├── temporal_heatmaps/                    # Temporal gradient visualizations
├── correlation_plots/                    # Motion correlation plots
└── species_analysis/                     # Species-specific analysis
```

## Results Format

The analysis produces:

1. **Temporal Gradients**: Gradient magnitudes for each frame
2. **Motion Magnitudes**: Optical flow-based motion measurements
3. **Temporal Importance**: Aggregated importance scores
4. **Motion Correlation**: Correlation between gradients and motion
5. **Species Statistics**: Aggregated results by fish species

## Troubleshooting

### Out of Memory Errors

1. **Reduce chunk size**:
   ```bash
   python run_memory_efficient_analysis.py --chunk-size 2
   ```

2. **Force CPU processing**:
   ```bash
   python run_memory_efficient_analysis.py --force-cpu
   ```

3. **Reduce frames per video**:
   ```bash
   python run_memory_efficient_analysis.py --max-frames 15
   ```

### Slow Performance

1. **Increase chunk size** if memory allows
2. **Use GPU** instead of CPU
3. **Reduce number of videos** per species

### Interrupted Analysis

The analysis automatically saves intermediate results. To resume:

```python
# Results are cached in temporary directory
# Analysis will automatically skip completed videos
python run_memory_efficient_analysis.py
```

## Comparison with 5-Frame Windows

| Aspect | 31-Frame (Memory-Efficient) | 5-Frame Windows |
|--------|----------------------------|-----------------|
| **Temporal Context** | Full behavioral patterns | Limited context |
| **Species Discrimination** | High accuracy | Reduced accuracy |
| **Memory Usage** | Optimized for constraints | Low memory usage |
| **Processing Speed** | Slower due to chunking | Faster |
| **Model Compatibility** | Matches training | Mismatch with training |

## Recommendations

1. **Use 31-frame analysis** when possible for accurate species discrimination
2. **Start with auto-detection** and adjust based on performance
3. **Monitor memory usage** during analysis
4. **Save intermediate results** for long-running analyses
5. **Consider hybrid approach** for real-time applications

## Performance Tips

1. **Close other GPU applications** during analysis
2. **Use SSD storage** for intermediate result caching
3. **Monitor system memory** in addition to GPU memory
4. **Adjust chunk size** based on actual memory usage
5. **Use multiple GPUs** if available (future enhancement)

## Future Enhancements

- Multi-GPU support for parallel processing
- Distributed processing across multiple machines
- Real-time analysis with streaming video
- Advanced memory prediction and optimization
- Integration with cloud computing platforms
