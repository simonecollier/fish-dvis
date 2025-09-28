#!/usr/bin/env python3
"""
Basic example of using the DVIS-DAQ temporal motion analyzer
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from temporal_analyzer import DVIS_TemporalAnalyzer
from config import TemporalAnalysisConfig

def main():
    """Run basic temporal motion analysis"""
    
    print("DVIS-DAQ Temporal Motion Analysis")
    print("=" * 40)
    
    # Initialize configuration
    config = TemporalAnalysisConfig()
    
    # Customize configuration if needed
    config.max_videos_per_species = 10  # Limit for testing
    config.use_val_set = True  # Use validation set
    
    # Initialize analyzer
    print("Initializing temporal analyzer...")
    analyzer = DVIS_TemporalAnalyzer(config)
    
    try:
        # Run comprehensive analysis
        print("Starting temporal motion awareness analysis...")
        results = analyzer.analyze_temporal_motion_awareness()
        
        # Print summary
        summary = results["summary_statistics"]
        print(f"\nAnalysis Summary:")
        print(f"Total species analyzed: {summary['total_species_analyzed']}")
        print(f"Total videos analyzed: {summary['total_videos_analyzed']}")
        print(f"Overall motion dependency: {summary['overall_motion_dependency']['classification']}")
        print(f"Motion dependent ratio: {summary['overall_motion_dependency']['motion_dependent_ratio']:.2f}")
        
        # Print species-specific results
        print(f"\nSpecies-Specific Results:")
        for species, result in results["species_results"].items():
            classification = result["average_classification"]["most_common_classification"]
            confidence = result["average_classification"]["average_confidence"]
            pearson_corr = result["average_correlation"].get("pearson_correlation", 0)
            
            print(f"{species}:")
            print(f"  Classification: {classification}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Pearson correlation: {pearson_corr:.3f}")
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {analyzer.output_paths['base']}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
    
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    main()
