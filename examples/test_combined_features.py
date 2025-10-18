#!/usr/bin/env python
"""
Test script for combined feature configurations.

This demonstrates the new ability to combine multiple feature configurations
using comma-separated lists.

Usage:
    python examples/test_combined_features.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.feature_config import load_feature_config
from src.features.pipeline import FeaturePipeline
import pandas as pd


def test_single_config():
    """Test loading a single feature configuration (existing behavior)."""
    print("="*80)
    print("TESTING SINGLE FEATURE CONFIG")
    print("="*80)
    
    try:
        config = load_feature_config('base_features')
        print(f"✓ Loaded single config: {config.name}")
        print(f"  Stats: {len(config.stats)}")
        print(f"  Transformers: {len(config.transformers)}")
        
        # Test pipeline building
        pipeline = config.build_pipeline(FeaturePipeline)
        print(f"✓ Built pipeline with {len(pipeline.transformers)} transformers")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load single config: {e}")
        return False


def test_combined_config():
    """Test loading combined feature configurations."""
    print("\n" + "="*80)
    print("TESTING COMBINED FEATURE CONFIGS")
    print("="*80)
    
    # Test combinations
    test_cases = [
        ("base_features,opponent_features", "Base + Opponent features"),
        ("default_features,opponent_features", "Default + Opponent features"),
        ("base_features,default_features", "Base + Default features"),
    ]
    
    results = []
    
    for config_string, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"Config string: '{config_string}'")
        
        try:
            config = load_feature_config(config_string)
            pipeline = config.build_pipeline(FeaturePipeline)
            
            print(f"✓ Successfully combined configurations")
            print(f"  Combined name: {config.name}")
            print(f"  Total stats: {len(config.stats)}")
            print(f"  Total transformers: {len(config.transformers)}")
            print(f"  Pipeline transformers: {len(pipeline.transformers)}")
            
            # Show transformer types
            transformer_types = [t.name for t in pipeline.transformers]
            print(f"  Transformer types: {transformer_types}")
            
            results.append((config_string, True, None))
            
        except Exception as e:
            print(f"✗ Failed to combine configs: {e}")
            results.append((config_string, False, str(e)))
    
    return results


def test_predefined_combined():
    """Test the predefined combined config file."""
    print("\n" + "="*80)
    print("TESTING PREDEFINED COMBINED CONFIG")
    print("="*80)
    
    try:
        config = load_feature_config('base_with_opponent')
        pipeline = config.build_pipeline(FeaturePipeline)
        
        print(f"✓ Loaded predefined combined config: {config.name}")
        print(f"  Description: {config.description}")
        print(f"  Stats: {len(config.stats)}")
        print(f"  Categorical features: {len(config.categorical_features)}")
        print(f"  Transformers: {len(config.transformers)}")
        print(f"  Pipeline transformers: {len(pipeline.transformers)}")
        
        # Show transformer details
        print(f"\n  Transformer details:")
        for i, transformer in enumerate(pipeline.transformers):
            print(f"    {i+1}. {transformer.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load predefined config: {e}")
        return False


def demonstrate_usage_examples():
    """Show practical usage examples."""
    print("\n" + "="*80)
    print("PRACTICAL USAGE EXAMPLES")
    print("="*80)
    
    examples = [
        {
            "name": "Minimal with Opponent Context", 
            "config": "base_features,opponent_features",
            "use_case": "Fast training with essential opponent matchup features"
        },
        {
            "name": "Full Features with Opponents",
            "config": "default_features,opponent_features", 
            "use_case": "Complete feature set including all stats + opponent context"
        },
        {
            "name": "Custom Combination",
            "config": "base_features,base_with_opponent",
            "use_case": "Base features + predefined opponent subset"
        }
    ]
    
    print("Available combinations for --feature-config:")
    print()
    
    for example in examples:
        print(f"📋 {example['name']}")
        print(f"   Config: --feature-config {example['config']}")
        print(f"   Use case: {example['use_case']}")
        print()
    
    print("Command examples:")
    print("  # Single config (existing behavior)")
    print("  python scripts/run_backtest.py --feature-config default_features --test-start 20250205 --test-end 20250206")
    print()
    print("  # Combined configs (new functionality)")
    print("  python scripts/run_backtest.py --feature-config base_features,opponent_features --test-start 20250205 --test-end 20250206")
    print()
    print("  # Multiple combinations")
    print("  python scripts/run_backtest.py --feature-config default_features,opponent_features --test-start 20250205 --test-end 20250206")


def main():
    """Run all tests and demonstrations."""
    print("FEATURE CONFIGURATION COMBINATION TESTS")
    print("=" * 80)
    
    # Test single config (backward compatibility)
    single_success = test_single_config()
    
    # Test combined configs
    combined_results = test_combined_config()
    
    # Test predefined combined config
    predefined_success = test_predefined_combined()
    
    # Show usage examples
    demonstrate_usage_examples()
    
    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    print(f"Single config test: {'✓ PASS' if single_success else '✗ FAIL'}")
    print(f"Predefined combined config: {'✓ PASS' if predefined_success else '✗ FAIL'}")
    
    print(f"\nCombined config tests:")
    for config_string, success, error in combined_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {config_string}: {status}")
        if error:
            print(f"    Error: {error}")
    
    total_tests = 1 + 1 + len(combined_results)
    passed_tests = int(single_success) + int(predefined_success) + sum(1 for _, success, _ in combined_results if success)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Combined feature configurations are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()