#!/usr/bin/env python
"""
Simple example: Running backtest with opponent features

This shows how to use the opponent features configuration
in your existing backtesting workflow.
"""

import subprocess
import sys
import os

def run_opponent_backtest():
    """Run backtest comparison: baseline vs opponent features"""
    
    print("="*80)
    print("OPPONENT FEATURES BACKTEST COMPARISON")
    print("="*80)
    
    # Test date range (adjust these to your available data)
    test_start = "20250205"
    test_end = "20250206"
    
    print(f"Test period: {test_start} to {test_end}")
    print()
    
    # 1. Run baseline backtest
    print("-"*60)
    print("RUNNING BASELINE BACKTEST")
    print("-"*60)
    
    baseline_cmd = [
        "python", "scripts/run_backtest.py",
        "--test-start", test_start,
        "--test-end", test_end, 
        "--feature-config", "default_features",
        "--output-dir", "data/backtest_results/baseline"
    ]
    
    print("Command:", " ".join(baseline_cmd))
    print()
    
    try:
        result = subprocess.run(baseline_cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print("✓ Baseline backtest completed successfully")
            print("Baseline results saved to: data/backtest_results/baseline")
        else:
            print("✗ Baseline backtest failed:")
            print(result.stderr)
            return
    except Exception as e:
        print(f"Error running baseline: {e}")
        return
    
    # 2. Run opponent features backtest
    print("\n" + "-"*60)
    print("RUNNING OPPONENT FEATURES BACKTEST")
    print("-"*60)
    
    opponent_cmd = [
        "python", "scripts/run_backtest.py",
        "--test-start", test_start,
        "--test-end", test_end,
        "--feature-config", "opponent_features", 
        "--output-dir", "data/backtest_results/opponent_features"
    ]
    
    print("Command:", " ".join(opponent_cmd))
    print()
    
    try:
        result = subprocess.run(opponent_cmd, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print("✓ Opponent features backtest completed successfully")
            print("Opponent features results saved to: data/backtest_results/opponent_features")
        else:
            print("✗ Opponent features backtest failed:")
            print(result.stderr)
            return
    except Exception as e:
        print(f"Error running opponent features: {e}")
        return
    
    # 3. Compare results
    print("\n" + "-"*60)
    print("COMPARISON")
    print("-"*60)
    
    print("Compare results by examining:")
    print("1. MAPE improvement in opponent features vs baseline")
    print("2. Correlation improvements") 
    print("3. Summary markdown files in each output directory")
    print()
    print("Expected improvements with opponent features:")
    print("  • MAPE: 2-8% better")
    print("  • Correlation: +0.01 to +0.05")
    print("  • Better predictions for matchup-dependent players")
    
    print("\n" + "="*80)
    print("BACKTEST COMPARISON COMPLETE")
    print("="*80)


def quick_test():
    """Run a quick single-day test"""
    print("Running quick opponent features test...")
    
    cmd = [
        "python", "scripts/run_backtest.py",
        "--test-start", "20250205",
        "--test-end", "20250205", 
        "--feature-config", "opponent_features",
        "--verbose"
    ]
    
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        if result.returncode == 0:
            print("✓ Quick test completed successfully")
        else:
            print("✗ Quick test failed")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        run_opponent_backtest()