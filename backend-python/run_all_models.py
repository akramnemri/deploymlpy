#!/usr/bin/env python3
"""
Run all ML training scripts in sequence
This script will execute all the training scripts and generate comparison results
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Running from directory: {script_dir}")
        
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Error running {script_name} (exit code: {result.returncode})")
            return False
        else:
            print(f"✅ Successfully ran {script_name}")
            return True
            
    except Exception as e:
        print(f"❌ Exception running {script_name}: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 Starting ML Model Training Pipeline")
    print("="*60)
    
    # List of scripts to run
    scripts = [
        'train_calories_model_gradient_boosting_fixed.py',
        'train_calories_model_svm_fixed.py', 
        'train_weight_model_gradient_boosting_fixed.py',
        'train_weight_model_svm_fixed.py',
        'model_comparison_fixed.py'
    ]
    
    success_count = 0
    total_count = len(scripts)
    
    for script in scripts:
        if run_script(script):
            success_count += 1
        print(f"\nProgress: {success_count}/{total_count} scripts completed successfully")
    
    print(f"\n{'='*60}")
    print("🏁 Training Pipeline Complete!")
    print(f"✅ Successful: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All models trained successfully!")
        print("\nGenerated files:")
        print("- calories_model_gradient_boosting.pkl")
        print("- calories_model_svm.pkl") 
        print("- weight_model_gradient_boosting.pkl")
        print("- weight_model_svm.pkl")
        print("- model_comparison_results.pkl")
        print("- model_comparison.png")
    else:
        print("⚠️  Some scripts failed. Check the output above for details.")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)