# Test script to verify data loading works correctly
import pandas as pd
import os

print("Testing data loading...")

# Test different possible paths
possible_paths = [
    '../upload/gym_members_exercise_tracking_synthetic_data.csv',
    '../datasets/gym_members_exercise_tracking_synthetic_data.csv',
    'gym_members_exercise_tracking_synthetic_data.csv',
    '../upload/fitness_dataset.csv',
    '../datasets/fitness_dataset.csv'
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Found dataset at: {path}")
        try:
            df = pd.read_csv(path)
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            break
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")
    else:
        print(f"❌ Not found: {path}")

# List current directory contents
print("\nCurrent directory contents:")
print(os.listdir('.'))

# List parent directory contents  
print("\nParent directory contents:")
if os.path.exists('../upload'):
    print("../upload/:", os.listdir('../upload'))
if os.path.exists('../datasets'):
    print("../datasets/:", os.listdir('../datasets'))