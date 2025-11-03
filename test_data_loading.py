#!/usr/bin/env python3
"""
Quick test script to verify data loading and feature extraction.
"""

import sys
import os

# Import functions from main script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing CNN_target_locator.py improvements...")
print("=" * 60)

# Test imports
print("\n[1] Testing imports...")
try:
    from CNN_target_locator import (
        parse_tem_file,
        create_feature_profile,
        load_all_data,
        build_improved_model
    )
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test file parsing
print("\n[2] Testing file parsing...")
try:
    test_file = "/home/user/movingloopML/1700/0moffset1.tem"
    df, metadata = parse_tem_file(test_file)

    if df is not None and metadata is not None:
        print(f"  ✓ Successfully parsed {test_file}")
        print(f"    - Shape: {df.shape}")
        print(f"    - Metadata: {metadata}")
    else:
        print(f"  ✗ Failed to parse {test_file}")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Parse error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test feature extraction
print("\n[3] Testing feature extraction...")
try:
    profile = create_feature_profile(df, metadata)

    if profile is not None:
        print(f"  ✓ Successfully created feature profile")
        print(f"    - Shape: {profile.shape}")
        print(f"    - Non-zero stations: {(profile.sum(axis=1) != 0).sum()}")
    else:
        print(f"  ✗ Failed to create feature profile")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Feature extraction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model building
print("\n[4] Testing model architecture...")
try:
    import tensorflow as tf
    input_shape = (profile.shape[0], profile.shape[1])
    model = build_improved_model(input_shape)

    print(f"  ✓ Successfully built model")
    print(f"    - Input shape: {input_shape}")
    print(f"    - Parameters: {model.count_params():,}")
    print(f"    - Layers: {len(model.layers)}")
except Exception as e:
    print(f"  ✗ Model building error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nThe script is ready to run.")
print("To train the model, run: python3 CNN_target_locator.py")
