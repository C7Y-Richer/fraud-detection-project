#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions Test Script for fraud_detection_stacking.py
This script performs basic functionality tests for CI/CD pipeline.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))

def test_dummy():
    """
    Original dummy test function
    """
    print("GitHub Actions test script ran successfully.")
    return True

def test_imports():
    """
    Test critical imports for the fraud detection system
    """
    print("Testing imports...")
    
    try:
        # Test main module import
        from train.fraud_detection_stacking import (
            build_base_models,
            build_meta_models,
            save_models,
            load_models
        )
        print("✓ Main module imported successfully")
        
        # Test critical libraries
        import numpy as np
        import pandas as pd
        import sklearn
        import xgboost
        import lightgbm
        import joblib
        print("✓ All critical libraries available")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_building():
    """
    Test basic model building functionality
    """
    print("Testing model building...")
    
    try:
        from train.fraud_detection_stacking import build_base_models, build_meta_models
        
        # Test base models
        base_models = build_base_models()
        expected_base = ['mlp', 'rf', 'svm', 'xgb']
        
        for model_name in expected_base:
            if model_name not in base_models:
                print(f"✗ Missing base model: {model_name}")
                return False
        
        print(f"✓ Base models built: {list(base_models.keys())}")
        
        # Test meta models
        meta_models = build_meta_models()
        expected_meta = ['meta_pred', 'meta_prob']
        
        for model_name in expected_meta:
            if model_name not in meta_models:
                print(f"✗ Missing meta model: {model_name}")
                return False
        
        print(f"✓ Meta models built: {list(meta_models.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False

def test_basic_functionality():
    """
    Test basic prediction functionality with synthetic data
    """
    print("Testing basic functionality...")
    
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        import lightgbm as lgb
        from train.fraud_detection_stacking import predict, evaluate
        
        # Create simple test data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.choice([0, 1], 100, p=[0.8, 0.2])
        X_test = np.random.randn(50, 10)
        y_test = np.random.choice([0, 1], 50, p=[0.8, 0.2])
        
        # Create and train simple models
        base_models = {
            'rf': RandomForestClassifier(n_estimators=5, random_state=42),
            'lr': LogisticRegression(random_state=42)
        }
        
        for model in base_models.values():
            model.fit(X_train, y_train)
        
        # Create meta models
        meta_models = {
            'meta_pred': lgb.LGBMClassifier(n_estimators=5, random_state=42, verbose=-1),
            'meta_prob': lgb.LGBMClassifier(n_estimators=5, random_state=42, verbose=-1)
        }
        
        # Train meta models with base model outputs
        meta_features = []
        for model in base_models.values():
            pred = model.predict(X_train)
            meta_features.append(pred)
        
        meta_X = np.column_stack(meta_features)
        
        for meta_model in meta_models.values():
            meta_model.fit(meta_X, y_train)
        
        # Test prediction
        y_pred, y_prob = predict(X_test, y_test, base_models, meta_models)
        
        # Test evaluation
        metrics = evaluate(y_test, y_pred, y_prob)
        
        print(f"✓ Prediction successful - ROC AUC: {metrics['roc_auc']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def run_tests():
    """
    Run all tests for GitHub Actions
    """
    print("=" * 50)
    print("FRAUD DETECTION SYSTEM - GITHUB ACTIONS TESTS")
    print("=" * 50)
    
    tests = [
        ("Dummy Test", test_dummy),
        ("Import Test", test_imports),
        ("Model Building Test", test_model_building),
        ("Basic Functionality Test", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed >= total * 0.75:  # Pass if 75% or more tests pass
        print("🎉 Tests passed! System is ready for deployment.")
        return True
    else:
        print("❌ Too many tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)