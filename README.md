# Fraud Detection Project

An explainable fraud detection framework based on the IEEE-CIS Fraud Detection dataset. This project integrates advanced feature engineering techniques with a multi-layer stacking ensemble model to enhance detection performance in imbalanced and dynamic e-commerce environments.

## Project Overview

This project aims to detect fraudulent online transactions using a combination of data preprocessing, ensemble modeling, and model interpretation techniques. It is built on a real-world dataset and addresses key challenges such as class imbalance and feature complexity.

**Key Features:**

- Preprocessing for high-cardinality, missing values, and data imbalance
- Extensive feature engineering (aggregation, time-related features, categorical encoding)
- Stacking ensemble model with multiple base learners
- Explainability using SHAP
- Evaluation using AUC, F1-score, precision, and recall

## Dataset

The [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) is used in this project. It includes over 590,000 online transactions with transaction and identity features. 

Due to file size limitations, raw CSV files and model artifacts are **not included** in the GitHub repository.

# Project Directory Structure

The following directory structure describes the organization of the project files:
```
.
├── README.md                         # Project introduction and usage instructions
├── requirements.txt                  # Python dependencies for the project
├── test.py                           # Script for testing functionality
├── test_script.py                    # Additional test script

├── base_models_contrast/             # Base model training and logging
│   ├── base_log651427.log            # Training log for logistic regression
│   ├── base_mlp651420.log            # MLP training log (version 1)
│   ├── base_mlp651421.log            # MLP training log (version 2)
│   ├── base_svm651424.log            # SVM training log
│   ├── base_xgb651426.log            # XGBoost training log
│   ├── run_log.py                    # Logistic regression training script
│   ├── run_log.sbatch                # SLURM batch script for logistic regression
│   ├── run_mlp.py                    # MLP training script
│   ├── run_mlp.sbatch                # SLURM batch script for MLP
│   ├── run_rf.py                     # Random Forest training script
│   ├── run_rf.sbatch                 # SLURM batch script for Random Forest
│   ├── run_svm.py                    # SVM training script
│   ├── run_svm.sbatch                # SLURM batch script for SVM
│   ├── run_xgb.py                    # XGBoost training script
│   └── run_xgb.sbatch                # SLURM batch script for XGBoost

├── data/                             # Dataset files
│   ├── sample_submission.csv         # Example of submission format for competition
│   ├── test_identity.csv             # Identity data for test set
│   └── train_identity.csv            # Identity data for training set

├── fraud_api/                        # Web API for model deployment
│   ├── app.py                        # Flask application entry point
│   ├── models/                       # Saved models for inference
│   ├── templates/                    # HTML templates for frontend
│   └── validate.py                   # Data validation logic for API input

├── train/                            # Model training and result storage
│   ├── catboost_info/                # CatBoost internal files (e.g., logs, caches)
│   ├── fraud_detection_results.csv   # Final fraud prediction results
│   ├── fraud_detection_results.png   # Visualization of results
│   ├── fraud_detection_stacking.py   # Stacking ensemble implementation
│   ├── kaggle_submission.sbatch      # SLURM script for submitting to Kaggle
│   ├── meta_model_data/              # Intermediate data for meta model
│   ├── models/                       # Trained models (base and meta learners)
│   ├── res.csv                       # Miscellaneous result file
│   ├── run_fraud_detection.sbatch    # SLURM job script for end-to-end run
│   ├── shap_analysis/                # SHAP value analysis for model interpretation
│   ├── shap_analysis_test/           # Additional SHAP analysis on test data
│   └── validate.py                   # Validation logic for training data

```
