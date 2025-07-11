# -*- coding: utf-8 -*-
"""
Fraud Detection System based on Stacking Model
Using IEEE CIS Fraud Detection Dataset

Base Models: MLP, Random Forest, SVM, XGBoost
Meta Models: LightGBM (two)
 - Meta Model 1: Uses predictions from base models as input
 - Meta Model 2: Uses prediction probabilities from base models as input
Final Result: Soft voting ensemble of prediction probabilities from both meta models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# Data loading and preprocessing function
def load_and_preprocess_data(train_path, train_id_path, test_path=None, sample_size=None, scaler=None):
    """
    Load and preprocess IEEE-CIS Fraud Detection data.
    
    Key steps:
    1. Drop high-missing and low-variance columns
    2. Group and reduce 'V' features via PCA
    3. Create rolling window and ratio features
    4. Frequency encoding & optional target encoding
    5. Save selected features and scaler
    """
    # -------------------- Load & Merge --------------------
    train1 = pd.read_csv(train_path)
    train2 = pd.read_csv(train_id_path)
    train_df = pd.merge(train1, train2, on="TransactionID", how="left").drop_duplicates("TransactionID")
    if sample_size:
        train_df = train_df.sample(sample_size, random_state=42)
    y_train = train_df['isFraud'].values
    X_train = train_df.drop(columns=['isFraud'])

    # -------------------- 1. Handle missing values first --------------------
    # Handle missing values first to avoid errors in subsequent calculations
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col].fillna("missing", inplace=True)
        else:
            X_train[col].fillna(-1, inplace=True)
            
    # -------------------- 2. Remove high-missing & low-variance --------------------
    missing_thresh = 0.9
    var_thresh = 1e-3
    missing_cols = X_train.columns[X_train.isnull().mean() > missing_thresh]
    # Calculate standard deviation only for numerical columns
    low_var_cols = X_train.select_dtypes(include=[np.number]).columns[X_train.select_dtypes(include=[np.number]).std() < var_thresh]
    X_train.drop(columns=missing_cols.union(low_var_cols), inplace=True)

    # -------------------- 2. Process V features with PCA --------------------
    v_cols = [col for col in X_train.columns if col.startswith('V')]
    if v_cols:
        v_filled = X_train[v_cols].fillna(-1)
        pca = PCA(n_components=5)
        v_pca = pca.fit_transform(v_filled)
        for i in range(5):
            X_train[f'V_PCA_{i+1}'] = v_pca[:, i]
        X_train.drop(columns=v_cols, inplace=True)

    # -------------------- 3. Construct rolling window & ratio features --------------------
    if 'TransactionDT' in X_train.columns:
        X_train['hour'] = (X_train['TransactionDT'] / 3600) % 24
        X_train['day'] = (X_train['TransactionDT'] / (3600 * 24)) % 7
        X_train['is_night'] = ((X_train['hour'] >= 22) | (X_train['hour'] <= 6)).astype(int)
        X_train['is_rush_hour'] = ((X_train['hour'].between(7, 9)) | (X_train['hour'].between(17, 19))).astype(int)

    if 'TransactionAmt' in X_train.columns:
        X_train['TransactionAmt_log'] = np.log1p(X_train['TransactionAmt'])
        X_train['TransactionAmt_decimal'] = X_train['TransactionAmt'] % 1
        X_train['TransactionAmt_round'] = (X_train['TransactionAmt_decimal'] == 0).astype(int)
        
        # Add transaction amount ratio features
        X_train['TransactionAmt_to_mean_card1'] = X_train['TransactionAmt'] / X_train.groupby(['card1'])['TransactionAmt'].transform('mean')
        X_train['TransactionAmt_to_mean_card4'] = X_train['TransactionAmt'] / X_train.groupby(['card4'])['TransactionAmt'].transform('mean')
        X_train['TransactionAmt_to_std_card1'] = X_train['TransactionAmt'] / X_train.groupby(['card1'])['TransactionAmt'].transform('std')
        X_train['TransactionAmt_to_std_card4'] = X_train['TransactionAmt'] / X_train.groupby(['card4'])['TransactionAmt'].transform('std')

    if 'id_02' in X_train.columns:
        X_train['id_02_to_mean_card1'] = X_train['id_02'] / X_train.groupby(['card1'])['id_02'].transform('mean')
        X_train['id_02_to_mean_card4'] = X_train['id_02'] / X_train.groupby(['card4'])['id_02'].transform('mean')
        X_train['id_02_to_std_card1'] = X_train['id_02'] / X_train.groupby(['card1'])['id_02'].transform('std')
        X_train['id_02_to_std_card4'] = X_train['id_02'] / X_train.groupby(['card4'])['id_02'].transform('std')
        
        # Handle infinity and NaN values
        for col in ['id_02_to_mean_card1', 'id_02_to_mean_card4', 'id_02_to_std_card1', 'id_02_to_std_card4']:
            X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan).fillna(-1)

    if 'D15' in X_train.columns:
        X_train['D15_to_mean_card1'] = X_train['D15'] / X_train.groupby(['card1'])['D15'].transform('mean')
        X_train['D15_to_mean_card4'] = X_train['D15'] / X_train.groupby(['card4'])['D15'].transform('mean')
        X_train['D15_to_std_card1'] = X_train['D15'] / X_train.groupby(['card1'])['D15'].transform('std')
        X_train['D15_to_std_card4'] = X_train['D15'] / X_train.groupby(['card4'])['D15'].transform('std')
        X_train['D15_to_mean_addr1'] = X_train['D15'] / X_train.groupby(['addr1'])['D15'].transform('mean')
        X_train['D15_to_mean_addr2'] = X_train['D15'] / X_train.groupby(['addr2'])['D15'].transform('mean')
        X_train['D15_to_std_addr1'] = X_train['D15'] / X_train.groupby(['addr1'])['D15'].transform('std')
        X_train['D15_to_std_addr2'] = X_train['D15'] / X_train.groupby(['addr2'])['D15'].transform('std')
        
        # Handle infinity and NaN values
        D15_ratio_cols = [
            'D15_to_mean_card1', 'D15_to_mean_card4', 'D15_to_std_card1', 'D15_to_std_card4',
            'D15_to_mean_addr1', 'D15_to_mean_addr2', 'D15_to_std_addr1', 'D15_to_std_addr2'
        ]
        for col in D15_ratio_cols:
            X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan).fillna(-1)

    # Process email domains
    if 'P_emaildomain' in X_train.columns:
        X_train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = X_train['P_emaildomain'].str.split('.', expand=True)
        # Handle NaN values
        for col in ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']:
            X_train[col] = X_train[col].fillna('missing')
    if 'R_emaildomain' in X_train.columns:
        X_train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = X_train['R_emaildomain'].str.split('.', expand=True)
        # Handle NaN values
        for col in ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']:
            X_train[col] = X_train[col].fillna('missing')
            
    # Label Encoding for all object type columns
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = LabelEncoder().fit_transform(X_train[col].astype(str))

    if 'card1' in X_train.columns:
        card1_amt_mean = X_train.groupby('card1')['TransactionAmt'].transform('mean')
        card1_amt_std = X_train.groupby('card1')['TransactionAmt'].transform('std').replace(0, 1)
        X_train['card1_amt_z'] = (X_train['TransactionAmt'] - card1_amt_mean) / card1_amt_std

    # -------------------- 4. Frequency encoding --------------------
    cat_cols = [col for col in X_train.columns if X_train[col].nunique() < 50]
    for col in cat_cols:
        freq = X_train[col].value_counts()
        X_train[col + "_freq"] = X_train[col].map(freq)

    # Missing values and encoding already handled above, no repetition here

    # -------------------- 5. Handle infinity and extreme values --------------------
    # Handle infinity and extreme values before standardization
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        # Replace infinity values
        X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN values with median
        X_train[col] = X_train[col].fillna(X_train[col].median() if not X_train[col].isna().all() else -1)
        # Handle extreme values (optional)
        # q1, q3 = X_train[col].quantile(0.01), X_train[col].quantile(0.99)
        # X_train[col] = X_train[col].clip(q1, q3)
    
    # -------------------- 6. Normalize --------------------
    if scaler is None:
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    else:
        X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])

    # -------------------- Remove high correlation --------------------
    corr = pd.DataFrame(X_train, columns=numerical_cols).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    selected_features = [col for col in X_train.columns if col not in to_drop]

    # -------------------- Save --------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(selected_features, "./models/selectedfeatures.pkl")
    joblib.dump(scaler, "./models/scaler.pkl")
    X_train = X_train[selected_features]
    
    # Initialize test data variables
    X_test = None
    y_test = None

    # -------------------- Optional: process test --------------------
    if test_path:
        test_df = pd.read_csv(test_path)
        if 'isFraud' in test_df.columns:
            y_test = test_df['isFraud'].values
            X_test = test_df.drop(columns=['isFraud'])
        else:
            y_test = None
            X_test = test_df

        # Align structure with training
        for col in X_test.columns:
            if col in missing_cols.union(low_var_cols):
                X_test.drop(columns=col, inplace=True, errors='ignore')
            elif X_test[col].dtype == 'object':
                X_test[col].fillna("missing", inplace=True)
            else:
                X_test[col].fillna(-1, inplace=True)

        if v_cols:
            v_filled_test = X_test[v_cols].fillna(-1)
            v_pca_test = pca.transform(v_filled_test)
            for i in range(5):
                X_test[f'V_PCA_{i+1}'] = v_pca_test[:, i]
            X_test.drop(columns=v_cols, inplace=True)

        for col in cat_cols:
            freq = X_test[col].value_counts()
            X_test[col + "_freq"] = X_test[col].map(freq)

        # Process time features
        if 'TransactionDT' in X_test.columns:
            X_test['hour'] = (X_test['TransactionDT'] / 3600) % 24
            X_test['day'] = (X_test['TransactionDT'] / (3600 * 24)) % 7
            X_test['is_night'] = ((X_test['hour'] >= 22) | (X_test['hour'] <= 6)).astype(int)
            X_test['is_rush_hour'] = (((X_test['hour'] >= 8) & (X_test['hour'] <= 10)) | ((X_test['hour'] >= 17) & (X_test['hour'] <= 19))).astype(int)

        # Process transaction amount features
        if 'TransactionAmt' in X_test.columns:
            X_test['TransactionAmt_log'] = np.log1p(X_test['TransactionAmt'])
            X_test['TransactionAmt_decimal'] = X_test['TransactionAmt'] % 1
            X_test['TransactionAmt_round'] = (X_test['TransactionAmt_decimal'] == 0).astype(int)
            
            # Add transaction amount ratio features
            if 'card1' in X_test.columns:
                card1_amt_mean = X_test.groupby('card1')['TransactionAmt'].transform('mean')
                card1_amt_std = X_test.groupby('card1')['TransactionAmt'].transform('std').replace(0, 1)
                X_test['card1_amt_z'] = (X_test['TransactionAmt'] - card1_amt_mean) / card1_amt_std
        
        # Add id_02 ratio features
        if 'id_02' in X_test.columns:
            if 'card1' in X_test.columns:
                X_test['id_02_to_mean_card1'] = X_test['id_02'] / X_test.groupby(['card1'])['id_02'].transform('mean')
                X_test['id_02_to_std_card1'] = X_test['id_02'] / X_test.groupby(['card1'])['id_02'].transform('std')
            if 'card4' in X_test.columns:
                X_test['id_02_to_mean_card4'] = X_test['id_02'] / X_test.groupby(['card4'])['id_02'].transform('mean')
                X_test['id_02_to_std_card4'] = X_test['id_02'] / X_test.groupby(['card4'])['id_02'].transform('std')
            
            # Handle infinity and NaN values
            for col in ['id_02_to_mean_card1', 'id_02_to_mean_card4', 'id_02_to_std_card1', 'id_02_to_std_card4']:
                if col in X_test.columns:
                    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan).fillna(-1)
        
        # Add D15 ratio features
        if 'D15' in X_test.columns:
            if 'card1' in X_test.columns:
                X_test['D15_to_mean_card1'] = X_test['D15'] / X_test.groupby(['card1'])['D15'].transform('mean')
                X_test['D15_to_std_card1'] = X_test['D15'] / X_test.groupby(['card1'])['D15'].transform('std')
            if 'card4' in X_test.columns:
                X_test['D15_to_mean_card4'] = X_test['D15'] / X_test.groupby(['card4'])['D15'].transform('mean')
                X_test['D15_to_std_card4'] = X_test['D15'] / X_test.groupby(['card4'])['D15'].transform('std')
            if 'addr1' in X_test.columns:
                X_test['D15_to_mean_addr1'] = X_test['D15'] / X_test.groupby(['addr1'])['D15'].transform('mean')
                X_test['D15_to_std_addr1'] = X_test['D15'] / X_test.groupby(['addr1'])['D15'].transform('std')
            if 'addr2' in X_test.columns:
                X_test['D15_to_mean_addr2'] = X_test['D15'] / X_test.groupby(['addr2'])['D15'].transform('mean')
                X_test['D15_to_std_addr2'] = X_test['D15'] / X_test.groupby(['addr2'])['D15'].transform('std')
            
            # Handle infinity and NaN values
            D15_ratio_cols = [
                'D15_to_mean_card1', 'D15_to_mean_card4', 'D15_to_std_card1', 'D15_to_std_card4',
                'D15_to_mean_addr1', 'D15_to_mean_addr2', 'D15_to_std_addr1', 'D15_to_std_addr2'
            ]
            for col in D15_ratio_cols:
                if col in X_test.columns:
                    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan).fillna(-1)
        
        # Process email domains
        if 'P_emaildomain' in X_test.columns:
            X_test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = X_test['P_emaildomain'].str.split('.', expand=True)
            # Handle NaN values
            for col in ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']:
                X_test[col] = X_test[col].fillna('missing')
        if 'R_emaildomain' in X_test.columns:
            X_test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = X_test['R_emaildomain'].str.split('.', expand=True)
            # Handle NaN values
            for col in ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']:
                X_test[col] = X_test[col].fillna('missing')
                
        # Label Encoding for all object type columns
        for col in X_test.columns:
            if X_test[col].dtype == 'object':
                X_test[col] = LabelEncoder().fit_transform(X_test[col].astype(str))
                
        # Handle infinity and extreme values
        numerical_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            # Replace infinity values
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
            # Fill NaN values with median
            X_test[col] = X_test[col].fillna(X_test[col].median() if not X_test[col].isna().all() else -1)
            
        # Standardize
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        # Use the same feature selection
        X_test = X_test[selected_features]
        
    return X_train, y_train, X_test, y_test, scaler



# Handle imbalanced data
def handle_imbalance(X, y, smotes=False):
    """
    Handle imbalanced data using SMOTE
    
    Parameters:
        X: Feature matrix
        y: Label vector
        
    Returns:
        X_resampled, y_resampled: Resampled data
    """
    print("Handling imbalanced data using SMOTE...")

    # Ensure X is in numpy array format
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    # Ensure y is in numpy array format
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_array, y_array)

    print(f"Fraud sample ratio in original data: {sum(y_array) / len(y_array):.4f}")
    print(f"Fraud sample ratio after resampling: {sum(y_resampled) / len(y_resampled):.4f}")

    if smotes:
        return X_resampled, y_resampled
    else:
        return X, y


# Build base models
def build_base_models():
    """
    Build base models
    
    Returns:
        models: Dictionary of base models
    """
    print("Building base models...")
    
    
    models = {
        # 'mlp': MLPClassifier(hidden_layer_sizes=(512, 256, 128),
        #                      activation='relu',  
        #                      solver='adam',  
        #                      alpha=1e-5,  # L2 regularization term to prevent overfitting
        #                      batch_size=4096,  
        #                      learning_rate='adaptive', 
        #                      learning_rate_init=1e-3, 
        #                      max_iter=100, 
        #                      shuffle=True,  
        #                      random_state=42,
        #                      # early_stopping=True,  
        #                      # validation_fraction=0.1, 
        #                      verbose=True),
        'rf': RandomForestClassifier(n_estimators=1000,
                                     min_samples_split=10,
                                     min_samples_leaf=4,
                                     max_features='sqrt',  
                                     class_weight='balanced_subsample',  # Handle imbalanced samples
                                     n_jobs=-1,
                                     random_state=42),
        'lgb': lgb.LGBMClassifier(n_estimators=2000,
                                  learning_rate=0.01,
                                  num_leaves=64,
                                  max_depth=-1,
                                  min_child_samples=100,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  reg_alpha=1.0,
                                  reg_lambda=1.0,
                                  scale_pos_weight=20,  
                                  importance_type='gain',
                                  objective='binary',
                                  boosting_type='gbdt',
                                  n_jobs=-1,
                                  random_state=42,
                                  verbose=-1),
        # 'svm': SVC(kernel='rbf', C=10, gamma=0.5, tol=1e-5, probability=True, random_state=42),
        
        'xgb': xgb.XGBClassifier(n_estimators=3000,                 
                                learning_rate=0.0139,              
                                max_depth=16,                      
                                min_child_weight=170,              
                                gamma=0.6665,                      
                                subsample=0.7,                     
                                colsample_bytree=0.7463,           
                                reg_alpha=0.3987,                  
                                reg_lambda=0.2431,                 
                                scale_pos_weight=20,  
                                tree_method='hist',            
                                eval_metric='auc',                 
                                use_label_encoder=False,           
                                n_jobs=-1,
                                random_state=42),
        'catboost': CatBoostClassifier(iterations=2000,
                                learning_rate=0.03,
                                depth=7,
                                l2_leaf_reg=3,
                                bagging_temperature=1,
                                random_strength=1,
                                border_count=254,
                                scale_pos_weight=20,
                                loss_function='Logloss',
                                eval_metric='AUC',
                                early_stopping_rounds=200,
                                task_type='CPU',       
                                verbose=500,
                                random_state=42)

    }
    return models


# Build meta models
def build_meta_models():
    """
    Build meta models
    
    Returns:
        meta_models: Dictionary of meta models
    """
    print("Building meta models...")
    meta_models = {
        'meta_pred': lgb.LGBMClassifier(n_estimators=100, random_state=42),
        'meta_prob': lgb.LGBMClassifier(n_estimators=100, random_state=42),
    }
    return meta_models


# Train and evaluate models
def train_and_evaluate(X, y, n_splits=6, meta_data_dir="meta_model_data"):
    """
    Train and evaluate models using K-fold cross-validation
    
    Parameters:
        X: Feature matrix
        y: Label vector
        n_splits: Number of cross-validation folds
        meta_data_dir: Directory to save meta-model training data
        
    Returns:
        base_models: Trained base models
        meta_models: Trained meta models
        cv_scores: Cross-validation scores
    """
    print(f"Training and evaluating models using {n_splits}-fold cross-validation...")

    # Initialize K-fold cross-validation
    # StratifiedGroupKFold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize models
    base_models = build_base_models()
    meta_models = build_meta_models()

    # Store AUC scores for each fold
    base_auc_scores = {model_name: [] for model_name in base_models}
    meta_auc_scores = {model_name: [] for model_name in meta_models}
    ensemble_auc_scores = []

    # Create lists to store prediction results for all resampled validation sets
    all_meta_train_pred = []
    all_meta_train_prob = []
    all_meta_y = []

    # K-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        # Ensure indices are in the correct format
        if isinstance(X, pd.DataFrame):
            X_train_fold = X.iloc[train_idx].values
            X_val_fold = X.iloc[val_idx].values
        else:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]

        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Handle imbalanced data (apply SMOTE only on training set)
        X_train_resampled, y_train_resampled = handle_imbalance(X_train_fold, y_train_fold, False)
        X_val_resampled, y_val_resampled = handle_imbalance(X_val_fold, y_val_fold, False)
        print(f"Number of samples in resampled validation set: {len(y_val_resampled)}")

        # Create prediction result arrays for the current fold's resampled validation set
        fold_meta_pred = np.zeros((len(y_val_resampled), len(base_models)))
        fold_meta_prob = np.zeros((len(y_val_resampled), len(base_models)))

        # Train base models
        for i, (model_name, model) in enumerate(base_models.items()):
            print(f"Training {model_name} model...")
            model.fit(X_train_resampled, y_train_resampled)

            # Predict on validation set
            y_pred = model.predict(X_val_resampled)
            y_prob = model.predict_proba(X_val_resampled)[:, 1]

            # Store current fold's prediction results
            fold_meta_pred[:, i] = y_pred
            fold_meta_prob[:, i] = y_prob

            # Calculate AUC score
            metrics = evaluate(y_val_resampled, y_pred, y_prob)

            # Print evaluation results
            print("\nModel evaluation results:")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"PR AUC: {metrics['pr_auc']:.4f}")
            print("\nClassification Report:")
            for label, values in metrics['classification_report'].items():
                if label in ['0', '1']:
                    print(f"Category {label}:")
                    print(f"  Precision: {values['precision']:.4f}")
                    print(f"  Recall: {values['recall']:.4f}")
                    print(f"  F1-score: {values['f1-score']:.4f}")
            auc_score = roc_auc_score(y_val_resampled, y_prob)
            base_auc_scores[model_name].append(auc_score)
            print(f"{model_name} Validation AUC: {auc_score:.4f}")

        # Add current fold's results to the total list
        all_meta_train_pred.append(fold_meta_pred)
        all_meta_train_prob.append(fold_meta_prob)
        all_meta_y.append(y_val_resampled)

    # Merge results from all folds
    meta_train_pred = np.vstack(all_meta_train_pred)
    meta_train_prob = np.vstack(all_meta_train_prob)
    meta_y = np.concatenate(all_meta_y)

    print(f"\nTotal number of samples for meta-model training data: {len(meta_y)}")

    # Train meta models
    print("\nTraining meta models...")
    # Ensure y is in numpy array format
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y

    # Meta-model 1: Use base model prediction results
    meta_models['meta_pred'].fit(meta_train_pred, meta_y)
    meta_pred_prob = meta_models['meta_pred'].predict_proba(meta_train_pred)[:, 1]
    meta_auc_scores['meta_pred'].append(roc_auc_score(meta_y, meta_pred_prob))
    print(f"Meta-model 1 (prediction results) AUC: {meta_auc_scores['meta_pred'][0]:.4f}")

    # Meta-model 2: Use base model prediction probabilities
    meta_models['meta_prob'].fit(meta_train_prob, meta_y)
    meta_prob_prob = meta_models['meta_prob'].predict_proba(meta_train_prob)[:, 1]
    meta_auc_scores['meta_prob'].append(roc_auc_score(meta_y, meta_prob_prob))
    print(f"Meta-model 2 (prediction probabilities) AUC: {meta_auc_scores['meta_prob'][0]:.4f}")

    # Soft voting ensemble
    ensemble_prob = (meta_pred_prob + meta_prob_prob) / 2
    ensemble_auc = roc_auc_score(meta_y, ensemble_prob)
    ensemble_auc_scores.append(ensemble_auc)
    print(f"Ensemble model AUC: {ensemble_auc:.4f}")

    # Save meta-model training data
    save_meta_training_data_to_csv(meta_train_pred, meta_train_prob, meta_y, meta_data_dir)

    # Summarize results
    cv_scores = {
        'base_models': {model_name: np.mean(scores) for model_name, scores in base_auc_scores.items()},
        'meta_models': {model_name: np.mean(scores) for model_name, scores in meta_auc_scores.items()},
        'ensemble': np.mean(ensemble_auc_scores)
    }

    return base_models, meta_models, cv_scores


# Prediction function
def predict(X, y,base_models, meta_models):
    """
    Make predictions using trained models
    
    Parameters:
        X: Feature matrix
        base_models: Trained base models
        meta_models: Trained meta models
        
    Returns:
        y_pred: Predicted labels
        y_prob: Prediction probabilities
    """
    # Ensure X is in correct format
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    # Make predictions with base models
    base_pred = np.zeros((X_array.shape[0], len(base_models)))
    base_prob = np.zeros((X_array.shape[0], len(base_models)))

    for i, (model_name, model) in enumerate(base_models.items()):
        print(f"Predicting with {model_name} model...")
        base_pred[:, i] = model.predict(X_array)
        base_prob[:, i] = model.predict_proba(X_array)[:, 1]
        # Evaluate model performance
        metrics = evaluate(y, base_pred[:, i], base_prob[:, i])

        # Print evaluation results
        print(f"\n{model_name} model evaluation results:")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")


    # Make predictions with meta models
    meta_pred_prob = meta_models['meta_pred'].predict_proba(base_pred)[:, 1]
    meta_prob_prob = meta_models['meta_prob'].predict_proba(base_prob)[:, 1]

    # Soft voting ensemble
    ensemble_prob = (meta_pred_prob + meta_prob_prob) / 2
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    return ensemble_pred, ensemble_prob


# Evaluation function
def evaluate(y_true, y_pred, y_prob):
    """
    Evaluate model performance
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        
    Returns:
        metrics: Evaluation metrics dictionary
    """
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_prob)

    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

    return metrics


# Visualization function
def visualize_results(y_true, y_prob, metrics):
    """
    Visualize model results
    
    Parameters:
        y_true: True labels
        y_prob: Prediction probabilities
        metrics: Evaluation metrics dictionary
    """
    # Set figure size
    plt.figure(figsize=(15, 12))

    # 1. ROC curve
    plt.subplot(2, 2, 1)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {metrics["roc_auc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend()

    # 2. PR curve
    plt.subplot(2, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision, label=f'PR AUC = {metrics["pr_auc"]:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend()

    # 3. Confusion matrix
    plt.subplot(2, 2, 3)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # 4. Prediction probability distribution
    plt.subplot(2, 2, 4)
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Non-Fraud', density=True)
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Fraud', density=True)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('fraud_detection_results.png')
    plt.show()


# Save models function
def save_models(base_models, meta_models, scaler, model_dir):
    """
    Save trained models to specified directory
    
    Parameters:
        base_models: Dictionary of trained base models
        meta_models: Dictionary of trained meta models
        scaler: Scaler object
        model_dir: Directory to save models
    """
    print(f"\nSaving models to {model_dir} directory...")

    # Create model directory if not exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save base models
    for model_name, model in base_models.items():
        model_path = os.path.join(model_dir, f"base_{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved base model {model_name} to {model_path}")

    # Save meta models
    for model_name, model in meta_models.items():
        model_path = os.path.join(model_dir, f"meta_{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved meta model {model_name} to {model_path}")

    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")


# Load models function
def load_models(model_dir):
    """
    Load models from specified directory
    
    Parameters:
        model_dir: Directory containing saved models
        
    Returns:
        base_models: Dictionary of loaded base models
        meta_models: Dictionary of loaded meta models
        scaler: Loaded scaler object
    """
    print(f"\nLoading models from {model_dir} directory...")

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist")
        return None, None, None

    try:
        # Load base models
        base_models = {}
        base_model_names = ['mlp', 'rf', 'svm', 'xgb']
        for model_name in base_model_names:
            model_path = os.path.join(model_dir, f"base_{model_name}.pkl")
            if os.path.exists(model_path):
                base_models[model_name] = joblib.load(model_path)
                print(f"Loaded base model {model_name} from {model_path}")
            else:
                print(f"Base model {model_name} does not exist")
                return None, None, None

        # Load meta models
        meta_models = {}
        meta_model_names = ['meta_pred', 'meta_prob']
        for model_name in meta_model_names:
            model_path = os.path.join(model_dir, f"meta_{model_name}.pkl")
            if os.path.exists(model_path):
                meta_models[model_name] = joblib.load(model_path)
                print(f"Loaded meta model {model_name} from {model_path}")
            else:
                print(f"Meta model {model_name} does not exist")
                return None, None, None

        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        else:
            print(f"Scaler does not exist")
            return None, None, None

        return base_models, meta_models, scaler

    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None


# Save meta-model training data to CSV
def save_meta_training_data_to_csv(meta_train_pred, meta_train_prob, meta_y, output_dir):
    """
    Save meta-model training data to CSV files
    
    Parameters:
        meta_train_pred: Base models' prediction results
        meta_train_prob: Base models' prediction probabilities
        meta_y: True labels
        output_dir: Output directory
    """
    print(f"\nSaving meta-model training data to {output_dir} directory...")

    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save base models' prediction results
    pred_columns = [f'model_{i}_pred' for i in range(meta_train_pred.shape[1])]
    pred_df = pd.DataFrame(meta_train_pred, columns=pred_columns)
    pred_df['true_label'] = meta_y
    pred_path = os.path.join(output_dir, "meta_train_pred.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Base models' prediction results saved to {pred_path}")

    # Save base models' prediction probabilities
    prob_columns = [f'model_{i}_prob' for i in range(meta_train_prob.shape[1])]
    prob_df = pd.DataFrame(meta_train_prob, columns=prob_columns)
    prob_df['true_label'] = meta_y
    prob_path = os.path.join(output_dir, "meta_train_prob.csv")
    prob_df.to_csv(prob_path, index=False)
    print(f"Base models' prediction probabilities saved to {prob_path}")


# Save predictions to CSV
def save_predictions_to_csv(y_true, y_pred, y_prob, file_path):
    """
    Save prediction results to CSV file
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        file_path: Path to save CSV file
    """
    print(f"\nSaving prediction results to {file_path}...")

    # Create DataFrame containing prediction results
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'fraud_probability': y_prob
    })

    # Save to CSV
    results_df.to_csv(file_path, index=False)
    print(f"Prediction results saved to {file_path}")


# Main function
def main():
    # Set data paths
    train_path = "../data/train_transaction.csv"
    train_id_path = "../data/train_identity.csv"
    test_path = ""
    model_dir = "models"  # Model save directory
    results_path = "fraud_detection_results.csv"  # Prediction results save path
    meta_data_dir = "meta_model_data"  # Meta-model training data save directory

    # Ask user whether to train a new model or load an existing one
    print("\nPlease select an option:")
    print("1. Train new model")
    print("2. Load existing models")

    try:
        # choice = input("Please enter your choice (1 or 2): ")

        if True:
            # Train a new model
            print("\nStarting to train new models...")

            # Load data (using simulated data)
            X, y, X_test, y_test, scaler = load_and_preprocess_data(train_path, train_id_path, test_path)

            # If no test set is provided, split the training set
            if X_test is None or y_test is None:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, y_train = X, y

            # Train and evaluate models
            base_models, meta_models, cv_scores = train_and_evaluate(X_train, y_train, n_splits=5,
                                                                     meta_data_dir=meta_data_dir)

            # Save models
            save_models(base_models, meta_models, scaler, model_dir)

        elif choice == "2":
            # Load existing models
            print("\nLoading existing models...")
            base_models, meta_models, scaler = load_models(model_dir)

            if base_models is None or meta_models is None:
                print("Failed to load models, please train models first")
                return

            # Load data (using simulated data)
            X, y, X_test, y_test, _ = load_and_preprocess_data(train_path, train_id_path, test_path, sample_size=10000,
                                                               scaler=scaler)

            # If no test set is provided, split the training set
            if X_test is None or y_test is None:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        else:
            print("Invalid option, please enter 1 or 2")
            return

        # Make predictions on test set
        print("\nMaking predictions on test set...")
        y_pred, y_prob = predict(X_test, y_test,base_models, meta_models)

        # Evaluate model performance
        metrics = evaluate(y_test, y_pred, y_prob)

        # Print evaluation results
        print("\nModel evaluation results:")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")
        print("\nClassification Report:")
        for label, values in metrics['classification_report'].items():
            if label in ['0', '1']:
                print(f"Category {label}:")
                print(f"  Precision: {values['precision']:.4f}")
                print(f"  Recall: {values['recall']:.4f}")
                print(f"  F1-score: {values['f1-score']:.4f}")

        # Save prediction results to CSV
        save_predictions_to_csv(y_test, y_pred, y_prob, results_path)

        # Visualize results
        visualize_results(y_test, y_prob, metrics)

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
