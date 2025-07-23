# -*- coding: utf-8 -*-
"""Use saved Stacking model for fraud detection on validation set"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Load model function
def load_models(model_dir):
    """
    Load models from specified directory
    
    Args:
        model_dir: Model save directory
        
    Returns:
        base_models: Base model dictionary
        meta_models: Meta model dictionary
        scaler: Standardizer
        selected_features: Feature list
    """
    print(f"\nLoading models from {model_dir} directory...")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist")
        return None, None, None
    
    try:
        # Load base models
        base_models = {}
        base_model_names = ['lgb', 'rf', 'xgb','catboost'] # Note: no svm here
        for model_name in base_model_names:
            model_path = os.path.join(model_dir, f"base_{model_name}.pkl")
            if os.path.exists(model_path):
                base_models[model_name] = joblib.load(model_path)
                print(f"Loading base model {model_name} from {model_path}")
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
                print(f"Loading meta model {model_name} from {model_path}")
            else:
                print(f"Meta model {model_name} does not exist")
                return None, None, None
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"Loading scaler from {scaler_path}")
        else:
            print(f"Scaler does not exist")
            return None, None, None
        
        # Load feature list
        # features_path = os.path.join(model_dir, "selectedfeatures.pkl")
        features_path = os.path.join(model_dir, "feature_info.pkl")
        selected_features = joblib.load(features_path)
        
        return base_models, meta_models, scaler, selected_features
    
    except Exception as e:
        print(f"Error occurred while loading models: {e}")
        return None, None, None, None

# Prediction function
def predict(X, base_models, meta_models):
    """
    Make predictions using trained models
    
    Args:
        X: Feature matrix
        base_models: Trained base models
        meta_models: Trained meta models
        
    Returns:
        y_prob: Prediction probability
    """
    # Ensure X is in correct format
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    # Use base models for prediction
    base_pred = np.zeros((X_array.shape[0], len(base_models)))
    base_prob = np.zeros((X_array.shape[0], len(base_models)))
    
    for i, (model_name, model) in enumerate(base_models.items()):
        
        base_pred[:, i] = model.predict(X_array)
        base_prob[:, i] = model.predict_proba(X_array)[:, 1]
        
    
    # Use meta models for prediction
    meta_pred_prob = meta_models['meta_pred'].predict_proba(base_pred)[:, 1]
    meta_prob_prob = meta_models['meta_prob'].predict_proba(base_prob)[:, 1]
    
    # Soft voting ensemble
    ensemble_prob = meta_pred_prob*0.25 + meta_prob_prob*0.75
    # ensemble_prob = meta_prob_prob
    # ensemble_prob = meta_pred_prob


    return ensemble_prob

def preprocess_validation_data(val_path, val_id_path, scaler, feature_info):
    print("Loading validation data...")
    try:
        val1 = pd.read_csv(val_path)
        val2 = pd.read_csv(val_id_path)
        val_df = pd.merge(val1, val2, on="TransactionID", how="left")
        transaction_ids = val_df['TransactionID']
        val_df = val_df.drop_duplicates(subset='TransactionID', keep='first')
    except Exception as e:
        print(f"Unable to load validation data: {e}")
        return None, None
    # Handle dashes in column names
    val_df.columns = [col.replace("id-", "id_") if col.startswith("id-") else col for col in val_df.columns]
    
    # Fill missing values
    for col in val_df.columns:
        if val_df[col].dtype == 'object':
            val_df[col].fillna('missing', inplace=True)
        else:
            val_df[col].fillna(-1, inplace=True)

    # Handle V features
    v_cols = [col for col in val_df.columns if col.startswith('V')]
    if v_cols:
        v_filled = val_df[v_cols].fillna(-1)
        pca = PCA(n_components=5)
        v_pca = pca.fit_transform(v_filled)
        for i in range(5):
            val_df[f'V_PCA_{i+1}'] = v_pca[:, i]
        val_df.drop(columns=v_cols, inplace=True)

    # Construct features
    if 'TransactionDT' in val_df.columns:
        val_df['hour'] = (val_df['TransactionDT'] / 3600) % 24
        val_df['day'] = (val_df['TransactionDT'] / (3600 * 24)) % 7
        val_df['is_night'] = ((val_df['hour'] >= 22) | (val_df['hour'] <= 6)).astype(int)
        val_df['is_rush_hour'] = ((val_df['hour'].between(7, 9)) | (val_df['hour'].between(17, 19))).astype(int)

    if 'TransactionAmt' in val_df.columns:
        val_df['TransactionAmt_log'] = np.log1p(val_df['TransactionAmt'])
        val_df['TransactionAmt_decimal'] = val_df['TransactionAmt'] % 1
        val_df['TransactionAmt_round'] = (val_df['TransactionAmt_decimal'] == 0).astype(int)

        # Add transaction amount ratio features
        if 'card1' in val_df.columns:
            card1_amt_mean = val_df.groupby('card1')['TransactionAmt'].transform('mean')
            card1_amt_std = val_df.groupby('card1')['TransactionAmt'].transform('std').replace(0, 1)
            val_df['card1_amt_z'] = (val_df['TransactionAmt'] - card1_amt_mean) / card1_amt_std
            val_df['TransactionAmt_to_mean_card1'] = val_df['TransactionAmt'] / val_df.groupby(['card1'])['TransactionAmt'].transform('mean')
            val_df['TransactionAmt_to_std_card1'] = val_df['TransactionAmt'] / val_df.groupby(['card1'])['TransactionAmt'].transform('std')
        if 'card4' in val_df.columns:
            val_df['TransactionAmt_to_mean_card4'] = val_df['TransactionAmt'] / val_df.groupby(['card4'])['TransactionAmt'].transform('mean')
            val_df['TransactionAmt_to_std_card4'] = val_df['TransactionAmt'] / val_df.groupby(['card4'])['TransactionAmt'].transform('std')
    
    # Add id_02 ratio features
    if 'id_02' in val_df.columns:
        if 'card1' in val_df.columns:
            val_df['id_02_to_mean_card1'] = val_df['id_02'] / val_df.groupby(['card1'])['id_02'].transform('mean')
            val_df['id_02_to_std_card1'] = val_df['id_02'] / val_df.groupby(['card1'])['id_02'].transform('std')
        if 'card4' in val_df.columns:
            val_df['id_02_to_mean_card4'] = val_df['id_02'] / val_df.groupby(['card4'])['id_02'].transform('mean')
            val_df['id_02_to_std_card4'] = val_df['id_02'] / val_df.groupby(['card4'])['id_02'].transform('std')
        
        # Handle infinity and NaN values
        for col in ['id_02_to_mean_card1', 'id_02_to_mean_card4', 'id_02_to_std_card1', 'id_02_to_std_card4']:
            if col in val_df.columns:
                val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan).fillna(-1)
    
    # Add D15 ratio features
    if 'D15' in val_df.columns:
        if 'card1' in val_df.columns:
            val_df['D15_to_mean_card1'] = val_df['D15'] / val_df.groupby(['card1'])['D15'].transform('mean')
            val_df['D15_to_std_card1'] = val_df['D15'] / val_df.groupby(['card1'])['D15'].transform('std')
        if 'card4' in val_df.columns:
            val_df['D15_to_mean_card4'] = val_df['D15'] / val_df.groupby(['card4'])['D15'].transform('mean')
            val_df['D15_to_std_card4'] = val_df['D15'] / val_df.groupby(['card4'])['D15'].transform('std')
        if 'addr1' in val_df.columns:
            val_df['D15_to_mean_addr1'] = val_df['D15'] / val_df.groupby(['addr1'])['D15'].transform('mean')
            val_df['D15_to_std_addr1'] = val_df['D15'] / val_df.groupby(['addr1'])['D15'].transform('std')
        if 'addr2' in val_df.columns:
            val_df['D15_to_mean_addr2'] = val_df['D15'] / val_df.groupby(['addr2'])['D15'].transform('mean')
            val_df['D15_to_std_addr2'] = val_df['D15'] / val_df.groupby(['addr2'])['D15'].transform('std')
        
        # Handle infinity and NaN values
        D15_ratio_cols = [
            'D15_to_mean_card1', 'D15_to_mean_card4', 'D15_to_std_card1', 'D15_to_std_card4',
            'D15_to_mean_addr1', 'D15_to_mean_addr2', 'D15_to_std_addr1', 'D15_to_std_addr2'
        ]
        for col in D15_ratio_cols:
            if col in val_df.columns:
                val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan).fillna(-1)
    # Handle P_emaildomain
    if 'P_emaildomain' in val_df.columns:
        p_series = val_df['P_emaildomain'].fillna('missing').astype(str)
        p_split = p_series.str.split('.', expand=True, n=2)
        for i in range(3):
            col_name = f'P_emaildomain_{i+1}'
            val_df[col_name] = p_split[i] if i in p_split.columns else 'missing'
            val_df[col_name] = val_df[col_name].fillna('missing')

    # Handle R_emaildomain
    if 'R_emaildomain' in val_df.columns:
        r_series = val_df['R_emaildomain'].fillna('missing').astype(str)
        r_split = r_series.str.split('.', expand=True, n=2)
        for i in range(3):
            col_name = f'R_emaildomain_{i+1}'
            val_df[col_name] = r_split[i] if i in r_split.columns else 'missing'
            val_df[col_name] = val_df[col_name].fillna('missing')
    
    # Frequency encoding
    cat_cols = [col for col in val_df.columns if val_df[col].nunique() < 50]
    for col in cat_cols:
        freq = val_df[col].value_counts()
        val_df[col + "_freq"] = val_df[col].map(freq)

    # Encode categorical features
    for col in val_df.select_dtypes(include=['object']).columns:
        val_df[col] = LabelEncoder().fit_transform(val_df[col].astype(str))

    # Handle infinity and extreme values
    numerical_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        # Replace infinity values
        val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan)
        # Fill NaN values with median
        val_df[col] = val_df[col].fillna(val_df[col].median() if not val_df[col].isna().all() else -1)

    # Get scaler feature names
    scaler_features = scaler.feature_names_in_
    
    # Check for missing features and add them
    missing_features = [col for col in scaler_features if col not in val_df.columns]
    print(f"Missing scaler features: {missing_features}")
    
    # Create columns for missing features and fill with default value -1
    for col in missing_features:
        val_df[col] = -1
    
    # Ensure feature order is consistent with training
    val_df_scaled = val_df[scaler_features].copy()
    
    # Standardize numerical features
    val_df_scaled = scaler.transform(val_df_scaled)
    
    # Put standardized values back into DataFrame
    val_df[scaler_features] = val_df_scaled

    # Load final selected feature list
    features_path = os.path.join('./models', "selectedfeatures.pkl")
    final_features = joblib.load(features_path)
    
    # Ensure only existing features are selected
    final_features_exist = [col for col in final_features if col in val_df.columns]
    print(f"Missing features: {set(final_features) - set(final_features_exist)}")
    
    # Select final features
    X_val = val_df[final_features_exist]
    
    # Check for missing features, if any, add missing features and fill with -1
    for col in final_features:
        if col not in val_df.columns:
            X_val[col] = -1

    return X_val, transaction_ids


# Main function
if __name__ == "__main__":
    # Define paths
    MODEL_DIR = "models"
    VAL_TRANSACTION_PATH = "./data/test_transaction.csv" # Assume validation data is in test_transaction.csv
    VAL_IDENTITY_PATH = "./data/test_identity.csv"   # Assume validation data is in test_identity.csv
    OUTPUT_PATH = "res.csv"

    # Load models and feature information
    base_models, meta_models, scaler, feature_info = load_models(MODEL_DIR)

    if base_models and meta_models and scaler and feature_info:
        # Load and preprocess validation data
        X_val, transaction_ids = preprocess_validation_data(
            VAL_TRANSACTION_PATH, 
            VAL_IDENTITY_PATH, 
            scaler,
            feature_info
        )
        
        if X_val is not None:
            # Use model for prediction
            print("\nStarting prediction...")
            y_prob = predict(X_val, base_models, meta_models)
            
            # Save prediction results
            print(f"\nSaving prediction results to {OUTPUT_PATH}")
            output_df = pd.DataFrame({
                'TransactionID': transaction_ids,
                'isFraud': y_prob
            })
            output_df.to_csv(OUTPUT_PATH, index=False)
            print("Prediction completed!")
        else:
            print("Failed to preprocess validation data")
    else:
        print("Failed to load models")