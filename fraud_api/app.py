# -*- coding: utf-8 -*-
"""Flask API server providing fraud detection prediction interface"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import joblib
from validate import load_models, predict, preprocess_validation_data
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store models
base_models = None
meta_models = None
scaler = None
feature_info = None
selected_features = None

def init_models():
    """Initialize models"""
    global base_models, meta_models, scaler, feature_info, selected_features
    
    MODEL_DIR = "models"
    if os.path.exists(MODEL_DIR):
        base_models, meta_models, scaler, feature_info = load_models(MODEL_DIR)
        
        # Load selected features
        features_path = os.path.join(MODEL_DIR, "selectedfeatures.pkl")
        if os.path.exists(features_path):
            selected_features = joblib.load(features_path)
        
        if base_models and meta_models and scaler:
            print("Models loaded successfully")
            return True
    
    print("Failed to load models")
    return False

def preprocess_csv_data(transaction_csv, identity_csv=None):
    """Preprocess CSV data, referring to validate.py methods"""
    try:
        # Parse transaction CSV
        from io import StringIO
        transaction_df = pd.read_csv(StringIO(transaction_csv))
        
        # If identity CSV exists, parse and merge
        if identity_csv and identity_csv.strip():
            identity_df = pd.read_csv(StringIO(identity_csv))
            # Merge data, referring to validate.py methods
            val_df = pd.merge(transaction_df, identity_df, on="TransactionID", how="left")
        else:
            val_df = transaction_df
        
        # Get TransactionID
        transaction_ids = val_df['TransactionID'] if 'TransactionID' in val_df.columns else pd.Series([0])
        val_df = val_df.drop_duplicates(subset='TransactionID', keep='first') if 'TransactionID' in val_df.columns else val_df
        
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
        
        # Handle email domains
        if 'P_emaildomain' in val_df.columns:
            val_df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = val_df['P_emaildomain'].str.split('.', expand=True)
            for col in ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']:
                val_df[col] = val_df[col].fillna('missing')
        
        if 'R_emaildomain' in val_df.columns:
            val_df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = val_df['R_emaildomain'].str.split('.', expand=True)
            for col in ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']:
                val_df[col] = val_df[col].fillna('missing')
        
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
            val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan)
            val_df[col] = val_df[col].fillna(val_df[col].median() if not val_df[col].isna().all() else -1)
        
        # Get scaler feature names
        if scaler and hasattr(scaler, 'feature_names_in_'):
            scaler_features = scaler.feature_names_in_
            
            # Check for missing features and add them
            for col in scaler_features:
                if col not in val_df.columns:
                    val_df[col] = -1
            
            # Ensure feature order is consistent with training
            val_df_scaled = val_df[scaler_features].copy()
            
            # Standardize numerical features
            val_df_scaled = scaler.transform(val_df_scaled)
            
            # Put standardized values back into DataFrame
            val_df[scaler_features] = val_df_scaled
        
        # Select final features
        if selected_features:
            for col in selected_features:
                if col not in val_df.columns:
                    val_df[col] = -1
            
            X = val_df[selected_features]
        else:
            X = val_df
        
        return X, transaction_ids
    
    except Exception as e:
        raise Exception(f"CSV data preprocessing failed: {str(e)}")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/predict/single', methods=['POST'])
def predict_single():
    """Single row data prediction interface - receives two rows of CSV data"""
    try:
        if not base_models or not meta_models:
            return jsonify({'error': 'Models not loaded'}), 500
        
        data = request.json
        if not data or 'transaction_csv' not in data:
            return jsonify({'error': 'Please provide transaction CSV data'}), 400
        
        transaction_csv = data['transaction_csv']
        identity_csv = data.get('identity_csv', '')
        
        if not transaction_csv.strip():
            return jsonify({'error': 'Transaction CSV data cannot be empty'}), 400
        
        # Preprocess CSV data
        X, transaction_ids = preprocess_csv_data(transaction_csv, identity_csv)
        
        # Make prediction
        prob = predict(X, base_models, meta_models)
        
        result = {
            'fraud_probability': float(prob[0]),
            'is_fraud': bool(prob[0] > 0.5),
            'risk_level': 'High' if prob[0] > 0.7 else 'Medium' if prob[0] > 0.3 else 'Low',
            'transaction_id': str(transaction_ids.iloc[0]) if len(transaction_ids) > 0 else 'N/A'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/file', methods=['POST'])
def predict_file():
    """File upload prediction interface - receives two CSV files"""
    try:
        if not base_models or not meta_models:
            return jsonify({'error': 'Models not loaded'}), 500
        
        if 'transaction_file' not in request.files:
            return jsonify({'error': 'Please upload transaction.csv file'}), 400
        
        transaction_file = request.files['transaction_file']
        identity_file = request.files['identity_file']
        if transaction_file.filename == '':
            return jsonify({'error': 'Please select transaction.csv file'}), 400
        
        if transaction_file and transaction_file.filename.endswith('.csv'):
            # Save transaction file
            transaction_filename = secure_filename(transaction_file.filename)
            transaction_filepath = os.path.join(app.config['UPLOAD_FOLDER'], transaction_filename)
            transaction_file.save(transaction_filepath)
            
            identity_filepath = None
            if identity_file and identity_file.filename != '' and identity_file.filename.endswith('.csv'):
                # Save identity file
                identity_filename = secure_filename(identity_file.filename)
                identity_filepath = os.path.join(app.config['UPLOAD_FOLDER'], identity_filename)
                identity_file.save(identity_filepath)
            
            try:
                
                # Use preprocessing method from validate.py
                X_val, transaction_ids = preprocess_validation_data(
                    transaction_filepath, 
                    identity_filepath, 
                    scaler,
                    feature_info
                )
                
                if X_val is not None:
                    # Make prediction
                    y_prob = predict(X_val, base_models, meta_models)
                    
                    # Build results
                    results = []
                    for i, (tid, prob) in enumerate(zip(transaction_ids, y_prob)):
                        result = {
                            'TransactionID': str(tid),
                            'fraud_probability': float(prob),
                            'is_fraud': bool(prob > 0.5),
                            'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
                        }
                        results.append(result)
                    
                    return jsonify({
                        'total_records': len(results),
                        'predictions': results
                    })
                else:
                    return jsonify({'error': 'Data preprocessing failed'}), 500
                    
            finally:
                # Delete temporary files
                if os.path.exists(transaction_filepath):
                    os.remove(transaction_filepath)
                if identity_filepath and os.path.exists(identity_filepath):
                    os.remove(identity_filepath)
        
        return jsonify({'error': 'Please upload CSV files'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check interface"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': base_models is not None and meta_models is not None
    })

if __name__ == '__main__':
    # Initialize models
    init_models()
    
    # Start Flask application
    app.run(debug=True, host='0.0.0.0', port=5001)