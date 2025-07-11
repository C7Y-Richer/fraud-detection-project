import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import os
import time

# ==== CONFIGURATION ====
MODEL_NAME = "svm"
RESULTS_DIR = f"/users/qdph9564/results/{MODEL_NAME}"
MODELS_DIR = f"/users/qdph9564/models/{MODEL_NAME}"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    train1 = pd.read_csv('/users/qdph9564/data/ieee-fraud-detection/train_transaction.csv')
    train2 = pd.read_csv('/users/qdph9564/data/ieee-fraud-detection/train_identity.csv')
    train = pd.merge(train1, train2, on="TransactionID", how="left")
    train = train.drop_duplicates(subset='TransactionID', keep='first')
    return train


def preprocess_data(train):
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col].fillna('missing', inplace=True)
        else:
            train[col].fillna(train[col].median(), inplace=True)

    label_encoder = LabelEncoder()
    for col in train.select_dtypes(include=['object']).columns:
        train[col] = label_encoder.fit_transform(train[col])

    N_UNIQUE_THRESHOLD = 55
    categorical_features = [feature for feature in train.columns
                             if train[feature].nunique() < N_UNIQUE_THRESHOLD]
    numerical_features = [feature for feature in train.select_dtypes(include=[np.number]).columns
                           if train[feature].nunique() >= N_UNIQUE_THRESHOLD]

    scaler_standard = StandardScaler()
    train[numerical_features] = scaler_standard.fit_transform(train[numerical_features])
    scaler_minmax = MinMaxScaler()
    train[numerical_features] = scaler_minmax.fit_transform(train[numerical_features])

    correlation_matrix = train[numerical_features].corr()
    correlation_threshold = 0.8
    high_correlation_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                colname = correlation_matrix.columns[i]
                high_correlation_features.add(colname)

    selected_features = [feature for feature in numerical_features if feature not in high_correlation_features]
    return train[selected_features], train['isFraud']


def save_outputs(model, y_valid, y_pred, X_valid, model_params):
    joblib.dump(model, f"{MODELS_DIR}/{MODEL_NAME}_model.pkl")

    with open(f"{RESULTS_DIR}/{MODEL_NAME}_report.txt", "w") as f:
        f.write("Accuracy: {:.4f}\n".format(accuracy_score(y_valid, y_pred)))
        f.write("Classification Report:\n")
        f.write(classification_report(y_valid, y_pred))

    proba_df = pd.DataFrame({
        'TransactionID': X_valid.index,
        'y_pred': y_pred,
        'proba_0': model.predict_proba(X_valid)[:, 0],
        'proba_1': model.predict_proba(X_valid)[:, 1]
    })
    proba_df.to_csv(f"{RESULTS_DIR}/{MODEL_NAME}_predictions.csv", index=False)

    with open(f"{RESULTS_DIR}/{MODEL_NAME}_params.json", "w") as f:
        json.dump(model_params, f, indent=4)


def train_model(X_train, y_train):
    # 使用 LinearSVC 和 CalibratedClassifierCV 包裹来支持 predict_proba
    base_model = LinearSVC(C=1.0, max_iter=10000, random_state=42)
    model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)  # 使用 Platt Scaling（sigmoid）
    model.fit(X_train, y_train)

    model_params = {
        "base_model": "LinearSVC",
        "C": 1.0,
        "max_iter": 10000,
        "calibration": "sigmoid",
        "cv": 5
    }

    return model, model_params


if __name__ == '__main__':
    start = time.time()
    print("[INFO] Job started.")

    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    model, model_params = train_model(X_train_smote, y_train_smote)
    y_pred = model.predict(X_valid)

    save_outputs(model, y_valid, y_pred, X_valid, model_params)

    print("[INFO] Job completed in {:.2f} minutes.".format((time.time() - start) / 60))