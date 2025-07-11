#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fraud Detection Model for HPC Environment
Usage: sbatch fraud_detection_hpc.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import time
import os


def main():
    # 记录开始时间
    start_time = time.time()
    print(f"Job started at: {time.ctime(start_time)}")

    # 1. 数据加载
    print("Loading data...")
    try:
        train1 = pd.read_csv('/users/qdph9564/data/ieee-fraud-detection/train_transaction.csv')
        train2 = pd.read_csv('/users/qdph9564/data/ieee-fraud-detection/train_identity.csv')
        train = pd.merge(train1, train2, on="TransactionID", how="left")
        train = train.drop_duplicates(subset='TransactionID', keep='first')
        print(f"Training data loaded. Shape: {train.shape}")
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return

    # 2. 数据预处理
    print("Preprocessing data...")

    # 处理缺失值
    for col in train.columns:
        if train[col].dtype == 'object':
            train[col].fillna('missing', inplace=True)
        else:
            train[col].fillna(train[col].median(), inplace=True)

    # 类别编码
    label_encoder = LabelEncoder()
    for col in train.select_dtypes(include=['object']).columns:
        train[col] = label_encoder.fit_transform(train[col])

    # 3. 特征工程
    print("Feature engineering...")

    # 区分数值型和类别型特征
    N_UNIQUE_THRESHOLD = 55
    categorical_features = [feature for feature in train.columns
                            if train[feature].nunique() < N_UNIQUE_THRESHOLD]
    numerical_features = [feature for feature in train.select_dtypes(include=[np.number]).columns
                          if train[feature].nunique() >= N_UNIQUE_THRESHOLD]

    # 标准化与归一化
    scaler_standard = StandardScaler()
    train[numerical_features] = scaler_standard.fit_transform(train[numerical_features])
    scaler_minmax = MinMaxScaler()
    train[numerical_features] = scaler_minmax.fit_transform(train[numerical_features])

    # 特征选择 (基于相关性)
    correlation_matrix = train[numerical_features].corr()
    correlation_threshold = 0.8
    high_correlation_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                colname = correlation_matrix.columns[i]
                high_correlation_features.add(colname)

    selected_features = [feature for feature in numerical_features if feature not in high_correlation_features]
    print(f"Selected {len(selected_features)} features for modeling")

    # 4. 准备训练数据
    X = train[selected_features]
    y = train['isFraud']

    # 划分训练集和验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # 处理类别不平衡 (SMOTE过采样)
    print("Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 5. 模型训练
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)  # 减少树的数量以加快HPC测试
    rf_model.fit(X_train_smote, y_train_smote)

    # 6. 模型评估
    print("Evaluating model...")
    y_pred = rf_model.predict(X_valid)

    print("\nModel Performance:")
    print("Accuracy:", accuracy_score(y_valid, y_pred))
    print("Classification Report:\n", classification_report(y_valid, y_pred))

    # 记录结束时间
    end_time = time.time()
    print(f"\nJob completed at: {time.ctime(end_time)}")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()