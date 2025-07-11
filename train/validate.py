# -*- coding: utf-8 -*-
"""使用已保存的Stacking模型对验证集进行欺诈检测"""

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 加载模型函数
def load_models(model_dir):
    """
    从指定目录加载模型
    
    参数:
        model_dir: 模型保存目录
        
    返回:
        base_models: 基础模型字典
        meta_models: 元模型字典
        scaler: 标准化器
        selected_features: 特征列表
    """
    print(f"\n从 {model_dir} 目录加载模型...")
    
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        print(f"模型目录 {model_dir} 不存在")
        return None, None, None
    
    try:
        # 加载基础模型
        base_models = {}
        base_model_names = ['lgb', 'rf', 'xgb','catboost'] # 注意：这里没有svm
        for model_name in base_model_names:
            model_path = os.path.join(model_dir, f"base_{model_name}.pkl")
            if os.path.exists(model_path):
                base_models[model_name] = joblib.load(model_path)
                print(f"加载基础模型 {model_name} 从 {model_path}")
            else:
                print(f"基础模型 {model_name} 不存在")
                return None, None, None
        
        # 加载元模型
        meta_models = {}
        meta_model_names = ['meta_pred', 'meta_prob']
        for model_name in meta_model_names:
            model_path = os.path.join(model_dir, f"meta_{model_name}.pkl")
            if os.path.exists(model_path):
                meta_models[model_name] = joblib.load(model_path)
                print(f"加载元模型 {model_name} 从 {model_path}")
            else:
                print(f"元模型 {model_name} 不存在")
                return None, None, None
        
        # 加载标准化器
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"加载标准化器从 {scaler_path}")
        else:
            print(f"标准化器不存在")
            return None, None, None
        
        # 加载特征列表
        # features_path = os.path.join(model_dir, "selectedfeatures.pkl")
        features_path = os.path.join(model_dir, "feature_info.pkl")
        selected_features = joblib.load(features_path)
        
        return base_models, meta_models, scaler, selected_features
    
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None, None, None, None

# 预测函数
def predict(X, base_models, meta_models):
    """
    使用训练好的模型进行预测
    
    参数:
        X: 特征矩阵
        base_models: 训练好的基础模型
        meta_models: 训练好的元模型
        
    返回:
        y_prob: 预测概率
    """
    # 确保X是正确的格式
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
        
    # 使用基础模型进行预测
    base_pred = np.zeros((X_array.shape[0], len(base_models)))
    base_prob = np.zeros((X_array.shape[0], len(base_models)))
    
    for i, (model_name, model) in enumerate(base_models.items()):
        base_pred[:, i] = model.predict(X_array)
        base_prob[:, i] = model.predict_proba(X_array)[:, 1]
    
    # 使用元模型进行预测
    meta_pred_prob = meta_models['meta_pred'].predict_proba(base_pred)[:, 1]
    meta_prob_prob = meta_models['meta_prob'].predict_proba(base_prob)[:, 1]
    
    # 软投票集成
    ensemble_prob = meta_pred_prob*0.25 + meta_prob_prob*0.75
    # ensemble_prob = meta_prob_prob
    # ensemble_prob = meta_pred_prob


    return ensemble_prob

def preprocess_validation_data(val_path, val_id_path, scaler, feature_info):
    print("加载验证数据...")
    try:
        val1 = pd.read_csv(val_path)
        val2 = pd.read_csv(val_id_path)
        val_df = pd.merge(val1, val2, on="TransactionID", how="left")
        transaction_ids = val_df['TransactionID']
        val_df = val_df.drop_duplicates(subset='TransactionID', keep='first')
    except Exception as e:
        print(f"无法加载验证数据: {e}")
        return None, None

    # 处理列名中的破折号
    val_df.columns = [col.replace("id-", "id_") if col.startswith("id-") else col for col in val_df.columns]

    # 填充缺失值
    for col in val_df.columns:
        if val_df[col].dtype == 'object':
            val_df[col].fillna('missing', inplace=True)
        else:
            val_df[col].fillna(-1, inplace=True)

    # 处理V特征
    v_cols = [col for col in val_df.columns if col.startswith('V')]
    if v_cols:
        v_filled = val_df[v_cols].fillna(-1)
        pca = PCA(n_components=5)
        v_pca = pca.fit_transform(v_filled)
        for i in range(5):
            val_df[f'V_PCA_{i+1}'] = v_pca[:, i]
        val_df.drop(columns=v_cols, inplace=True)

    # 构造特征
    if 'TransactionDT' in val_df.columns:
        val_df['hour'] = (val_df['TransactionDT'] / 3600) % 24
        val_df['day'] = (val_df['TransactionDT'] / (3600 * 24)) % 7
        val_df['is_night'] = ((val_df['hour'] >= 22) | (val_df['hour'] <= 6)).astype(int)
        val_df['is_rush_hour'] = ((val_df['hour'].between(7, 9)) | (val_df['hour'].between(17, 19))).astype(int)

    if 'TransactionAmt' in val_df.columns:
        val_df['TransactionAmt_log'] = np.log1p(val_df['TransactionAmt'])
        val_df['TransactionAmt_decimal'] = val_df['TransactionAmt'] % 1
        val_df['TransactionAmt_round'] = (val_df['TransactionAmt_decimal'] == 0).astype(int)

        # 添加交易金额比率特征
        if 'card1' in val_df.columns:
            card1_amt_mean = val_df.groupby('card1')['TransactionAmt'].transform('mean')
            card1_amt_std = val_df.groupby('card1')['TransactionAmt'].transform('std').replace(0, 1)
            val_df['card1_amt_z'] = (val_df['TransactionAmt'] - card1_amt_mean) / card1_amt_std
            val_df['TransactionAmt_to_mean_card1'] = val_df['TransactionAmt'] / val_df.groupby(['card1'])['TransactionAmt'].transform('mean')
            val_df['TransactionAmt_to_std_card1'] = val_df['TransactionAmt'] / val_df.groupby(['card1'])['TransactionAmt'].transform('std')
        if 'card4' in val_df.columns:
            val_df['TransactionAmt_to_mean_card4'] = val_df['TransactionAmt'] / val_df.groupby(['card4'])['TransactionAmt'].transform('mean')
            val_df['TransactionAmt_to_std_card4'] = val_df['TransactionAmt'] / val_df.groupby(['card4'])['TransactionAmt'].transform('std')

    # 添加id_02比率特征
    if 'id_02' in val_df.columns:
        if 'card1' in val_df.columns:
            val_df['id_02_to_mean_card1'] = val_df['id_02'] / val_df.groupby(['card1'])['id_02'].transform('mean')
            val_df['id_02_to_std_card1'] = val_df['id_02'] / val_df.groupby(['card1'])['id_02'].transform('std')
        if 'card4' in val_df.columns:
            val_df['id_02_to_mean_card4'] = val_df['id_02'] / val_df.groupby(['card4'])['id_02'].transform('mean')
            val_df['id_02_to_std_card4'] = val_df['id_02'] / val_df.groupby(['card4'])['id_02'].transform('std')
        
        # 处理无穷大和NaN值
        for col in ['id_02_to_mean_card1', 'id_02_to_mean_card4', 'id_02_to_std_card1', 'id_02_to_std_card4']:
            if col in val_df.columns:
                val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan).fillna(-1)

    # 添加D15比率特征
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
        
        # 处理无穷大和NaN值
        D15_ratio_cols = [
            'D15_to_mean_card1', 'D15_to_mean_card4', 'D15_to_std_card1', 'D15_to_std_card4',
            'D15_to_mean_addr1', 'D15_to_mean_addr2', 'D15_to_std_addr1', 'D15_to_std_addr2'
        ]
        for col in D15_ratio_cols:
            if col in val_df.columns:
                val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan).fillna(-1)

    # 处理邮箱域名
    if 'P_emaildomain' in val_df.columns:
        val_df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = val_df['P_emaildomain'].str.split('.', expand=True)
        # 处理NaN值
        for col in ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']:
            val_df[col] = val_df[col].fillna('missing')
    if 'R_emaildomain' in val_df.columns:
        val_df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = val_df['R_emaildomain'].str.split('.', expand=True)
        # 处理NaN值
        for col in ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']:
            val_df[col] = val_df[col].fillna('missing')

    # 频率编码
    cat_cols = [col for col in val_df.columns if val_df[col].nunique() < 50]
    for col in cat_cols:
        freq = val_df[col].value_counts()
        val_df[col + "_freq"] = val_df[col].map(freq)

    # 编码分类特征
    for col in val_df.select_dtypes(include=['object']).columns:
        val_df[col] = LabelEncoder().fit_transform(val_df[col].astype(str))

    # 处理无穷大和极端值
    numerical_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        # 替换无穷大值
        val_df[col] = val_df[col].replace([np.inf, -np.inf], np.nan)
        # 使用中位数填充NaN值
        val_df[col] = val_df[col].fillna(val_df[col].median() if not val_df[col].isna().all() else -1)

    # 获取scaler的特征名称
    scaler_features = scaler.feature_names_in_
    
    # 检查缺失的特征并添加
    missing_features = [col for col in scaler_features if col not in val_df.columns]
    print(f"缺失的scaler特征: {missing_features}")
    
    # 为缺失的特征创建列并填充默认值-1
    for col in missing_features:
        val_df[col] = -1
    
    # 确保特征顺序与训练时一致
    val_df_scaled = val_df[scaler_features].copy()
    
    # 标准化数值特征
    val_df_scaled = scaler.transform(val_df_scaled)
    
    # 将标准化后的值放回DataFrame
    val_df[scaler_features] = val_df_scaled

    # 加载最终选择的特征列表
    features_path = os.path.join('./models', "selectedfeatures.pkl")
    final_features = joblib.load(features_path)
    
    # 确保只选择存在的特征
    final_features_exist = [col for col in final_features if col in val_df.columns]
    print(f"缺失的特征: {set(final_features) - set(final_features_exist)}")
    
    # 选择最终特征
    X_val = val_df[final_features_exist]
    
    # 检查是否有缺失的特征，如果有，则添加缺失的特征并填充为-1
    for col in final_features:
        if col not in val_df.columns:
            X_val[col] = -1

    return X_val, transaction_ids


# 主函数
if __name__ == "__main__":
    # 定义路径
    MODEL_DIR = "models"
    VAL_TRANSACTION_PATH = "../data/test_transaction.csv" # 假设验证数据在 test_transaction.csv
    VAL_IDENTITY_PATH = "../data/test_identity.csv"   # 假设验证数据在 test_identity.csv
    OUTPUT_PATH = "res.csv"

    # 加载模型和特征信息
    base_models, meta_models, scaler, feature_info = load_models(MODEL_DIR)

    if base_models and meta_models and scaler and feature_info:
        # 加载并预处理验证数据
        X_val, transaction_ids = preprocess_validation_data(
            VAL_TRANSACTION_PATH, 
            VAL_IDENTITY_PATH, 
            scaler,
            feature_info
        )
        
        if X_val is not None:
            # 使用模型进行预测
            print("\n开始预测...")
            y_prob = predict(X_val, base_models, meta_models)
            
            # 保存预测结果
            print(f"\n保存预测结果到 {OUTPUT_PATH}")
            output_df = pd.DataFrame({
                'TransactionID': transaction_ids,
                'isFraud': y_prob
            })
            output_df.to_csv(OUTPUT_PATH, index=False)
            print("预测完成！")
        else:
            print("预处理验证数据失败")
    else:
        print("加载模型失败")