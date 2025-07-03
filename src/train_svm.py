# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

def load_data(csv_path):
    """读取 CSV 文件，返回 DataFrame"""
    print(f"🚀 正在加载数据文件：{csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取 CSV 文件，样本数量：{df.shape[0]}")
        return df
    except Exception as e:
        print(f"❌ 读取 CSV 文件 {csv_path} 出错：{e}")
        return None

def prepare_data(df, feature_columns, label_column="actionType"):
    """
    根据指定的特征列和标签列，将 DataFrame 分离为特征矩阵 X 和标签向量 y。
    """
    print("🚀 正在准备数据（分离特征和标签）...")
    if label_column not in df.columns:
        print(f"❌ 标签列 {label_column} 不存在！")
        return None, None
    try:
        X = df[feature_columns].values
        y = df[label_column].values
        print(f"✅ 特征数据维度：{X.shape}，标签数量：{len(y)}")
        return X, y
    except Exception as e:
        print(f"❌ 数据分离出错：{e}")
        return None, None

def train_svm(X_train, y_train):
    """
    使用 SVC（支持向量机）训练模型，返回训练好的模型。
    """
    print("🚀 开始训练 SVM 模型...")
    try:
        svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        svm.fit(X_train, y_train)
        print("✅ SVM 模型训练完成")
        return svm
    except Exception as e:
        print(f"❌ SVM 模型训练失败：{e}")
        return None

from sklearn.metrics import classification_report

def evaluate_model(model, X, y, data_type="测试集"):
    """
    使用给定数据集对模型进行评估，并输出准确率和分类报告。
    """
    print(f"🚀 开始评估{data_type}...")
    try:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, zero_division=0)  # 设置 zero_division=0 来处理无样本的情况
        print(f"✅ {data_type}准确率：{acc:.4f}")
        print("✅ 分类报告：")
        print(report)
    except Exception as e:
        print(f"❌ 模型评估出错：{e}")


def save_model(model, model_path):
    """
    将训练好的模型保存到指定文件。
    """
    print(f"🚀 正在保存模型到：{model_path}")
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("✅ 模型保存成功")
    except Exception as e:
        print(f"❌ 模型保存失败：{e}")

if __name__ == "__main__":
    # 设定训练集、验证集和测试集 CSV 文件路径
    TRAIN_DATA_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_train_fft_normalized.csv"
    VERIFY_DATA_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_verify_fft_normalized.csv"
    TEST_DATA_CSV  = "D:/vscode_work/badminton_classification/data/processed/processed_test_fft_normalized.csv"
    
    # 设置模型保存路径
    MODEL_SAVE_PATH = "D:/vscode_work/badminton_classification/models/svm_model.pkl"
    
    # 定义需要使用的特征列（与归一化后的 CSV 文件中的标题一致）
    feature_columns = [
        "Ax_fft_mean", "Ax_fft_std", "Ax_fft_max", "Ax_dom_bin",
        "Ay_fft_mean", "Ay_fft_std", "Ay_fft_max", "Ay_dom_bin",
        "Az_fft_mean", "Az_fft_std", "Az_fft_max", "Az_dom_bin",
        "angularSpeedX_fft_mean", "angularSpeedX_fft_std", "angularSpeedX_fft_max", "angularSpeedX_dom_bin",
        "angularSpeedY_fft_mean", "angularSpeedY_fft_std", "angularSpeedY_fft_max", "angularSpeedY_dom_bin",
        "angularSpeedZ_fft_mean", "angularSpeedZ_fft_std", "angularSpeedZ_fft_max", "angularSpeedZ_dom_bin"
    ]
    
    # 加载训练集、验证集和测试集数据
    print("🚀 加载训练集数据...")
    df_train = load_data(TRAIN_DATA_CSV)
    print("🚀 加载验证集数据...")
    df_verify = load_data(VERIFY_DATA_CSV)
    print("🚀 加载测试集数据...")
    df_test = load_data(TEST_DATA_CSV)
    
    if df_train is None or df_verify is None or df_test is None:
        exit(1)
    
    # 分离特征和标签
    X_train, y_train = prepare_data(df_train, feature_columns, label_column="actionType")
    X_verify, y_verify = prepare_data(df_verify, feature_columns, label_column="actionType")
    X_test, y_test = prepare_data(df_test, feature_columns, label_column="actionType")
    
    if X_train is None or y_train is None or X_verify is None or y_verify is None or X_test is None or y_test is None:
        exit(1)
    
    # 训练 SVM 模型
    svm_model = train_svm(X_train, y_train)
    if svm_model is None:
        exit(1)
    
    # 评估数据集、验证集和测试集
    evaluate_model(svm_model, X_train, y_train, data_type="训练集")
    evaluate_model(svm_model, X_verify, y_verify, data_type="验证集")
    evaluate_model(svm_model, X_test, y_test, data_type="测试集")
    
    # 保存模型
    save_model(svm_model, MODEL_SAVE_PATH)
