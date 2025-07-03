# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

def load_data(csv_path):
    print(f"🚀 正在加载数据文件：{csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取 CSV 文件，样本数量：{df.shape[0]}")
        return df
    except Exception as e:
        print(f"❌ 读取 CSV 文件 {csv_path} 出错：{e}")
        return None

def prepare_data(df, feature_columns, label_column="actionType"):
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

def train_adaboost_rf(X_train, y_train):
    print("🚀 开始训练 AdaBoost（随机森林版）模型...")
    try:
        base_estimator = RandomForestClassifier(
            n_estimators=30,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        ada_rf = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=200,
            learning_rate=0.6,
            random_state=42,
            algorithm='SAMME'
        )
        ada_rf.fit(X_train, y_train)
        print("✅ AdaBoost（随机森林版）模型训练完成")
        return ada_rf
    except Exception as e:
        print(f"❌ AdaBoost（随机森林版）模型训练失败：{e}")
        return None

def evaluate_model(model, X, y, dataset_name="测试集"):
    print(f"🚀 开始评估模型在 {dataset_name} 上的表现...")
    try:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        print(f"✅ {dataset_name} 准确率：{acc:.4f}")
        print(f"✅ {dataset_name} 分类报告：\n{report}")
    except Exception as e:
        print(f"❌ 模型评估出错：{e}")

def save_model(model, model_path):
    print(f"🚀 正在保存模型到：{model_path}")
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("✅ 模型保存成功")
    except Exception as e:
        print(f"❌ 模型保存失败：{e}")

if __name__ == "__main__":
    TRAIN_DATA_CSV  = "D:/vscode_work/badminton_classification/data/processed/processed_train_fft_normalized.csv"
    VERIFY_DATA_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_verify_fft_normalized.csv"
    TEST_DATA_CSV   = "D:/vscode_work/badminton_classification/data/processed/processed_test_fft_normalized.csv"
    MODEL_SAVE_PATH = "D:/vscode_work/badminton_classification/models/adaboost_rf_model.pkl"

    feature_columns = [
        "Ax_fft_mean", "Ax_fft_std", "Ax_fft_max", "Ax_dom_bin",
        "Ay_fft_mean", "Ay_fft_std", "Ay_fft_max", "Ay_dom_bin",
        "Az_fft_mean", "Az_fft_std", "Az_fft_max", "Az_dom_bin",
        "angularSpeedX_fft_mean", "angularSpeedX_fft_std", "angularSpeedX_fft_max", "angularSpeedX_dom_bin",
        "angularSpeedY_fft_mean", "angularSpeedY_fft_std", "angularSpeedY_fft_max", "angularSpeedY_dom_bin",
        "angularSpeedZ_fft_mean", "angularSpeedZ_fft_std", "angularSpeedZ_fft_max", "angularSpeedZ_dom_bin"
    ]

    print("🚀 读取并准备训练集数据...")
    df_train = load_data(TRAIN_DATA_CSV)
    X_train, y_train = prepare_data(df_train, feature_columns)

    print("🚀 读取并准备验证集数据...")
    df_verify = load_data(VERIFY_DATA_CSV)
    X_verify, y_verify = prepare_data(df_verify, feature_columns)

    print("🚀 读取并准备测试集数据...")
    df_test = load_data(TEST_DATA_CSV)
    X_test, y_test = prepare_data(df_test, feature_columns)

    model = train_adaboost_rf(X_train, y_train)

    evaluate_model(model, X_train, y_train, dataset_name="训练集")
    evaluate_model(model, X_verify, y_verify, dataset_name="验证集")
    evaluate_model(model, X_test, y_test, dataset_name="测试集")

    save_model(model, MODEL_SAVE_PATH)
