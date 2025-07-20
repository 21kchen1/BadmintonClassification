# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
from scipy.fft import fft

def compute_fft_features(signal, fs=100):
    """
    对一维时域信号进行 FFT 变换，并提取频域特征。

    参数：
      signal: numpy 数组，表示时域信号
      fs: 采样率，默认100Hz
    返回：
      mean_amp: FFT 幅值均值
      std_amp: FFT 幅值标准差
      max_amp: FFT 幅值最大值
      dom_freq_bin: FFT 幅值最大值所在的频率桶索引
    """
    fft_vals = fft(signal)
    # 取前半部分幅值（FFT 结果对称）
    fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])
    mean_amp = np.mean(fft_magnitude)
    std_amp = np.std(fft_magnitude)
    max_amp = np.max(fft_magnitude)
    dom_freq_bin = int(np.argmax(fft_magnitude))
    return mean_amp, std_amp, max_amp, dom_freq_bin

def extract_fft_features_from_row(row):
    """
    针对 CSV 中单个样本（行），对每个传感器通道（Ax, Ay, Az, angularSpeedX, angularSpeedY, angularSpeedZ）
    进行 FFT 特征提取。假设这些列中存储的是 JSON 格式的字符串。

    返回：
      一个字典，键名为 "{通道}_fft_mean" 等特征名称。
    """
    features = {}
    channels = ["Ax", "Ay", "Az", "angularSpeedX", "angularSpeedY", "angularSpeedZ"]
    for col in channels:
        try:
            # 解析 JSON 字符串成列表
            signal_list = json.loads(row[col])
            signal_array = np.array(signal_list)
        except Exception as e:
            print(f"❌ 解析列 {col} 时出错，行索引 {row.name}：{e}")
            signal_array = np.array([])
        if signal_array.size == 0:
            features[f"{col}_fft_mean"] = 0
            features[f"{col}_fft_std"] = 0
            features[f"{col}_fft_max"] = 0
            features[f"{col}_dom_bin"] = 0
        else:
            try:
                mean_amp, std_amp, max_amp, dom_bin = compute_fft_features(signal_array)
            except Exception as e:
                print(f"❌ FFT 计算错误，在列 {col}，行索引 {row.name}：{e}")
                mean_amp, std_amp, max_amp, dom_bin = 0, 0, 0, 0
            features[f"{col}_fft_mean"] = mean_amp
            features[f"{col}_fft_std"] = std_amp
            features[f"{col}_fft_max"] = max_amp
            features[f"{col}_dom_bin"] = dom_bin
    return features

def process_fft_features(input_csv, output_csv):
    print(f"🚀 开始处理文件：{input_csv}")
    try:
        df = pd.read_csv(input_csv)
        print(f"✅ 成功读取 CSV 文件，样本总数：{df.shape[0]}")
    except Exception as e:
        print(f"❌ 读取 CSV 文件 {input_csv} 出错：{e}")
        return

    fft_features_list = []
    for idx, row in df.iterrows():
        try:
            features = extract_fft_features_from_row(row)
            fft_features_list.append(features)
            if idx < 5:
                print(f"样本 {idx} 的 FFT 特征：{features}")
        except Exception as e:
            print(f"❌ 处理第 {idx} 行时出错：{e}")

    df_fft = pd.DataFrame(fft_features_list)
    # 如果原数据中有动作标签 actionType，则加上
    if "actionType" in df.columns:
        df_fft["actionType"] = df["actionType"]

    try:
        df_fft.to_csv(output_csv, index=False)
        print(f"✅ 频域特征数据已成功保存至：{output_csv}")
    except Exception as e:
        print(f"❌ 保存 CSV 文件 {output_csv} 出错：{e}")

if __name__ == "__main__":
    TRAIN_INPUT_CSV = r"..\Result\Data\Test_Process.csv"
    # TRAIN_INPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_train.csv"
    # VERIFY_INPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_verify.csv"
    # TEST_INPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_test.csv"

    TRAIN_OUTPUT_CSV = r"..\Result\Data\Test_FFT.csv"
    # TRAIN_OUTPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_train_fft.csv"
    # VERIFY_OUTPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_verify_fft.csv"
    # TEST_OUTPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_test_fft.csv"

    print("🚀 开始处理训练集数据...")
    process_fft_features(TRAIN_INPUT_CSV, TRAIN_OUTPUT_CSV)

    # print("🚀 开始处理验证集数据...")
    # process_fft_features(VERIFY_INPUT_CSV, VERIFY_OUTPUT_CSV)

    # print("🚀 开始处理测试集数据...")
    # process_fft_features(TEST_INPUT_CSV, TEST_OUTPUT_CSV)
