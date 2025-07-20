# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize_datasets(train_csv, verify_csv, test_csv,
                       train_out_csv, verify_out_csv, test_out_csv,
                       feature_columns):
    """
    读取训练集、验证集和测试集的 CSV 文件，对指定的特征列进行标准化（Z-score标准化）。
    注意：归一化时必须使用训练集的均值和标准差对验证集和测试集进行转换，
          这样确保训练、验证和测试数据在相同的尺度下进行模型训练和评估。

    参数：
      train_csv: 训练集输入 CSV 文件路径（如 processed_train_fft.csv）
      verify_csv: 验证集输入 CSV 文件路径（如 processed_verify_fft.csv）
      test_csv: 测试集输入 CSV 文件路径（如 processed_test_fft.csv）
      train_out_csv: 归一化后训练集输出 CSV 文件路径
      verify_out_csv: 归一化后验证集输出 CSV 文件路径
      test_out_csv: 归一化后测试集输出 CSV 文件路径
      feature_columns: 需要归一化的特征列列表

    输出：
      将归一化后的训练集、验证集和测试集分别保存到指定 CSV 文件中，同时在终端输出提示信息。
    """
    print("🚀 开始读取训练集数据...")
    try:
        df_train = pd.read_csv(train_csv)
        print(f"✅ 训练集样本数：{df_train.shape[0]}")
    except Exception as e:
        print(f"❌ 读取训练集 CSV 文件 {train_csv} 出错：{e}")
        return

    print("🚀 开始读取验证集数据...")
    try:
        df_verify = pd.read_csv(verify_csv)
        print(f"✅ 验证集样本数：{df_verify.shape[0]}")
    except Exception as e:
        print(f"❌ 读取验证集 CSV 文件 {verify_csv} 出错：{e}")
        return

    print("🚀 开始读取测试集数据...")
    try:
        df_test = pd.read_csv(test_csv)
        print(f"✅ 测试集样本数：{df_test.shape[0]}")
    except Exception as e:
        print(f"❌ 读取测试集 CSV 文件 {test_csv} 出错：{e}")
        return

    # 检查特征列是否存在
    for df_name, df in [("训练集", df_train), ("验证集", df_verify), ("测试集", df_test)]:
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            print(f"❌ {df_name} 缺少以下特征列：{missing}")
            return

    # 创建 StandardScaler 并用训练集拟合
    scaler = StandardScaler()
    try:
        X_train = df_train[feature_columns].values
        scaler.fit(X_train)
        print("✅ 训练集归一化参数（均值和标准差）已计算")
    except Exception as e:
        print(f"❌ 训练集归一化拟合出错：{e}")
        return

    # 分别对训练集、验证集和测试集进行转换
    try:
        X_train_norm = scaler.transform(X_train)
        X_verify_norm = scaler.transform(df_verify[feature_columns].values)
        X_test_norm = scaler.transform(df_test[feature_columns].values)
        print("✅ 训练集、验证集和测试集均已进行标准化转换")
    except Exception as e:
        print(f"❌ 数据转换出错：{e}")
        return

    # 构造归一化后的 DataFrame
    df_train_norm = pd.DataFrame(X_train_norm, columns=feature_columns)
    df_verify_norm = pd.DataFrame(X_verify_norm, columns=feature_columns)
    df_test_norm = pd.DataFrame(X_test_norm, columns=feature_columns)

    # 如果原数据中有动作标签 actionType，则加上
    if "actionType" in df_train.columns:
        df_train_norm["actionType"] = df_train["actionType"]
    if "actionType" in df_verify.columns:
        df_verify_norm["actionType"] = df_verify["actionType"]
    if "actionType" in df_test.columns:
        df_test_norm["actionType"] = df_test["actionType"]

    # 保存归一化后的数据到 CSV 文件
    try:
        df_train_norm.to_csv(train_out_csv, index=False)
        print(f"✅ 归一化后的训练集数据已保存至：{train_out_csv}")
    except Exception as e:
        print(f"❌ 保存训练集 CSV 文件出错：{e}")

    try:
        df_verify_norm.to_csv(verify_out_csv, index=False)
        print(f"✅ 归一化后的验证集数据已保存至：{verify_out_csv}")
    except Exception as e:
        print(f"❌ 保存验证集 CSV 文件出错：{e}")

    try:
        df_test_norm.to_csv(test_out_csv, index=False)
        print(f"✅ 归一化后的测试集数据已保存至：{test_out_csv}")
    except Exception as e:
        print(f"❌ 保存测试集 CSV 文件出错：{e}")

if __name__ == "__main__":
    # 输入路径
    TRAIN_INPUT_CSV = r"..\Result\Data\Test_FFT.csv"
    VERIFY_INPUT_CSV = r"..\Result\Data\Test_FFT.csv"
    TEST_INPUT_CSV  = r"..\Result\Data\Test_FFT.csv"
    # TRAIN_INPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_train_fft.csv"
    # VERIFY_INPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_verify_fft.csv"
    # TEST_INPUT_CSV  = "D:/vscode_work/badminton_classification/data/processed/processed_test_fft.csv"

    # 输出路径
    TRAIN_OUTPUT_CSV = r"..\Result\Data\Test_Normalize.csv"
    # TRAIN_OUTPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_train_fft_normalized.csv"
    # VERIFY_OUTPUT_CSV = "D:/vscode_work/badminton_classification/data/processed/processed_verify_fft_normalized.csv"
    # TEST_OUTPUT_CSV  = "D:/vscode_work/badminton_classification/data/processed/processed_test_fft_normalized.csv"

    # 特征列
    feature_columns = [
        "Ax_fft_mean", "Ax_fft_std", "Ax_fft_max", "Ax_dom_bin",
        "Ay_fft_mean", "Ay_fft_std", "Ay_fft_max", "Ay_dom_bin",
        "Az_fft_mean", "Az_fft_std", "Az_fft_max", "Az_dom_bin",
        "angularSpeedX_fft_mean", "angularSpeedX_fft_std", "angularSpeedX_fft_max", "angularSpeedX_dom_bin",
        "angularSpeedY_fft_mean", "angularSpeedY_fft_std", "angularSpeedY_fft_max", "angularSpeedY_dom_bin",
        "angularSpeedZ_fft_mean", "angularSpeedZ_fft_std", "angularSpeedZ_fft_max", "angularSpeedZ_dom_bin"
    ]

    print("🚀 开始对训练集、验证集和测试集数据进行标准化归一化...")
    normalize_datasets(TRAIN_INPUT_CSV, VERIFY_INPUT_CSV, TEST_INPUT_CSV,
                       TRAIN_OUTPUT_CSV, None, None,
                       feature_columns)
