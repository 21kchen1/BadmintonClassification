# -*- coding: utf-8 -*-
import os
import json
import pandas as pd

def process_single_json(file_path):
    """
    处理单个 JSON 文件，提取其中每个动作的加速度计和陀螺仪数据及各自时间戳，
    同时提取 label 部分中的 "actionType" 作为动作标签。
    JSON 文件格式示例（每个文件包含多个动作）：
    [
      {
         "info": { "recordName": "...", ... },
         "label": {
             "positionX": ...,
             "positionY": ...,
             "actionType": "BackhandTransition",
             "actionEval": "Normal"
         },
         "data": {
             "ACCELEROMETER": {
                 "Gx": [...],
                 "Gy": [...],
                 "Gz": [...],
                 "timestamp": [...]
             },
             "GYROSCOPE": {
                 "angularSpeedX": [...],
                 "angularSpeedY": [...],
                 "angularSpeedZ": [...],
                 "timestamp": [...]
             },
             "AUDIO": { ... }
         }
      },
      { ... },   // 其他动作记录
      ...
    ]
    只提取 "ACCELEROMETER" 与 "GYROSCOPE" 部分数据（包括各自时间戳），以及 "label" 中的 "actionType"。
    """
    actions = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 {file_path} 出错：{e}")
        return actions

    # 如果 JSON 文件最外层是列表，则遍历每个记录；否则封装成列表
    if not isinstance(data, list):
        data = [data]

    for i, record in enumerate(data):
        # 提取记录名称（来自 info 部分）
        info = record.get("info", {})
        record_name = info.get("recordName", "unknown")

        # 处理 label 部分：如果没有则用空字典
        label_info = record.get("label") or {}
        action_type = label_info.get("actionType", "unknown")

        # 获取 data 部分
        data_section = record.get("data")
        if not data_section:
            print(f"⚠ 记录 {i+1} ({record_name}) 中没有 'data' 部分，跳过该记录")
            continue

        # 提取加速度计数据（ACCELEROMETER）
        accel = data_section.get("ACCELEROMETER")
        if accel is None:
            print(f"⚠ 记录 {i+1} ({record_name}) 中缺少 ACCELEROMETER 数据，跳过该记录")
            continue
        accel_Gx = accel.get("Gx", [])
        accel_Gy = accel.get("Gy", [])
        accel_Gz = accel.get("Gz", [])
        accel_timestamp = accel.get("timestamp", [])

        # 提取陀螺仪数据（GYROSCOPE）
        gyro = data_section.get("GYROSCOPE")
        if gyro is None:
            print(f"⚠ 记录 {i+1} ({record_name}) 中缺少 GYROSCOPE 数据，跳过该记录")
            continue
        gyro_X = gyro.get("angularSpeedX", [])
        gyro_Y = gyro.get("angularSpeedY", [])
        gyro_Z = gyro.get("angularSpeedZ", [])
        gyro_timestamp = gyro.get("timestamp", [])

        # 检查必须数据是否存在（至少有时间戳和部分传感器数据）
        if not (accel_timestamp and gyro_timestamp):
            print(f"⚠ 记录 {i+1} ({record_name}) 中缺少时间戳，跳过该记录")
            continue

        # 构造一个动作记录，将列表数据以 JSON 字符串形式保存（便于写入 CSV）
        action_dict = {
            "recordName": record_name,
            "actionType": action_type,  # 新增动作类型标签
            "accel_timestamp": json.dumps(accel_timestamp, ensure_ascii=False),
            "Ax": json.dumps(accel_Gx, ensure_ascii=False),
            "Ay": json.dumps(accel_Gy, ensure_ascii=False),
            "Az": json.dumps(accel_Gz, ensure_ascii=False),
            "gyro_timestamp": json.dumps(gyro_timestamp, ensure_ascii=False),
            "angularSpeedX": json.dumps(gyro_X, ensure_ascii=False),
            "angularSpeedY": json.dumps(gyro_Y, ensure_ascii=False),
            "angularSpeedZ": json.dumps(gyro_Z, ensure_ascii=False)
        }
        actions.append(action_dict)
        print(f"✅ 记录 {i+1} ({record_name}) 处理完成")
    print(f"✅ 文件 {os.path.basename(file_path)} 处理完成，共提取 {len(actions)} 个动作")
    return actions

def process_directory(data_dir):
    """
    遍历指定目录下的所有 JSON 文件，调用 process_single_json() 处理每个文件，
    并将所有动作记录合并成一个 DataFrame 返回。
    """
    all_actions = []
    if not os.path.exists(data_dir):
        print(f"❌ 目录 {data_dir} 不存在！")
        return pd.DataFrame()

    json_files = [filename for filename in os.listdir(data_dir) if filename.endswith(".json")]
    if not json_files:
        print(f"❌ 在目录 {data_dir} 中没有找到 JSON 文件！")
        return pd.DataFrame()

    for filename in json_files:
        file_path = os.path.join(data_dir, filename)
        print(f"📌 读取文件: {filename}")
        actions = process_single_json(file_path)
        all_actions.extend(actions)
    df = pd.DataFrame(all_actions)
    print(f"✅ 从目录 {data_dir} 中共提取 {df.shape[0]} 条动作记录")
    return df

def preprocess_and_save(train_dir, test_dir, train_save_path, test_save_path):
    """
    分别处理训练集和测试集目录下的所有 JSON 文件，并保存为两个 CSV 文件。
    """
    print("🚀 开始处理训练集数据...")
    df_train = process_directory(train_dir)
    if not df_train.empty:
        df_train.to_csv(train_save_path, index=False)
        print(f"✅ 训练集数据已保存至: {train_save_path}")
    else:
        print("❌ 训练集数据为空，请检查数据目录！")

    print("🚀 开始处理测试集数据...")
    df_test = process_directory(test_dir)
    if not df_test.empty:
        df_test.to_csv(test_save_path, index=False)
        print(f"✅ 测试集数据已保存至: {test_save_path}")
    else:
        print("❌ 测试集数据为空，请检查数据目录！")

if __name__ == "__main__":
    # 请确保你已经在项目根目录下运行该脚本
    # 定义训练集和测试集的原始 JSON 数据目录（绝对路径推荐）
    # TRAIN_DATA_DIR = "D:/vscode_work/badminton_classification/data/raw/train/"
    TRAIN_DATA_DIR = r"G:\Badminton\BADS_CLL_TEST"
    # TEST_DATA_DIR  = "D:/vscode_work/badminton_classification/data/raw/test/"

    # 定义处理后 CSV 数据的保存路径
    # TRAIN_SAVE_PATH = "D:/vscode_work/badminton_classification/data/processed/processed_train.csv"
    TRAIN_SAVE_PATH = r"..\Result\Data\Test_Process.csv"
    # TEST_SAVE_PATH  = "D:/vscode_work/badminton_classification/data/processed/processed_test.csv"

    # preprocess_and_save(TRAIN_DATA_DIR, TEST_DATA_DIR, TRAIN_SAVE_PATH, TEST_SAVE_PATH)
    preprocess_and_save(TRAIN_DATA_DIR, None, TRAIN_SAVE_PATH, None)
