import os
import pandas as pd

# 应该保存文件名称
def merge_sensor_pair(acc_file, gyro_file, output_file):
    """
    对一对加速度和角速度 CSV 文件进行处理和合并，并将合并后的数据保存到 output_file。
    """
    try:
        # 读取加速度和角速度数据
        df_acc = pd.read_csv(acc_file)
        df_gyro = pd.read_csv(gyro_file)
    except Exception as e:
        print(f"❌ 读取文件出错：{e}")
        return None

    # 检查是否存在 'sensorTimestamp' 列
    if "sensorTimestamp" not in df_acc.columns:
        print(f"❌ 加速度文件 {acc_file} 缺少 'sensorTimestamp' 列")
        return None
    if "sensorTimestamp" not in df_gyro.columns:
        print(f"❌ 角速度文件 {gyro_file} 缺少 'sensorTimestamp' 列")
        return None

    # 按 'sensorTimestamp' 排序（merge_asof 要求按合并键排序）
    df_acc = df_acc.sort_values("sensorTimestamp")
    df_gyro = df_gyro.sort_values("sensorTimestamp")

    # 使用 merge_asof 根据 'sensorTimestamp' 进行近似合并
    try:
        df_merged = pd.merge_asof(df_acc, df_gyro, on="sensorTimestamp", direction="nearest", suffixes=("_acc", "_gyro"))
    except Exception as e:
        print(f"❌ 合并文件时出错：{e}")
        return None

    # 将合并后的数据保存到 output_file
    try:
        # df_merged.to_csv(output_file, index=False)
        print(f"✅ 合并后的数据已保存至：{output_file}")
    except Exception as e:
        print(f"❌ 保存 CSV 文件 {output_file} 出错：{e}")
        return None

    return df_merged

def process_all_sensor_files(acc_folder, gyro_folder, output_folder):
    """
    遍历加速度和角速度文件夹中的 CSV 文件，对每对文件进行合并，
    并将每对文件的合并结果保存为一个新的文件到 output_folder。
    """
    acc_files = sorted([f for f in os.listdir(acc_folder) if f.endswith(".csv")])
    gyro_files = sorted([f for f in os.listdir(gyro_folder) if f.endswith(".csv")])

    if not acc_files:
        print(f"❌ 在 {acc_folder} 中没有找到 CSV 文件！")
        return
    if not gyro_files:
        print(f"❌ 在 {gyro_folder} 中没有找到 CSV 文件！")
        return

    if len(acc_files) != len(gyro_files):
        print("⚠ 警告：加速度文件数量与角速度文件数量不匹配，按较少的数量处理。")

    file_count = min(len(acc_files), len(gyro_files))
    for i in range(file_count):
        if acc_files[i][ : acc_files[i].rfind("_")] != gyro_files[i][ : gyro_files[i].rfind("_")]:
            print(f"{acc_files[i][ : acc_files[i].rfind('_')]} 名称不匹配")
            exit()
        acc_path = os.path.join(acc_folder, acc_files[i])
        gyro_path = os.path.join(gyro_folder, gyro_files[i])
        output_file = os.path.join(output_folder, f"{acc_files[i][ : acc_files[i].rfind('_')]}_AAG.csv")  # 给每个合并文件命名为 merged_1.csv, merged_2.csv ...
        print(f"🚀 正在处理文件对：{acc_files[i]} 与 {gyro_files[i]}")
        merge_sensor_pair(acc_path, gyro_path, output_file)

if __name__ == "__main__":
    # 修改下面的路径为你的实际文件夹路径
    ACC_FOLDER = r"..\data\raw\accelerometer"
    GYRO_FOLDER = r"..\data\raw\gyroscope"
    OUTPUT_FOLDER = r"..\data\processed_fir\merged_files_2"

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 处理所有加速度和角速度文件
    process_all_sensor_files(ACC_FOLDER, GYRO_FOLDER, OUTPUT_FOLDER)
