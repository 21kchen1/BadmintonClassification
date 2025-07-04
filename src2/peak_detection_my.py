import os
import pandas as pd

def process_peaks_and_save(merged_df: pd.DataFrame, threshold=27, window_size=10000, output_folder="output") -> int:
    """
    使用时域窗口+阈值检测波峰，并保存达到条件的击球数据前后一秒的时域数据。

    参数：
        merged_df: 合并后的传感器数据 DataFrame
        threshold: 用于波峰检测的阈值
        window_size: 时域窗口大小（单位：毫秒，10秒=10000毫秒）
        output_folder: 保存提取数据的文件夹

    返回：
        无
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取数据的总长度
    total_length = len(merged_df)
    # 检测到的击球数量
    total_peak = 0
    # 每次跳过窗口，避免重叠
    i = 0
    while i < total_length:
        current_timestamp = merged_df.iloc[i]['unixTimestamp_acc']

        # 定义时域窗口：每个窗口为10秒（10000毫秒）
        start_time = current_timestamp
        end_time = current_timestamp + window_size

        # 过滤当前窗口内的数据
        window_data = merged_df[(merged_df['unixTimestamp_acc'] >= start_time) & (merged_df['unixTimestamp_acc'] <= end_time)]

        if len(window_data) == 0:
            i += 1
            continue

        # 选择Gx作为波峰检测的信号
        gx_values = window_data['Gx']  # Gx的数值直接使用

        # 选择中心位置
        mid_index = len(window_data) // 2
        mid_value = gx_values.iloc[mid_index]

        # 中间位置值是否为最大值，同时中间位置值大于阈值
        if mid_value == gx_values.max() and mid_value > threshold:
            peak_value = gx_values.max()
            peak_index = gx_values.idxmax()

            # 获取前后一秒的数据
            start_time = merged_df.iloc[peak_index]['unixTimestamp_acc'] - 1000  # 前1秒
            end_time = merged_df.iloc[peak_index]['unixTimestamp_acc'] + 1000  # 后1秒
            segment_data = merged_df[(merged_df['unixTimestamp_acc'] >= start_time) & (merged_df['unixTimestamp_acc'] <= end_time)]

            # 保存数据
            output_file = os.path.join(output_folder, f"peak_data_{start_time}_{end_time}.csv")
            # segment_data.to_csv(output_file, index=False)
            print(f"✅ 保存数据：{output_file}")
            total_peak += 1
            # 跳过当前窗口，不重复捕获击球点
            i = window_data.index[-1]
        # 窗口前移
        i += 1
    return total_peak

def process_multiple_files(input_folder, output_folder, threshold=27, window_size=10000):
    """
    读取文件夹中的所有合并数据文件，进行波峰检测并保存前后一秒的击球数据。

    参数：
        input_folder: 包含多个文件的文件夹路径
        output_folder: 保存提取数据的文件夹路径
        threshold: 用于波峰检测的阈值
        window_size: 时域窗口大小（单位：毫秒）

    返回：
        无
    """
    # 获取文件夹中的所有文件
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".csv")])

    if not files:
        print(f"❌ 在 {input_folder} 中没有找到 CSV 文件！")
        return

    all_peak = 0
    try:
        for file in files:
            input_file_path = os.path.join(input_folder, file)
            print(f"🚀 正在处理文件：{file}")
            # 读取合并的文件
            merged_df = pd.read_csv(input_file_path)
            # 调用波峰检测并保存结果
            all_peak += process_peaks_and_save(merged_df, threshold, window_size, output_folder)
    except Exception as e:
        print(f"❌ 处理文件 {file} 时出错：{e}")
        print("⚠ 出现错误，停止程序。开始训练模型...")
    finally:
        print(f"all_peak: {all_peak}")

# 主程序入口
if __name__ == "__main__":
    # 设置输入文件夹和输出文件夹路径
    INPUT_FOLDER = r"..\data\processed_fir\merged_files"
    OUTPUT_FOLDER = r"..\data\processed_fir\peaks2"

    # 处理所有文件并执行波峰检测
    process_multiple_files(INPUT_FOLDER, OUTPUT_FOLDER, threshold=27, window_size=2000)
    print("✅ 波峰检测完成，开始训练模型...")
    # 在此处可以插入调用训练模型的代码
    print("✅ 训练完成！")
