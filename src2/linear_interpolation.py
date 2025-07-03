import pandas as pd
import numpy as np

def interpolate_data(input_csv, output_csv):
    """
    对合并后的传感器数据进行线性插值处理，处理零值并保存到新的CSV文件。
    
    参数：
    input_csv: 输入的CSV文件路径
    output_csv: 输出的CSV文件路径
    """
    try:
        # 读取合并后的CSV文件
        df = pd.read_csv(input_csv)
        print(f"✅ 成功读取 CSV 文件，样本总数：{df.shape[0]}")
    except Exception as e:
        print(f"❌ 读取 CSV 文件 {input_csv} 出错：{e}")
        return

    # 将零值替换为 NaN 以便进行插值
    df.replace(0, np.nan, inplace=True)

    # 对所有的数值列进行线性插值
    numeric_columns = ['Gx', 'Gy', 'Gz', 'angularSpeedX', 'angularSpeedY', 'angularSpeedZ']
    
    for col in numeric_columns:
        # 进行线性插值
        df[col] = df[col].interpolate(method='linear')
    
    # 保存处理后的数据到新的CSV文件
    try:
        df.to_csv(output_csv, index=False)
        print(f"✅ 线性插值处理完成，数据已保存至：{output_csv}")
    except Exception as e:
        print(f"❌ 保存 CSV 文件 {output_csv} 出错：{e}")

if __name__ == "__main__":
    # 设置输入和输出文件路径
    INPUT_CSV = "D:/vscode_work/badminton_classification/data/processed_fir/merged_sensor_data.csv"
    OUTPUT_CSV = "D:/vscode_work/badminton_classification/data/processed_fir/processed_merged_data_interpolated.csv"
    
    # 调用插值函数
    interpolate_data(INPUT_CSV, OUTPUT_CSV)
