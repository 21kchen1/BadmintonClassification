import pandas as pd
import numpy as np
import os

# 更新文件路径
file_path = 'D:/vscode_work/badminton_classification/data/processed/processed_all_train_data.csv'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"❌ 文件未找到，请检查路径: {file_path}")
else:
    data = pd.read_csv(file_path)
    print("✅ 文件加载成功！")

    # 打印所有列名，以确认列名是否正确
    print("数据列名:", data.columns)

    # 设置阈值
    threshold = 27

    # 用于保存有效样本数量
    valid_samples_count = 0

    # 假设' Ax '为列名，若存在额外的空格或其他字符，请根据实际情况修改
    for idx, row in data.iterrows():
        # 检查实际列名，并解析Ax数据（假设Ax列为字符串形式的数组）
        ax_values = eval(row['Ax'])
        
        # 检测Ax数据中的最大值是否超过阈值
        if max(abs(np.array(ax_values))) >= threshold:
            valid_samples_count += 1

    # 输出有效样本数量
    print(f"总共提取到的有效样本数量: {valid_samples_count}")
