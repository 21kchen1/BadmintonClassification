#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2025-07-06 11:33:57
@Author  :   Chen
@File    :   src2\detectionF.py
@Software:   VSCode
@Description:
    击球点检测算法
"""

import pandas as pd

def windowPeak(merged_df: pd.DataFrame, threshold: int= 27, window_size: int= 2000) -> int:
    """
    使用时域窗口+阈值检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

    Args:
        merged_df (pd.DataFrame): 包含 gx 的数据框架
        threshold (int, optional): 检测阈值. Defaults to 27.
        window_size (int, optional): 窗口大小. Defaults to 4000.

    Returns:
        int: 检测到的中值时间戳
    """
    # 获取数据的总长度
    total_length = len(merged_df)
    # 每次跳过窗口，避免重叠
    i = 0
    while i < total_length:
        current_timestamp = merged_df.iloc[i]['unixTimestamp_acc']

        start_time = current_timestamp
        end_time = current_timestamp + window_size

        # 过滤当前窗口内的数据
        window_data = merged_df[(merged_df['unixTimestamp_acc'] >= start_time) & (merged_df['unixTimestamp_acc'] <= end_time)]

        if len(window_data) == 0:
            i += 1
            continue

        # 选择作为波峰检测的信号
        gx_values = window_data['Gy']

        # 检查是否有值超过阈值
        if gx_values.max() >= threshold:
            peak_index = window_data.index.get_loc(gx_values.idxmax())
            # 返回中间时间戳
            return merged_df.iloc[peak_index]['unixTimestamp_acc']

        # 跳过当前窗口（窗口之间不重叠）
        i = window_data.index[-1] + 1
    return -1