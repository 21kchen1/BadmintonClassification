#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2025-07-06 11:33:57
@Author  :   Chen
@File    :   src2\DetectionF.py
@Software:   VSCode
@Description:
    击球点检测算法
"""

import pandas as pd

class DetectionF:
    """
    检测函数。
    """
    def __init__(self, stand: str, threshold: int= 27, windowSize: int= 2000, isABS: bool= False, *params) -> None:
        """
        初始化

        Args:
            stand (str): 检测标准
            threshold (int, optional): 检测阈值. Defaults to 27.
            windowSize (int, optional): 检测窗口. Defaults to 2000.
            isABS (bool, optional): 是否绝对值. Defaults to False.
            *params (any): 其余参数
        """
        self._stand = stand
        self._threshold = threshold
        self._windowSize = windowSize
        self._isABS = isABS

    def check(self, df: pd.DataFrame) -> int:
        """
        检测

        Args:
            df (pd.DataFrame): 需要检测的数据

        Returns:
            int: 检测结果
        """
        return -1

class WindowPeak(DetectionF):
    """
    时域窗口+阈值检测击球波峰
    """
    def __init__(self, stand: str, threshold: int= 27, windowSize: int= 2000, isABS: bool= False) -> None:
        """
        初始化

        Args:
            stand (str): 检测标准
            threshold (int, optional): 检测阈值. Defaults to 27.
            windowSize (int, optional): 检测窗口. Defaults to 2000.
            isABS (bool, optional): 是否绝对值. Defaults to False.
        """
        super().__init__(stand, threshold, windowSize, isABS)

    def check(self, df: pd.DataFrame) -> int:
        """
        使用时域窗口+阈值检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            df (pd.DataFrame): 数据

        Returns:
            int: 检测到的中值时间戳
        """
        # 获取数据的总长度
        total_length = len(df)
        # 每次跳过窗口，避免重叠
        i = 0
        while i < total_length:
            current_timestamp = df.iloc[i]['unixTimestamp_acc']

            start_time = current_timestamp
            end_time = current_timestamp + self._windowSize

            # 过滤当前窗口内的数据
            window_data = df[(df['unixTimestamp_acc'] >= start_time) & (df['unixTimestamp_acc'] <= end_time)]

            if len(window_data) == 0:
                i += 1
                continue

            # 选择作为波峰检测的信号
            gx_values = window_data[self._stand]

            # 是否绝对值
            if self._isABS: gx_values = gx_values.abs()
            # 检查是否有值超过阈值
            if gx_values.max() >= self._threshold:
                return int(window_data.loc[gx_values.idxmax()]['unixTimestamp_acc'])

            # 跳过当前窗口（窗口之间不重叠）
            i += len(window_data)
        return -1
