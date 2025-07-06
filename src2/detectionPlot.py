#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2025-07-06 14:42:55
@Author  :   Chen
@File    :   src2\DetectionPlot.py
@Software:   VSCode
@Description:
    可视化
"""

from typing import List
import pandas as pd
import matplotlib.pyplot as plt

def compareTestDetection(testDF: pd.DataFrame, testMid: int, detectionDF: pd.DataFrame, detectionMid: int, columns: List[str], index: str) -> None:
    """
    比较测试与检测数据

    Args:
        testDF (pd.DataFrame): 测试数据
        testMid (int): 测试中点时间戳
        detectionDF (pd.DataFrame): 检测数据
        detectionMid (int): 检测中点时间戳
        columns (List[str]): 需要显示的列的名称列表
        index (str): 对应的下标名称
    """

    plt.figure(figsize=(8, 4))

    # 测试子图
    plt.subplot(1, 2, 1)
    for column in columns:
        plt.plot(testDF[index], testDF[column], label= column, marker= "o")
    plt.axvline(x= testMid, color= "red", linestyle= "--", label= "testMid")
    plt.title("test")
    plt.legend()

    # 检测子图
    plt.subplot(1, 2, 2)
    for column in columns:
        plt.plot(detectionDF[index], detectionDF[column], label= column, marker= "o")
    plt.axvline(x= detectionMid, color= "red", linestyle= "--", label= "detectionMid")
    plt.title("detection")
    plt.legend()

    plt.show()