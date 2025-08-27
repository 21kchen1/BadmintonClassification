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

import sys

from src2.testDetectF import DetectF, SlideWindowF

sys.path.append("..\\")
from typing import Dict, List
import pandas as pd
from src2 import DetectionF
import matplotlib.pyplot as plt

def compareTestDetection(testDFD: Dict[str, pd.DataFrame], testMid: int, detectionDFD: Dict[str, pd.DataFrame], detectionMid: int, columns: List[DetectionF.DetectionF.StandUnit], index: str) -> None:
    """
    比较测试与检测数据

    Args:
        testDFD (Dict[str, pd.DataFrame]): 测试数据
        testMid (int): 测试中点时间戳
        detectionDFD (Dict[str, pd.DataFrame]): 检测数据
        detectionMid (int): 检测中点时间戳
        columns (List[DetectionF.DetectionF.StandUnit]): 需要显示的列的数据单元
        index (str): 对应的下标名称
    """

    plt.figure(figsize=(8, 4))

    # 测试子图
    plt.subplot(1, 2, 1)
    for column in columns:
        plt.plot(testDFD[column.typeName][index], testDFD[column.typeName][column.stand], label= f"{column.typeName} {column.stand}", marker= "o")
    plt.axvline(x= testMid, color= "red", linestyle= "--", label= "testMid")
    plt.title("test")
    plt.legend()

    # 检测子图
    plt.subplot(1, 2, 2)
    for column in columns:
        plt.plot(detectionDFD[column.typeName][index], detectionDFD[column.typeName][column.stand], label= f"{column.typeName} {column.stand}", marker= "o")
    plt.axvline(x= detectionMid, color= "red", linestyle= "--", label= "detectionMid")
    plt.title("detection")
    plt.legend()

    plt.show()

def compareTestDetect(testDFD: Dict[str, pd.DataFrame], testMid: int, detectionDFD: Dict[str, pd.DataFrame], detectionMid: int, columns: List[SlideWindowF.ConfigUnit], index: str) -> None:
    """
    比较测试与检测数据

    Args:
        testDFD (Dict[str, pd.DataFrame]): 测试数据
        testMid (int): 测试中点时间戳
        detectionDFD (Dict[str, pd.DataFrame]): 检测数据
        detectionMid (int): 检测中点时间戳
        columns (List[DetectionF.DetectionF.StandUnit]): 需要显示的列的数据单元
        index (str): 对应的下标名称
    """

    plt.figure(figsize=(8, 4))

    # 测试子图
    plt.subplot(1, 2, 1)
    for column in columns:
        plt.plot(testDFD[column.typeDataName][index], testDFD[column.typeDataName][column.typeDataAttr], label= f"{column.typeDataName} {column.typeDataAttr}", marker= "o")
    plt.axvline(x= testMid, color= "red", linestyle= "--", label= "testMid")
    plt.title("test")
    plt.legend()

    # 检测子图
    plt.subplot(1, 2, 2)
    for column in columns:
        plt.plot(detectionDFD[column.typeDataName][index], detectionDFD[column.typeDataName][column.typeDataAttr], label= f"{column.typeDataName} {column.typeDataAttr}", marker= "o")
    plt.axvline(x= detectionMid, color= "red", linestyle= "--", label= "detectionMid")
    plt.title("detection")
    plt.legend()

    plt.show()