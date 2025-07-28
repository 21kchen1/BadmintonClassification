#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2025-07-06 15:04:06
@Author  :   Chen
@File    :   Util\DataFrame.py
@Software:   VSCode
@Description:
    pandas dataframe 的辅助函数
"""

from typing import Dict
import pandas as pd

def getSubDataFrameDict(dataFrameDict: Dict[str, pd.DataFrame], startT: int, endT: int) -> Dict[str, pd.DataFrame]:
    """
    获取子数据框架字典

    Args:
        dataFrameDict (Dict[str, pd.DataFrame]): 数据框架字典
        startT (int): 开始数据
        endT (int): 结束时间

    Returns:
        Dict[str, pd.DataFrame]: 子时间框架字典
    """
    subDataFrameDict = {}
    for typeName, dataFrame in dataFrameDict.items():
        subDataFrameDict[typeName] = dataFrame[(dataFrame["unixTimestamp"] >= startT) & (dataFrame["unixTimestamp"] <= endT)]

    return subDataFrameDict