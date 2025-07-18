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

from copy import deepcopy
from math import inf
import math
from typing import List
import pandas as pd


class DetectionF:
    """
    检测函数。
    """
    THE_TIMESTAMP = 'unixTimestamp_acc'

    class StandUnit:
        """
        标准单元
        """
        def __init__(self, stand: str, threshold: int, isABS: bool= False, bias: int= 0) -> None:
            """
            初始化

            Args:
                stand (str): 标准名称
                threshold (int): 阈值
                isABS (bool, optional): 是否绝对值. Defaults to False.
                bias (int, optional): 偏差位移. Defaults to 0.
            """
            self.stand = stand
            self.threshold = threshold
            self.isABS = isABS
            self.bias = bias

    def __init__(self, standUnits: List[StandUnit], windowSize: int= 2000, *params) -> None:
        """
        初始化

        Args:
            standUnits (List[StandUnit): 检测标准列表
            windowSize (int, optional): 检测窗口. Defaults to 2000.
            *params (any): 其余参数
        """
        self._standUnits = standUnits
        self._windowSize = windowSize

    def _judge(self, windowDF: pd.DataFrame) -> int:
        return -1

    def check(self, df: pd.DataFrame) -> int:
        """
        检测

        Args:
            df (pd.DataFrame): 需要检测的数据

        Returns:
            int: 检测结果
        """
        # 获取数据的总长度
        total_length = len(df)
        # 每次跳过窗口，避免重叠
        i = 0
        while i < total_length:
            current_timestamp = df.iloc[i][self.THE_TIMESTAMP]

            start_time = current_timestamp
            end_time = current_timestamp + self._windowSize

            # 过滤当前窗口内的数据
            window_data = df[(df[self.THE_TIMESTAMP] >= start_time) & (df[self.THE_TIMESTAMP] <= end_time)]

            if len(window_data) == 0:
                i += 1
                continue

            # 判断
            judgeResult = self._judge(window_data)
            if judgeResult != -1:
                return judgeResult

            # 跳过当前窗口（窗口之间不重叠）
            i += len(window_data)
        return -1

    def _getWindowMaxTimestampDIY(self, windowDF: pd.DataFrame) -> int:
        """
        自定义方法获取窗口数据的最大数据时间戳

        Args:
            windowDF (pd.DataFrame): 窗口数据

        Returns:
            int: 最大数据时间戳
        """
        # 第一个数据作为基准单元
        standUnit = self._standUnits[0]
        return int(windowDF.loc[(windowDF[standUnit.stand] ** 2).idxmax()][self.THE_TIMESTAMP]) #type: ignore

    def midCheck(self, df: pd.DataFrame) -> int:
        """
        区间中点检测

        Args:
            df (pd.DataFrame): 需要检测的数据

        Returns:
            int: 检测结果
        """
        halfWindowSize = self._windowSize // 2
        # 确定第一个检测点时间戳
        checkT = int(df[df[self.THE_TIMESTAMP] >= df.iloc[0][self.THE_TIMESTAMP] + halfWindowSize].iloc[0][self.THE_TIMESTAMP])
        # 遍历
        for index in range(0, len(df)):
            # 如果早于检测点则跳过
            if df.iloc[index][self.THE_TIMESTAMP] < checkT:
                continue
            # 获取窗口数据
            starT = checkT - halfWindowSize
            endT = checkT + halfWindowSize
            windowData = df[(df[self.THE_TIMESTAMP] >= starT) & (df[self.THE_TIMESTAMP] <= endT)]
            # 获取窗口最大值的时间戳
            # 这个最大值应该是可以自定义的
            windowMaxTimestamp = self._getWindowMaxTimestampDIY(windowData)
            # print((windowData[standUnit.stand] ** 2).max())
            # 如果中间时间戳数据不是最大值
            # 且晚于当前时间戳
            # print(f"windowMaxTimestamp: {windowMaxTimestamp}, checkT: {checkT}")
            if windowMaxTimestamp > checkT:
                # 更新检查时间戳 跳到下一个极值
                checkT = windowMaxTimestamp
                continue

            # 保存当前时间戳
            midT = deepcopy(checkT)
            # 获取后续数据
            afterWindow = df[df[self.THE_TIMESTAMP] >= midT + self._windowSize // 2]
            # 超过数据范围
            if len(afterWindow) <= 0:
                checkT = math.inf
            # 跳过当前窗口
            else:
                checkT = int(afterWindow.iloc[0][self.THE_TIMESTAMP])

            # 如果中间时间戳数据不是最大值
            # 且早于当前时间戳
            if windowMaxTimestamp < midT:
                # 跳过当前窗口
                continue

            # 判断中值是否符合条件
            judgeResult = self._judge(windowData)

            # 不符合条件
            if judgeResult == -1:
                continue
            # 符合条件
            return judgeResult
        return -1


class WindowPeak(DetectionF):
    """
    时域窗口+阈值检测击球波峰
    """

    def _judge(self, windowDF: pd.DataFrame) -> int:
        """
        使用时域窗口+阈值检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            windowDF (pd.DataFrame): 窗口数据

        Returns:
            int: 检测到的中值时间戳
        """
        # 选择作为波峰检测的信号
        standUnit = self._standUnits[0]
        theValues = windowDF[standUnit.stand]

        # 是否绝对值
        if standUnit.isABS: theValues = theValues.abs()
        # 检查是否有值超过阈值
        if theValues.max() >= standUnit.threshold:
            return int(windowDF.loc[theValues.idxmax()][self.THE_TIMESTAMP]) + standUnit.bias #type: ignore
        return -1

class WindowPeakS(DetectionF):
    """
    时域窗口+阈值平方检测击球波峰
    """

    def _judge(self, windowDF: pd.DataFrame) -> int:
        """
        使用时域窗口+阈值检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            windowDF (pd.DataFrame): 窗口数据

        Returns:
            int: 检测到的中值时间戳
        """
        standUnit = self._standUnits[0]
        # 选择作为波峰检测的信号 并 平方
        theValues = windowDF[standUnit.stand] ** 2

        # 检查是否有值超过阈值
        if theValues.max() >= standUnit.threshold:
            return int(windowDF.loc[theValues.idxmax()][self.THE_TIMESTAMP]) + standUnit.bias #type: ignore
        return -1

class WindowPeakMS(DetectionF):
    """
    时域窗口+多阈值平方检测击球波峰
    """

    def _judge(self, windowDF: pd.DataFrame) -> int:
        """
        使用时域窗口+多阈值检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            windowDF (pd.DataFrame): 窗口数据

        Returns:
            int: 检测到的中值时间戳
        """
        # 目标时间戳
        midTimestamp = 0
        for index, standUnit in enumerate(self._standUnits):
            # 选择作为波峰检测的信号
            theValues = windowDF[standUnit.stand]
            # 是否绝对值 平方
            if standUnit.isABS:
                theValues = theValues ** 2
            # 是否小于阈值
            if theValues.max() < standUnit.threshold:
                return -1

            # 获取检测到的时间戳
            theTimestamp = int(windowDF.loc[theValues.idxmax()][self.THE_TIMESTAMP]) #type: ignore
            # 基准时间戳
            if index == 0:
                midTimestamp = theTimestamp + standUnit.bias
                continue
            # 检测波峰距离是否过大
            if abs(midTimestamp - theTimestamp) > 2000:
                return -1
            # 计算平均值
            midTimestamp = (midTimestamp * index + theTimestamp + standUnit.bias) // (index + 1)

        return midTimestamp

class WindowPeakDiff(DetectionF):
    """
    时域窗口 + 阈值检测击球波峰 + 差分
    """

    class DiffStandUnit(DetectionF.StandUnit):
        """
        差分标准单元
        """
        def __init__(self, stand: str, threshold: int, diffThreshold: int, diffDistance: int, isABS: bool= False, bias: int= 0) -> None:
            """
            初始化

            Args:
                stand (str): 标准名称
                threshold (int): 阈值
                diffThreshold (int): 差分阈值
                diffDistance (int): 差分距离
                isABS (bool, optional): 是否绝对值. Defaults to False.
                bias (int, optional): 偏差位移. Defaults to 0.
            """
            super().__init__(stand, threshold, isABS, bias)
            self.diffThreshold = diffThreshold
            self.diffDistance = diffDistance

    def __init__(self, diffStandUnit: List[DiffStandUnit], windowSize: int= 2000) -> None:
        """
        初始化

        Args:
            diffStandUnit (List[DiffStandUnit]): 差分标准列表
            windowSize (int, optional): 检测窗口. Defaults to 2000.
        """
        super().__init__([], windowSize)
        self.diffStandUnit = diffStandUnit

    def _judge(self, windowDF: pd.DataFrame) -> int:
        """
        使用时域窗口+阈值检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            windowDF (pd.DataFrame): 窗口数据

        Returns:
            int: 检测到的中值时间戳
        """
        # 选择作为波峰检测的信号
        theValues = windowDF[self.diffStandUnit[0].stand]

        # 是否绝对值
        if self.diffStandUnit[0].isABS: theValues = theValues.abs()

        # 获取差分大小
        diffIloc = windowDF.index.get_loc(theValues.idxmax())
        # 如果差分异常，则选取最前一个来判断
        if not isinstance(diffIloc, int):
            diffIloc = self.diffStandUnit[0].diffDistance
        diffIloc -= self.diffStandUnit[0].diffDistance
        theDiff = self.diffStandUnit[0].diffThreshold
        if diffIloc >= 0:
            theDiff = abs(theValues.max() - theValues.iloc[diffIloc])
        # 检查是否有值超过阈值
        if theValues.max() >= self.diffStandUnit[0].threshold and theDiff >= self.diffStandUnit[0].diffThreshold:
            return int(windowDF.loc[theValues.idxmax()][self.THE_TIMESTAMP]) #type: ignore
        return -1

class WindowMaxEnergyPeak(DetectionF):
    """
    时域窗口+最大能量峰值检测
    """

    def _judge(self, windowDF: pd.DataFrame) -> int:
        """
        时域窗口+最大能量峰值检测，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            windowDF (pd.DataFrame): 窗口数据

        Returns:
            int: 检测到的中值时间戳
        """
        # 最大能量
        maxEnergy = 0
        # 最大频道
        maxChannel = 0
        for index, standUnit in enumerate(self._standUnits):
            # 评分计算能量
            theEnergy = (windowDF[standUnit.stand] ** 2).sum()
            if maxEnergy > theEnergy:
                continue
            maxEnergy = theEnergy
            maxChannel = index

        # 选取最大能量
        standUnit = self._standUnits[maxChannel]
        theValues = (windowDF[standUnit.stand] ** 2) if standUnit.isABS else windowDF[standUnit.stand]

        # 检查是否有值超过阈值
        if theValues.max() >= standUnit.threshold:
            return int(windowDF.loc[theValues.idxmax()][self.THE_TIMESTAMP]) + standUnit.bias #type: ignore
        return -1