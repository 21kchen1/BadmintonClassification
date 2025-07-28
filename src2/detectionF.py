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
from typing import Dict, List
import pandas as pd

from Util.DataFrame import getSubDataFrameDict


class DetectionF:
    """
    检测函数。
    """
    THE_TIMESTAMP = 'unixTimestamp'

    class StandUnit:
        """
        标准单元
        """
        def __init__(self, typeName: str, stand: str, threshold: int, isABS: bool= False, bias: int= 0) -> None:
            """
            初始化

            Args:
                typeName (str): 数据类型名称
                stand (str): 标准名称
                threshold (int): 阈值
                isABS (bool, optional): 是否绝对值. Defaults to False.
                bias (int, optional): 偏差位移. Defaults to 0.
            """
            self.typeName = typeName
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

    def _judge(self, windowDFD: Dict[str, pd.DataFrame]) -> int:
        """
        判断函数

        Args:
            windowDFD (Dict[str, pd.DataFrame]): 数据框架字典

        Returns:
            int: 判断结果
        """
        return -1

    def check(self, dfd: Dict[str, pd.DataFrame]) -> int:
        """
        检测

        Args:
            dfd (pd.DataFrame): 需要检测的数据字典

        Returns:
            int: 检测结果
        """
        # 第一个标准作为主标准
        mdf = dfd[self._standUnits[0].typeName]
        # 获取数据的总长度
        total_length = len(mdf)
        # 每次跳过窗口，避免重叠
        i = 0
        while i < total_length:
            current_timestamp = mdf.iloc[i][self.THE_TIMESTAMP]

            start_time = current_timestamp
            end_time = current_timestamp + self._windowSize

            # 过滤当前窗口内的数据
            subDFD = getSubDataFrameDict(dfd, start_time, end_time)
            # window_data = df[(df[self.THE_TIMESTAMP] >= start_time) & (df[self.THE_TIMESTAMP] <= end_time)]

            if len(subDFD[self._standUnits[0].typeName]) == 0:
                i += 1
                continue

            # 判断
            judgeResult = self._judge(dfd)
            if judgeResult != -1:
                return judgeResult

            # 跳过当前窗口（窗口之间不重叠）
            i += len(subDFD[self._standUnits[0].typeName])
        return -1

    def _getWindowMaxTimestampDIY(self, windowDFD: Dict[str, pd.DataFrame]) -> int:
        """
        自定义方法获取窗口数据的最大数据时间戳

        Args:
            windowDFD (pd.DataFrame): 窗口数据框架字典

        Returns:
            int: 最大数据时间戳
        """
        # 第一个数据作为基准单元
        standUnit = self._standUnits[0]
        theDF = windowDFD[standUnit.typeName]
        return int(theDF.loc[(theDF[standUnit.stand] ** 2).idxmax()][self.THE_TIMESTAMP]) #type: ignore

    def midCheck(self, dfd: Dict[str, pd.DataFrame]) -> int:
        """
        区间中点检测

        Args:
            dfd (pd.DataFrame): 需要检测的数据框架字典

        Returns:
            int: 检测结果
        """
        halfWindowSize = self._windowSize // 2
        # 选择第一个标准为主标准
        mdf = dfd[self._standUnits[0].typeName]
        # 确定第一个检测点时间戳
        checkT = int(mdf[mdf[self.THE_TIMESTAMP] >= mdf.iloc[0][self.THE_TIMESTAMP] + halfWindowSize].iloc[0][self.THE_TIMESTAMP])
        # 遍历
        for index in range(0, len(mdf)):
            # 如果早于检测点则跳过
            if mdf.iloc[index][self.THE_TIMESTAMP] < checkT:
                continue
            # 获取窗口数据
            starT = int(checkT - halfWindowSize)
            endT = int(checkT + halfWindowSize)
            # windowData = df[(df[self.THE_TIMESTAMP] >= starT) & (df[self.THE_TIMESTAMP] <= endT)]
            windowDFD = getSubDataFrameDict(dfd, starT, endT)
            # 获取窗口最大值的时间戳
            windowMaxTimestamp = self._getWindowMaxTimestampDIY(windowDFD)
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
            afterWindow = mdf[mdf[self.THE_TIMESTAMP] >= midT + self._windowSize // 2]
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
            judgeResult = self._judge(windowDFD)

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

    def _judge(self, windowDFD: Dict[str, pd.DataFrame]) -> int:
        """
        使用时域窗口+阈值检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            windowDFD (pd.DataFrame): 窗口数据字典

        Returns:
            int: 检测到的中值时间戳
        """
        standUnit = self._standUnits[0]
        # 选择作为波峰检测的信号 并 平方
        theValues = windowDFD[standUnit.typeName][standUnit.stand] ** 2

        # 检查是否有值超过阈值
        if theValues.max() >= standUnit.threshold:
            return int(windowDFD[standUnit.typeName].loc[theValues.idxmax()][self.THE_TIMESTAMP]) + standUnit.bias #type: ignore
        return -1

class WindowPeakSA(DetectionF):
    """
    时域窗口+阈值平方+音频检测击球波峰
    """

    def _judge(self, windowDFD: Dict[str, pd.DataFrame]) -> int:
        """
        使用时域窗口+阈值平方+音频检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            windowDFD (pd.DataFrame): 窗口数据字典

        Returns:
            int: 检测到的中值时间戳
        """
        standUnit = self._standUnits[0]
        # 选择作为波峰检测的信号 并 平方
        theValues = windowDFD[standUnit.typeName][standUnit.stand] ** 2
        # 最大值的时间戳
        maxTimestamp = int(windowDFD[standUnit.typeName].loc[theValues.idxmax()][self.THE_TIMESTAMP]) #type: ignore

        audioValues = windowDFD["AUDIO"]["values"]
        audioMaxTimestamp = int(windowDFD["AUDIO"].loc[audioValues.idxmax()][self.THE_TIMESTAMP]) if len(windowDFD["AUDIO"]) > 0 else -math.inf #type: ignore

        # 检查是否有值超过阈值
        if theValues.max() >= standUnit.threshold and abs(audioMaxTimestamp - maxTimestamp) < 2000 and audioValues.max() >= 30:
            return maxTimestamp + standUnit.bias
        return -1


class WindowPeakMS(DetectionF):
    """
    时域窗口+多阈值平方检测击球波峰
    """

    def _judge(self, windowDFD: Dict[str, pd.DataFrame]) -> int:
        """
        使用时域窗口+多阈值检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            windowDFD (pd.DataFrame): 窗口数据框架字典

        Returns:
            int: 检测到的中值时间戳
        """
        # 目标时间戳
        midTimestamp = 0
        for index, standUnit in enumerate(self._standUnits):
            # 选择作为波峰检测的信号
            theDF = windowDFD[standUnit.typeName]
            theValues = theDF[standUnit.stand] ** 2
            # 是否小于阈值
            if theValues.max() < standUnit.threshold:
                return -1

            # 获取检测到的时间戳
            theTimestamp = int(theDF.loc[theValues.idxmax()][self.THE_TIMESTAMP]) #type: ignore
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
        def __init__(self, typeName: str, stand: str, threshold: int, diffThreshold: int, diffDistance: int, isABS: bool= False, bias: int= 0) -> None:
            """
            初始化

            Args:
                typeName (str): 类型数据名称
                stand (str): 标准名称
                threshold (int): 阈值
                diffThreshold (int): 差分阈值
                diffDistance (int): 差分距离
                isABS (bool, optional): 是否绝对值. Defaults to False.
                bias (int, optional): 偏差位移. Defaults to 0.
            """
            super().__init__(typeName, stand, threshold, isABS, bias)
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