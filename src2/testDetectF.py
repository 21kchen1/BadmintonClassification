#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2025-08-21 16:20:08
@Author  :   Chen
@File    :   Model\DetectionF.py
@Software:   VSCode
@Description:
    击球检测算法
"""

from copy import deepcopy
import math
from typing import Dict, List, Union
import pandas as pd


class DetectF:
    """
    击球检测算法
    """
    FormatData = Dict[str, pd.DataFrame]
    """
    格式数据
    """

    def check(self, dfDict: FormatData) -> List[int]:
        """
        检测 dfDict 的所有击球时间戳

        Args:
            dfDict (FormatData): 格式流式数据

        Returns:
            List[int]: 检测的击球时间戳
        """
        ...

class SlideWindowF(DetectF):
    """
    滑动窗口检测函数
    """
    THE_TIMESTAMP = 'unixTimestamp'
    """
    时间戳
    """

    class ConfigUnit:
        """
        配置单元
        """
        def __init__(self, typeDataName: str, typeDataAttr: str, threshold: int, isSquare: bool= False, bias: int= 0) -> None:
            """
            初始化

            Args:
                typeDataName (str): 类型数据字符标识
                typeDataAttr (str): 作为检测标准的类型数据元素字符标识
                threshold (int): 检测阈值
                isSquare (bool, optional): 是否平方. Defaults to False.
                bias (int, optional): 结果偏移量. Defaults to 0.
            """
            self.typeDataName = typeDataName
            self.typeDataAttr = typeDataAttr
            self.threshold = threshold
            self.isSquare = isSquare
            self.bias = bias

    class WindowJudge:
        """
        窗口检测
        """
        def check(self, configUnits: List["SlideWindowF.ConfigUnit"], windowDFD: DetectF.FormatData) -> Union[int, None]:
            """
            窗口检测

            Args:
                configUnits (List[SlideWindowF.ConfigUnit]): 配置单元列表
                windowDFD (DetectF.FormatData): 窗口格式数据

            Returns:
                Union[int, None]: 检测结果时间戳
            """
            ...

    def __init__(self, configUnits: List[ConfigUnit], windowJudge: WindowJudge, windowSize: int= 2000, *params) -> None:
        """
        初始化

        Args:
            configUnits (List[ConfigUnit]): 检测标准列表
            windowJudge (WindowJudgeF): 窗口检测方法
            windowSize (int, optional): 检测窗口. Defaults to 2000.
            *params (any): 其余参数
        """
        self._configUnits = configUnits
        self._windowJudge = windowJudge
        self._windowSize = windowSize

    def getSubDataFrameDict(self, dataFrameDict: DetectF.FormatData, startT: int, endT: int) -> Dict[str, pd.DataFrame]:
        """
        根据时间戳获取子数据框架字典
        1. 时间戳必须有序

        Args:
            dataFrameDict (Dict[str, pd.DataFrame]): 时间戳有序数据框架字典
            startT (int): 开始数据
            endT (int): 结束时间

        Returns:
            Dict[str, pd.DataFrame]: 子时间框架字典
        """
        subDataFrameDict = {}
        for typeName, dataFrame in dataFrameDict.items():
            # 找到时间戳范围的起始和结束索引
            start_idx = dataFrame[self.THE_TIMESTAMP].searchsorted(startT, side="left")
            end_idx = dataFrame[self.THE_TIMESTAMP].searchsorted(endT, side= "right")
            # 切片数据
            subDataFrameDict[typeName] = dataFrame.iloc[start_idx:end_idx]
        return subDataFrameDict

    def _getWindowMaxTimestamp(self, windowDFD: DetectF.FormatData) -> int:
        """
        自定义方法获取窗口数据的最大数据时间戳

        Args:
            windowDFD (DetectF.FormatData): 窗口数据框架字典

        Returns:
            int: 最大数据时间戳
        """
        # 第一个数据作为基准单元
        standUnit = self._configUnits[0]
        theDF = windowDFD[standUnit.typeDataName]
        return int(theDF.loc[(theDF[standUnit.typeDataAttr] ** 2).idxmax()][self.THE_TIMESTAMP]) #type: ignore

    def check(self, dfDict: Dict[str, pd.DataFrame]) -> List[int]:
        """
        检测 dfDict 的所有击球时间戳

        Args:
            dfDict (FormatData): 格式流式数据

        Returns:
            List[int]: 检测的击球时间戳
        """
        # 检测结果
        checkTimestamps = []
        # 半窗口大小
        halfWindowSize = self._windowSize // 2
        # 选择第一个标准为主标准
        mdf = dfDict[self._configUnits[0].typeDataName]
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
            # 获取子窗口
            # windowData = df[(df[self.THE_TIMESTAMP] >= starT) & (df[self.THE_TIMESTAMP] <= endT)]
            windowDFD = self.getSubDataFrameDict(dfDict, starT, endT)
            # 获取窗口最大值的时间戳
            windowMaxTimestamp = self._getWindowMaxTimestamp(windowDFD)
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
            judgeResult = self._windowJudge.check(self._configUnits,windowDFD)

            # 不符合条件
            if judgeResult is None:
                continue
            # 符合条件
            checkTimestamps.append(judgeResult)
        return checkTimestamps

class peckJudge(SlideWindowF.WindowJudge):
    """
    阈值检测击球波峰
    """

    def check(self, configUnits: List[SlideWindowF.ConfigUnit], windowDFD: Dict[str, pd.DataFrame]) -> Union[int, None]:
        """
        阈值检测击球波峰
        1. 选择第一个 configUnit 测试
        2. 如果 isSquare is True，则平方数据后进行检测

        Args:
            windowDFD (pd.DataFrame): 窗口格式数据

        Returns:
            Union[int, None]: 检测到的中值时间戳
        """
        # 选择作为波峰检测的信号
        standUnit = configUnits[0]
        theValues = windowDFD[standUnit.typeDataName][standUnit.typeDataAttr]

        # 是否平方
        if standUnit.isSquare: theValues = theValues ** 2
        # 检查是否有值超过阈值
        if theValues.max() >= standUnit.threshold:
            return int(windowDFD[standUnit.typeDataName].loc[theValues.idxmax()][SlideWindowF.THE_TIMESTAMP]) + standUnit.bias # type: ignore
        return None

class peckAudioJudge(SlideWindowF.WindowJudge):
    """
    时域窗口+阈值平方+音频检测击球波峰
    """

    def check(self, configUnits: List[SlideWindowF.ConfigUnit], windowDFD: Dict[str, pd.DataFrame]) -> Union[int, None]:
        """
        使用时域窗口+阈值平方+音频检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

        Args:
            windowDFD (pd.DataFrame): 窗口数据字典

        Returns:
            int: 检测到的中值时间戳
        """
        standUnit = configUnits[0]
        # 选择作为波峰检测的信号 并 平方
        theValues = windowDFD[standUnit.typeDataName][standUnit.typeDataAttr] ** 2
        # 最大值的时间戳
        # maxTimestamp = int(windowDFD[standUnit.typeDataName].loc[theValues.idxmax()][SlideWindowF.THE_TIMESTAMP]) #type: ignore
        maxTimestamp = int(windowDFD[standUnit.typeDataName].loc[theValues.idxmax()][SlideWindowF.THE_TIMESTAMP].item())

        audioValues = windowDFD["AUDIO"]["values"]
        audioMaxTimestamp = int(windowDFD["AUDIO"].loc[audioValues.idxmax()][SlideWindowF.THE_TIMESTAMP].item()) if len(windowDFD["AUDIO"]) > 0 else -math.inf

        # 检查是否有值超过阈值
        if theValues.max() >= standUnit.threshold and abs(audioMaxTimestamp - maxTimestamp) < 1000 and audioValues.max() >= 30:
        # if theValues.max() >= standUnit.threshold and audioValues.max() >= 30:
            return maxTimestamp + standUnit.bias
        return None