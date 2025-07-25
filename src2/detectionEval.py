#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2025-07-06 11:33:01
@Author  :   Chen
@File    :   src2\DetectionEval.py
@Software:   VSCode
@Description:
    击球动作检测算法准确率计算
"""

import sys
sys.path.append("..\\")
import pickle
from typing import Callable, Dict, List, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
from src2.DetectionPlot import compareTestDetection
from src2 import DetectionF
from Util.Json import loadJson
from Util.Path import getFPsByEndwith, getFPsByName

class DataTimeSet:
    """
    时间戳与原始数据集合
    """
    def __init__(self, indexAndTimeList: List[Tuple[int, int]], dataFrame: pd.DataFrame) -> None:
        """
        初始化

        Args:
            indexAndTimeList (List[Tuple[int, int]]): 下标与时间戳元组列表
            dataFrame (pd.DataFrame): 原始数据框架
        """
        self.indexAndTimeList = indexAndTimeList
        self.dataFrame = dataFrame

def getNameToDataTimeSetDict(datasetRoot: str, initDataRoot: str, savePath: Union[str, None]= None) -> Union[Dict[str, DataTimeSet], None]:
    """
    获取 数据集数据名称 与 时间戳与原始数据集合 字典

    Args:
        datasetRoot (str): 数据集根路径
        initDataRoot (str): 原始数据根路径
        savePath (Union[str, None], optional): 字典保存路径. Defaults to None.

    Returns:
        Union[Dict[str, DataTimeSet], None]: 数据集数据名称 与 时间戳与原始数据集合 字典
    """
    nameToDataTimeSetDict = {}
    # 获取数据集 json 路径列表
    jsonsPathList = getFPsByEndwith(datasetRoot, ".json")

    for jsonsPath in jsonsPathList:
        # 获取 json 列表
        jsons = loadJson(jsonsPath)
        # 记录名称
        recordName = jsons[0]["info"]["recordName"]

        # 遍历并获取中间的时间戳 获取时间戳序列
        timeList = [(index, theJson["info"]["startTimestamp"] + 1000) for index, theJson in enumerate(jsons) if not theJson["data"] is None]

        # 获取对应原始数据路径
        initDataPaths = getFPsByName(initDataRoot, recordName, first= True)
        if len(initDataPaths) != 1:
            print(f"原始数据异常: {recordName}")
            return None
        dataFrame = pd.read_csv(initDataPaths[0])

        # 生成键值对
        nameToDataTimeSetDict[recordName] = DataTimeSet(timeList, dataFrame)
        print(f"{recordName} 载入完成")

    # 保存
    if not savePath is None:
        with open(savePath, "wb") as file:
            pickle.dump(nameToDataTimeSetDict, file)
    return nameToDataTimeSetDict

def loadNameToDataTimeSetDict(loadPath: str) -> Union[Dict[str, DataTimeSet], None]:
    """
    载入 数据集数据名称 与 时间戳与原始数据集合 字典

    Args:
        loadPath (str): 载入路径

    Returns:
        Union[Dict[str, DataTimeSet], None]: 载入字典
    """
    nameToDataTimeSetDict = None
    with open(loadPath, "rb") as file:
        nameToDataTimeSetDict = pickle.load(file)
    return nameToDataTimeSetDict

def detectionEvalAcc(detectionF: Callable[[pd.DataFrame], int], checkHalfRange: int, bias: int, nameToDataTimeSetDict: Dict[str, DataTimeSet], plotList: Union[List[str], None]= None) -> Tuple[int, int, int]:
    """
    计算检测函数准确率

    Args:
        detectionF (Callable[[pd.DataFrame], int]): 检测函数
        checkHalfRange (int): 测试范围半径（ms）
        bias (int): 容许的最大偏差时间（ms）
        nameToDataTimeSetDict (Dict[str, DataTimeSet]): 数据集数据名称 与 时间戳与原始数据集合 字典
        plotList (Union[List[str], None]): 可视化列表. Defaults to None.

    Returns:
        Tuple[int, int]: 检测数、总数、漏警数
    """
    # 测试总数
    allNum = 0
    # 检测总数
    detectionNum = 0
    # 漏检总数
    missAlarmNum = 0
    # 遍历数据单元
    for name, dataTimeSetDict in nameToDataTimeSetDict.items():

        allNum += len(dataTimeSetDict.indexAndTimeList)
        print(f"检测 {name}。。。")
        # 对每个时间戳测试
        for index, midT in dataTimeSetDict.indexAndTimeList:
            # if "Man_Low_ForehandHigh_LeiYang_2_1" in name and index == 6:
            #     print()
            # 原始数据框架
            dataframe = dataTimeSetDict.dataFrame
            # 起始数据
            startT = midT - checkHalfRange
            endT = midT + checkHalfRange
            testDF = dataframe[(dataframe['unixTimestamp_acc'] >= startT) & (dataframe['unixTimestamp_acc'] <= endT)] # type: ignore
            # 检测
            detectionT = detectionF(testDF) # type: ignore

            # 检测失败 是否在合理范围内
            if detectionT == -1 or abs(detectionT - midT) > bias:
                # 漏警记录
                if detectionT == -1: missAlarmNum += 1
                # 可视化失败类型
                if plotList:
                    detectionDF = dataframe[(dataframe['unixTimestamp_acc'] >= detectionT - checkHalfRange) & (dataframe['unixTimestamp_acc'] <= detectionT + checkHalfRange)] # type: ignore
                    compareTestDetection(testDF, midT, detectionDF, detectionT, plotList, index= "unixTimestamp_acc") # type: ignore
                print(f"{index} miss!")
                continue
            # 检测成功
            detectionNum += 1
            print(f"{index} check!")
        print(f"{name} 检测完成。")

    return detectionNum, allNum, missAlarmNum

DATASET_PATH = r"G:\Badminton\BADS_CLL_E_CLEAN"
INIT_DATA_PATH = r"..\data\processed_fir\merged_files"
SAVE_PATH = r"..\src2\theSet\set2.pkl"
LOAD_PATH = r"..\src2\theSet\set2.pkl"
LOAD = True

def main() -> None:
    # 获取名称与数据时间集合字典
    nameToDataTimeSetDict = None
    if not LOAD:
        nameToDataTimeSetDict = getNameToDataTimeSetDict(DATASET_PATH, INIT_DATA_PATH, SAVE_PATH)
    else:
        nameToDataTimeSetDict = loadNameToDataTimeSetDict(LOAD_PATH)
    if nameToDataTimeSetDict is None:
        return
    # 构建检测函数 angularSpeedX Gx
    standUnits = [
        DetectionF.WindowPeak.StandUnit("angularSpeedY", 21, True, 107),
        # DetectionF.WindowPeak.StandUnit("angularSpeedX", 21, True, 0),
        # DetectionF.WindowPeak.StandUnit("angularSpeedZ", 21, True, 0),
        # DetectionF.WindowPeak.StandUnit("Gy", 200, True, 0),
        # DetectionF.WindowPeak.StandUnit("Gx", 400, True, 0),
        # DetectionF.WindowPeak.StandUnit("Gz", 400, True, 0),
    ]
    detectionF = DetectionF.WindowPeakS(standUnits, windowSize= 2000)
    # detectionF = DetectionF.WindowMaxEnergyPeak(standUnits, windowSize= 2000)
    # 8 秒检测范围
    plotList = [standUnit.stand for standUnit in standUnits]
    detectionNum, allNum, missAlarmNum = detectionEvalAcc(detectionF.midCheck, 4000, 500, nameToDataTimeSetDict, )
    print(f"detectionNum: {detectionNum}, allNum: {allNum}, missAlarmNum: {missAlarmNum}\
        \n accRate: {float(detectionNum) / float(allNum)}, missAlarmRate: {float(missAlarmNum) / float(allNum)}")

if __name__ == "__main__":
    main()