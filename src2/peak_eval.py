from json import load
import sys
from typing import Callable, Dict, List
import pandas as pd

sys.path.append("..\\")
from Util.Json import loadJson
from Util.Path import getFPsByEndwith

def peakF(merged_df: pd.DataFrame, threshold: int=27, window_size: int=10000) -> int:
    """
    使用时域窗口+阈值检测击球波峰，并保存达到条件的击球数据前后一秒的时域数据。

    Args:
        merged_df (pd.DataFrame): 包含 gx 的数据框架
        threshold (int, optional): 检测阈值. Defaults to 27.
        window_size (int, optional): 窗口大小. Defaults to 10000.

    Returns:
        int: 检测到的中值时间戳
    """
    # 获取数据的总长度
    total_length = len(merged_df)
    # 每次跳过窗口，避免重叠
    i = 0
    while i < total_length:
        current_timestamp = merged_df.iloc[i]['unixTimestamp_acc']

        # 定义时域窗口：每个窗口为10秒（10000毫秒）
        start_time = current_timestamp
        end_time = current_timestamp + window_size

        # 过滤当前窗口内的数据
        window_data = merged_df[(merged_df['unixTimestamp_acc'] >= start_time) & (merged_df['unixTimestamp_acc'] <= end_time)]

        if len(window_data) == 0:
            i += 1
            continue

        # 选择Gx作为波峰检测的信号
        gx_values = window_data['Gx']  # Gx的数值直接使用

        # 检查是否有值超过阈值
        if gx_values.max() >= threshold:
            # peak_value = gx_values.max()
            peak_index = gx_values.idxmax()
            # 返回中间时间戳
            return merged_df.iloc[peak_index]['unixTimestamp_acc']

        # 跳过当前窗口（窗口之间不重叠）
        i = window_data.index[-1] + 1
    return -1


def getNameTimesDict(datasetPath: str) -> Dict[str, List[int]]:
    """
    获取数据名称与击球时间戳列表字典

    Args:
        datasetPath (str): 数据集路径

    Returns:
        Dict[str, List[int]]: 数据名称与击球时间戳列表字典
    """
    nameTimesDict = {}
    jsonsPathList = getFPsByEndwith(datasetPath, ".json")

    for jsonsPath in jsonsPathList:
        # 获取 json 列表
        jsons = loadJson(jsonsPath)
        # 记录名称
        recordName = jsons[0]["info"]["recordName"]
        # 遍历并获取中间的时间戳
        nameTimesDict[recordName] = [theJson["info"]["startTimestamp"] + 1000 for theJson in jsons]
        print(nameTimesDict[recordName])

    return nameTimesDict

def peakEvalAcc(peakF: Callable[[pd.DataFrame], float], datasetPath: str, initDataPath: str) -> float:
    """
    计算检测函数准确率

    Args:
        peakF (Callable[[pd.DataFrame], float]): 检测函数
        datasetPath (str): 数据集路径
        initDataPath (str): 原始数据路径

    Returns:
        float: 准确率
    """
    # 获取名称与时间戳列表字典
    nameTimesDict = getNameTimesDict(datasetPath)

DATASET_PATH = r"G:\Badminton\BADS_CLL"
INIT_DATA_PATH = r"data\processed_fir\merged_files"

def main() -> None:
    peakEvalAcc(peakF, DATASET_PATH, INIT_DATA_PATH)

if __name__ == "__main__":
    main()