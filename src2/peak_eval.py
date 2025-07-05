import pickle
import sys
from typing import Callable, Dict, List, Union
import pandas as pd

sys.path.append("..\\")
from Util.Json import loadJson
from Util.Path import getFPsByEndwith, getFPsByName

def peakF(merged_df: pd.DataFrame, threshold: int= 27, window_size: int= 2000) -> int:
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

        # 选择Gx作为波峰检测的信号
        gx_values = window_data['Gx']  # Gx的数值直接使用

        # 检查是否有值超过阈值
        if gx_values.max() >= threshold:
            peak_index = window_data.index.get_loc(gx_values.idxmax())
            # 返回中间时间戳
            return merged_df.iloc[peak_index]['unixTimestamp_acc']

        # 跳过当前窗口（窗口之间不重叠）
        i = window_data.index[-1] + 1
    return -1

class DataTimeSet:
    """
    时间戳与原始数据集合
    """
    def __init__(self, timeList: List[int], dataFrame: pd.DataFrame) -> None:
        """
        初始化

        Args:
            timeList (List[int]): 时间戳列表
            dataFrame (pd.DataFrame): 原始数据框架
        """
        self.timeList = timeList
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
        timeList = [theJson["info"]["startTimestamp"] + 1000 for theJson in jsons]

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

def peakEvalAcc(peakF: Callable[[pd.DataFrame], float], checkHalfRange: int, nameToDataTimeSetDict: Dict[str, DataTimeSet]) -> float:
    """
    计算检测函数准确率

    Args:
        peakF (Callable[[pd.DataFrame], float]): 检测函数
            输入 数据框架
            生成 中间时间戳
        checkHalfRange (int): 测试范围半径（ms）
        nameToDataTimeSetDict (Dict[str, DataTimeSet]): 数据集数据名称 与 时间戳与原始数据集合 字典

    Returns:
        float: 准确率
    """
    # 测试总数
    allNum = 0
    # 检测总数
    peakNum = 0
    # 遍历数据单元
    for name, dataTimeSetDict in nameToDataTimeSetDict.items():
        allNum += len(dataTimeSetDict.timeList)
        print(f"检测 {name}。。。")
        # 对每个时间戳测试
        for index, midTimestamp in enumerate(dataTimeSetDict.timeList):
            # 原始数据框架
            dataframe = dataTimeSetDict.dataFrame
            # 起始数据
            startT = midTimestamp - checkHalfRange
            endT = midTimestamp + checkHalfRange
            testDataframe = dataframe[(dataframe['unixTimestamp_acc'] >= startT) & (dataframe['unixTimestamp_acc'] <= endT)]
            # 检测
            peakTimestamp = peakF(testDataframe)
            # 是否在合理范围内
            if abs(peakTimestamp - midTimestamp) < 500:
                peakNum += 1
                print(f"{index} check!")
        print(f"{name} 检测完成。")

    return float(peakNum) / float(allNum)

DATASET_PATH = r"G:\Badminton\BADS_CLL"
INIT_DATA_PATH = r"..\data\processed_fir\merged_files"
SAVE_PATH = r"..\src2\theSet\set1.pkl"
LOAD_PATH = r"..\src2\theSet\set1.pkl"
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
    # 8 秒检测范围
    print(peakEvalAcc(peakF, 2000, nameToDataTimeSetDict))

if __name__ == "__main__":
    main()