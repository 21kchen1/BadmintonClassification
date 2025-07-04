#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2025-05-28 15:11:04
@Author  :   Chen
@File    :   Util\Json.py
@Software:   VSCode
@Description:
    JSON 自定义序列化逻辑
    通过将 list 转换为 str 再格式化，使得保存的 JSON 中的 list 不会换行
"""

import json
from typing import Dict, List, Union

class _SingleLineArrayEncoder(json.JSONEncoder):
    """
    json 编码器
    将字符串化的 list 还原为 list
    """
    def encode(self, obj) -> str:
        result = super().encode(obj)
        result = result.replace("'", '"').replace('"[', "[").replace(']"', "]").replace("\\", "")
        return result

def _dictListToStr(data: dict) -> Union[dict, str]:
    """
    递归转换 list
    将字典中所有 list 转换为 str

    Args:
        data (dict): 需要转换的字典

    Returns:
        Union[dict, str]: 转换完成的字典
    """
    # 将字典转换为字符串
    if isinstance(data, list):
        return str(data)
    if not isinstance(data, dict):
        return data
    # 如果是字典，则将所有值中的 list 转换为字符串
    for key, value in data.items():
        data[key] = _dictListToStr(value)
    return data

def compactArrayJson(datas: list) -> str:
    """
    紧凑数组编码
    JSON 中的 list 将不会换行

    Args:
        datas (list): 列表或字典的 JSON

    Returns:
        str: 编码完成的 JSON
    """
    # 数据预处理
    if isinstance(datas, list):
        obj = [_dictListToStr(data) for data in datas]
    elif isinstance(datas, dict):
        obj = _dictListToStr(datas)
    # 编码
    return json.dumps(obj, cls= _SingleLineArrayEncoder, ensure_ascii= False, indent= 4)

def loadJson(path: str) -> List[Dict]:
    """
    读取并生成 json 数据字典列表

    Args:
        path (str): json 存储地址

    Returns:
        List[Dict]: 数据字典列表
    """
    with open(path, "r", encoding= "utf-8") as file:
        data = json.load(file)
    return data

def saveJson(filePath: str, json: List[dict]) -> None:
    """
    构建路径保存 json

    Args:
        filePath (str): 存储文件路径
        json (List[str]): 存储 json
    """
    with open(filePath, "w", encoding= "utf-8") as file:
        file.write(compactArrayJson(json))