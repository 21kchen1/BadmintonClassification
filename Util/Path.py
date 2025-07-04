#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

r"""
@DATE    :   2025-03-11 14:49:36
@Author  :   Chen
@File    :   Util\Path.py
@Software:   VSCode
@Description:
    路径相关的方法
"""

import os
import re
from typing import List

def getFilePaths(rootPath: str) -> List[str]:
    """
    获取根目录中所有文件路径

    Args:
        rootPath (str): 根目录

    Returns:
        list: 文件路径列表
    """
    filePaths = []
    for root, _, fileNames in os.walk(rootPath):
        filePaths.extend([os.path.join(root, fileName) for fileName in fileNames])
    return filePaths

def getDirPaths(rootPath: str) -> List[str]:
    """
    获取根目录中所有文件夹路径

    Args:
        rootPath (str): 根目录

    Returns:
        List[str]: 文件夹路径列表
    """
    dirPaths = []
    for root, dirNames, _ in os.walk(rootPath):
        dirPaths.extend([os.path.join(root, dirName) for dirName in dirNames])
    return dirPaths

def getFPsByEndwith(rootPath: str, endwith: str) -> List[str]:
    """
    获取根目录中指定后缀文件路径

    Args:
        rootPath (str): 根目录
        endwith (str): 后缀

    Returns:
        List[str]: 文件路径列表
    """
    filePaths = []
    for path in getFilePaths(rootPath):
        if not path.endswith(endwith):
            continue
        filePaths.append(path)
    return filePaths

def getDPsByName(rootPath: str, dirName: str) -> List[str]:
    """
    获取根目录下指定文件夹路径列表

    Args:
        rootPath (str): 根目录
        dirName (str): 指定文件夹

    Returns:
        List[str]: 文件夹路径列表
    """
    dirPaths = []
    for dirPath in getDirPaths(rootPath):
        if not re.split(r"[\\/]", dirPath)[-1] == dirName:
            continue
        dirPaths.append(dirPath)
    return dirPaths