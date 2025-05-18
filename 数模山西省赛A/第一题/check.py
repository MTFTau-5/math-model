import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data(filepath):
    """
    从指定的 CSV 文件中加载数据，提取 '时间' 和 '发电量' 列。
    :param filepath: CSV 文件的路径
    :return: 包含时间和发电量的数组，如果文件未找到或出现其他错误则返回 None
    """
    try:
        df = pd.read_csv(filepath)
        data = df[['时间', '发电量']].values
        really_time = data[:, 0]
        power = data[:, 1]
        print("时间:", really_time)
        print("发电量:", power)
        return really_time, power
    except FileNotFoundError:
        print(f"文件 {filepath} 未找到。")
        return None, None
    except Exception as e:
        print(f"读取文件时出现错误: {e}")
        return None, None

# 加载数据
really_time, power = load_data("/home/mtftau-5/workplace/数模/输出文件/电站1时间转化_新数据.csv")

if really_time is not None and power is not None:
    # 遍历发电量数据，检查是否存在递减情况
    for i in range(len(power) - 1):
        if power[i] > power[i + 1]:
            # 检查时间间隔的平方是否大于 25
            if (really_time[i + 1] - really_time[i]) ** 2 > 25:
                continue
            else:
                print("错误数据:", power[i], power[i + 1], i)
                break
        else:
            # 当遍历到倒数第二个数据时，如果没有发现错误则输出数据正确
            if i == len(power) - 2:
                print("数据正确")
            continue
    