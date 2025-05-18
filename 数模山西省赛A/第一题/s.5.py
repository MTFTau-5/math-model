import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zhplot

def dataload(filepath):
    df = pd.read_excel(filepath)
    # data = df[['时间', '发电量']].values
    data = df[['时间', '当日累计发电量kwh']]
    # 将时间列转换为 datetime 类型
    data['时间'] = pd.to_datetime(data['时间'])
    return data

def process_and_plot(data):
    # 按小时重采样并计算平均值
    hourly_avg = data.set_index('时间').resample('H')['当日累计发电量kwh'].mean()
    # 去除缺失值
    hourly_avg = hourly_avg.dropna()

    # 筛选出前四天的数据
    start_time = hourly_avg.index.min()
    end_time = start_time + pd.Timedelta(days=4)
    hourly_avg = hourly_avg[(hourly_avg.index >= start_time) & (hourly_avg.index < end_time)]

    time_avg = hourly_avg.index
    power_avg = hourly_avg.values

    plt.figure(figsize=(10, 6))
    plt.plot(time_avg, power_avg, 'o-', label='小时均值')
    plt.xlabel('时间')
    plt.ylabel('均值大小')
    plt.title('前四天每小时均值')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

filepath = '/home/mtftau-5/workplace/2025赛题/2025赛题/2025_A题/附件/电站3发电数据.xlsx'
data = dataload(filepath)
process_and_plot(data)
    