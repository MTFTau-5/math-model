import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zhplot

def newton_interpolation(x, y, target_x):
    n = len(x)
    if n == 0:
        return np.nan
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / (x[i + j] - x[i])

    result = divided_diff[0, 0]
    product = 1.0
    for i in range(1, n):
        product *= (target_x - x[i - 1])
        result += divided_diff[0, i] * product

    return result

# 填充缺失温度值的函数
def fill_missing_temperature(filepath, output_filepath):
    df = pd.read_excel(filepath)
    data = df[['时间', '当前温度', '最高温度', '最低温度', '天气', '风向', '风速', '湿度', '日出时间', '日落时间']].copy()
    
    data.loc[:, 'x'] = range(1, len(data) + 1)

    missing_indices = data[data['当前温度'].isnull()].index

    for idx in missing_indices:
        start = max(0, idx - 10)
        end = min(len(data), idx + 10)
        local_data = data.iloc[start:end]

        known_data = local_data.dropna()
        x = known_data['x'].values
        y = known_data['当前温度'].values

        if len(known_data) == 0:
            print(f"在索引 {idx} 附近没有已知数据，跳过插值。")
            continue

        target_x = data.at[idx, 'x']
        interpolated_value = newton_interpolation(x, y, target_x)
        data.loc[idx, '当前温度'] = interpolated_value

    data = data.drop(columns=['x'])
    data.to_excel(output_filepath, index=False)
    print(f"插值完成，结果已保存到 {output_filepath}")
    return data

# 可视化函数，一次显示 4 天的数据
def visualize_temperature(data, start_date):
    # 确保时间列是日期时间类型
    data['时间'] = pd.to_datetime(data['时间'])
    # 筛选出连续 4 天的数据
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=4)
    filtered_data = data[(data['时间'] >= start_date) & (data['时间'] < end_date)]

    if filtered_data.empty:
        print("未找到符合条件的数据，请检查起始日期。")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data['时间'], filtered_data['当前温度'], label='当前温度', marker='o', markersize=3)
    plt.xlabel('时间')
    plt.ylabel('当前温度')
    plt.title(f'当前温度随时间变化（{start_date} - {end_date - pd.Timedelta(days=1)}）')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

input_filepath = "/home/mtftau-5/workplace/2025赛题/2025赛题/2025_A题/附件/电站3天气数据.xlsx"
output_filepath = "/home/mtftau-5/workplace/2025赛题/2025赛题/2025_A题/附件/3插值后天气数据.xlsx"
interpolated_data = fill_missing_temperature(input_filepath, output_filepath)

start_date = '2024-02-27'
visualize_temperature(interpolated_data, start_date)
