import pandas as pd
import numpy as np

def newton_interpolation(x, y, target_x):

    n = len(x)
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

def fill_missing_temperature(filepath, output_filepath):
    df = pd.read_excel(filepath)
    # data = df[['时间', '当前温度', '最高温度', '最低温度', '天气', '风向', '风速', '湿度', '日出时间', '日落时间']].copy()  
    data = df[['日期', '比值']].copy()
    data.loc[:, 'x'] = range(1, len(data) + 1)


    missing_indices = data[data['比值'].isnull()].index

    for idx in missing_indices:

        start = max(0, idx - 100)
        end = min(len(data), idx + 100)
        local_data = data.iloc[start:end]


        known_data = local_data.dropna()
        x = known_data['x'].values
        y = known_data['比值'].values


        target_x = data.at[idx, 'x']
        interpolated_value = newton_interpolation(x, y, target_x)
        data.loc[idx, '比值'] = interpolated_value 

    data = data.drop(columns=['x'])

    data.to_excel(output_filepath, index=False)
    print(f"插值完成，结果已保存到 {output_filepath}")

input_filepath = "/home/mtftau-5/workplace/数模/输出文件/电站3最大发电量与辐照值比值.xlsx"
output_filepath = "/home/mtftau-5/workplace/数模/输出文件/电站3最大发电量与辐照值比值（最小二乘法拟合）.xlsx"
fill_missing_temperature(input_filepath, output_filepath)
