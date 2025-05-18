import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    df = pd.read_excel(filepath)
    data = df[['时间', '辐照强度w/m2']].values
    really_time = data[:, 0]
    light = data[:, 1]
    print("时间:", really_time)
    print("辐照强度:", light)
    return really_time, light


def chafun(really_time, light):
    x = []
    length = len(really_time)
    for i in range(length):
        if i < length - 2:
            a = light[i]
            b = light[i + 1]
            c = light[i + 2]
            if (b - a) * (b - c) > 0 and (b - a) ** 2 > (a * 0.2) ** 2 and (b - c) ** 2 > (c / 5) ** 2:
                x.append((a + c) / 2)
            else:
                x.append(light[i])
        else:
            x.append(light[i])
    return x



def visualize_data(really_time, light):
    limited_time = really_time[:480 * 3]
    limited_light = light[:480 * 3]
    plt.plot(limited_time, limited_light, label='Light Data')
    plt.xlabel('Time')
    plt.ylabel('Light (w/m2)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    filepath = "/home/mtftau-5/workplace/2025赛题/2025赛题/2025_A题/附件/电站4环境监测仪数据.xlsx"
    really_time, light = load_data(filepath)

    x = chafun(really_time, light)

    filename = '/home/mtftau-5/workplace/数模/2文件/电站4辐照值.xlsx'

    df = pd.DataFrame({'时间': really_time, '辐照强度': x})
    df.to_excel(filename, index=False)
    print(f"Filtered data saved to {filename}")

    visualize_data(really_time, x)
    