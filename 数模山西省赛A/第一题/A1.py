import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def error_compare(power):
    for i in range(len(power) - 1):
        if power[i + 1] - power[i] < 0 and power[i + 1] != 0.0:
            # 处理异常值
            if i > 0 and i < len(power) - 2 and power[i - 1] > 0 and power[i + 2] > 0:
                power[i + 1] = (power[i] + power[i + 2]) / 2
            elif power[i + 1] == 0 and i > 0:
                power[i + 1] = power[i]
            print("错误数据:", power[i], power[i + 1])
    return power

def load_data(filepath):
    df = pd.read_excel(filepath)
    data = df[['时间', '当日累计发电量kwh']].values
    really_time = data[:, 0]
    power = data[:, 1]
    return really_time, power

def string_to_datetime(really_time):
    reall_time = really_time.split(" ")
    date_str = reall_time[0]
    time_str = reall_time[1]
    return date_str, time_str

def save_xlsx(power, result1, filename='time-kwh.xlsx'):
    df = pd.DataFrame({'时间': result1, '发电量': power})
    df.to_excel(filename, index=False)

def visualize_data(really_time, power):
    limited_time = really_time[:480 * 2]
    limited_power = power[:480 * 2]
    plt.plot(limited_time, limited_power, label='Power Data')
    plt.xlabel('Time')
    plt.ylabel('Power (kWh)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    filepath = "/home/mtftau-5/workplace/2025赛题/2025赛题/2025_A题/附件/电站4发电数据.xlsx"
    really_time, power = load_data(filepath)
    time_str = []
    date_str = []
    for i in range(int(len(really_time))):
        date, time = string_to_datetime(really_time[i])
        time_str.append(time)
        date_str.append(date)
    print("时间字符串:", time_str)

    min_time = []
    for i in range(len(time_str)):
        q = time_str[i].split(":")
        hour = float(q[0])
        minute = float(q[1])
        time = hour + minute / 60
        result1 = round(time, 3)
        print("当前时间:", result1)
        power_value = power[i]
        print("当前发电量:", power_value)
        min_time.append(result1)

    power = error_compare(power)
    save_xlsx(power, min_time, filename='time-kwh.xlsx')
    visualize_data(min_time, power)    