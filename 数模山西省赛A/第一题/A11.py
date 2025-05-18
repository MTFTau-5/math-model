import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pre_traverse(really_time, power):
    print("开始遍历数据...")
    for i in range(len(really_time)):
        time_str = str(really_time[i])
        power_val = power[i]
        if pd.isna(power_val) or power_val == 0:
            print(f"时间 {time_str} 的发电量数据异常（0或缺失）")
        if "21:55" in time_str:
            print(f"检测到 21:55 时间点，当前发电量：{power_val}")
    print("数据遍历完成。")


def error_compare(power, really_time):
    # 处理发电量下降异常（原逻辑保留）
    for i in range(len(power) - 1):
        if power[i + 1] - power[i] < 0 and power[i + 1] != 0.0:
            if i > 0 and i < len(power) - 2 and power[i - 1] > 0 and power[i + 2] > 0:
                power[i + 1] = (power[i] + power[i + 2]) / 2
            print(f"错误数据（下降异常）: 前值={power[i]}, 后值={power[i + 1]}")

    for i in range(len(really_time)):
        time_str = str(really_time[i])
        if "21:55" in time_str:
            current_val = power[i]
            if pd.isna(current_val) or current_val == 0:  
                for j in range(i-1, -1, -1):
                    prev_val = power[j]
                    if not pd.isna(prev_val) and prev_val != 0:  
                        power[i] = prev_val
                        print(f"时间 {time_str} 已继承前一位有效数据：{prev_val}")
                        break
                else: 
                    print(f"时间 {time_str} 无前序有效数据，无法继承")
    return power


def load_data(filepath):
    df = pd.read_excel(filepath)
    really_time = df['时间'].values
    power = df['当日累计发电量kwh'].values.astype(np.float64)
    return really_time, power


def save_xlsx(power, really_time, filename='time-kwh.xlsx'):
    df = pd.DataFrame({
        '时间': really_time,
        '发电量': power
    })
    df.to_excel(filename, index=False)
    print(f"数据已保存到 {filename}")


def visualize_data(really_time, power):
    plt.figure(figsize=(12, 6))
    plt.plot(really_time, power, label='Power Data', linestyle='-', marker='o', markersize=3)
    plt.xlabel('Time')
    plt.ylabel('Power (kWh)')
    plt.title('Power Data Visualization')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    filepath = "/home/mtftau-5/workplace/2025赛题/2025赛题/2025_A题/附件/电站1发电数据.xlsx"
    really_time, power = load_data(filepath)
    
    pre_traverse(really_time, power)
    power = error_compare(power, really_time)
    save_xlsx(power, really_time)
    # visualize_data(really_time, power)  # 按需取消注释
