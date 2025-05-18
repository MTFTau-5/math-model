import numpy as np
import pandas as pd
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def load_data(filepath):
    df = pd.read_csv(filepath)
    data = df[['时间', '发电量']].values
    really_time = data[:, 0]
    power = data[:, 1]
    print("时间:", really_time)
    print("发电量:", power)
    return really_time, power
def get_data(really_time,power):
    q = []
    new_time = []
    ret_time = []
    new_power = []
    for i in really_time:
        new_time.append(i-int(i))
    for i in range(len(new_time)-1):
        if power[i+1]-power[i]>=0:
            new_power.append(power[i])
            ret_time.append(new_time[i])
        if new_time[i+1] - new_time[i]<0:
            new_power.append(' ')
            ret_time.append(' ')
        
    return ret_time, new_power
def chafun(really_time, power):
    x = []
    y = []
    length = len(really_time)
    for i in range(length - 1):
        a = really_time[i]
        b = power[i]
        c = really_time[i + 1]
        d = power[i + 1]
        # e = power[i + 2]
  
        if (c-a)**2 > 25:
            x.append(' ')
            y.append(' ')
        elif (d-b) >= 0:
            x.append((d-b)/(c-a))     
            y.append(really_time[i])       
    return y, x
really_time, power = load_data("/home/mtftau-5/workplace/数模/输出文件/电站1时间转化.csv")
x =[ ]
y = []
y, x = chafun(really_time, power)  
h_time, h_power = get_data(really_time, power)
print(f"h_time:{len(h_time)}, h_power:{len(h_power)}")
print(f"y:{len(y)}x:{len(x)}")
# print("\n前 12 个点的发电量:") 
# for i in range(min(240, len(h_power))):
#     print( h_power[i])
