import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import zhplot

# 加载数据
data = pd.read_excel('math-model-main\B题(1)\B题\附件\附件3.xlsx')
lmd = data['波数 (cm-1)'].values
rsl = data['反射率 (%)'].values
data = pd.read_excel('math-model-main\B题(1)\B题\附件\附件4.xlsx')
lmd1 = data['波数 (cm-1)'].values
rsl1 = data['反射率 (%)'].values
# 选择分析范围
mask = (lmd >= 1000) & (lmd <= lmd.max())
lmd = lmd[mask]
rsl1 = rsl1[mask]
lmd1 = lmd1[mask]
rsl = rsl[mask]


rsl11 = gaussian_filter1d(rsl1, sigma=50)

peak_wavenumbers = []
peak_wavenumbers_ = []
window_size = 100
for i in range(window_size, len(rsl1)-window_size):
    is_peak = True
    for j in range(1, window_size+1):
        if rsl1[i] <= rsl1[i-j] or rsl1[i] <= rsl1[i+j]:
            is_peak = False
            break
    if is_peak:
        peak_wavenumbers.append(lmd[i])
        print(i);
for i in range(window_size, len(rsl1)-window_size):
    is_peak = True
    for j in range(1, window_size+1):
        if rsl1[i] >= rsl1[i-j] or rsl1[i] >= rsl1[i+j]:
            is_peak = False
            break
    if is_peak:
        peak_wavenumbers_.append(lmd[i])
        print(i);
# 计算相邻峰值的间隔
if len(peak_wavenumbers) >= 2:
    intervals = np.diff(peak_wavenumbers)  # 计算相邻峰值的波数差
    avg_interval = np.mean(intervals)     # 计算平均间隔
    std_interval = np.std(intervals)      # 计算标准差
    print(f"峰值波数(cm⁻¹):", ["%.2f" % x for x in peak_wavenumbers])
    print(f"检测到 {len(peak_wavenumbers)} 个峰值")
    print("相邻峰值间隔(cm⁻¹):", ["%.2f" % x for x in intervals])
    print(f"平均间隔: {avg_interval:.2f} ± {std_interval:.2f} cm⁻¹")
    

    n = 2.1
    thickness = 1 / (2 * n * avg_interval) * 1e7 
    print(f"计算得到的薄膜厚度: {thickness:.2f} nm")
else:
    print("检测到的峰值不足，无法计算间隔")

# 可视化
plt.show()
plt.plot(lmd, ((1-rsl)/rsl)/((1-rsl1)/rsl1), 'b-', label='比值 ')
# plt.plot(peak_wavenumbers, rsl1[np.isin(lmd, peak_wavenumbers)], 'ro', label='检测到的峰值')

# 标注间隔
if len(peak_wavenumbers) >= 2:
    for i in range(len(peak_wavenumbers)-1):
        mid = (peak_wavenumbers[i] + peak_wavenumbers[i+1])/2
        plt.annotate(f'Δ={intervals[i]:.1f}', 
                    xy=(mid, np.mean(rsl1[np.isin(lmd, peak_wavenumbers)])),
                    ha='center', va='bottom', fontsize=8)

plt.xlabel('波数 (cm-1)')
plt.ylabel('反射率 (%)')
plt.title(f'峰值间隔分析 (平均间隔: {avg_interval:.2f} cm⁻¹)' if len(peak_wavenumbers)>=2 else '峰值检测')
plt.legend()
plt.grid(True)
plt.show()

# 计算相邻峰值的间隔
if len(peak_wavenumbers_) >= 2:
    intervals = np.diff(peak_wavenumbers_)  # 计算相邻峰值的波数差
    avg_interval = np.mean(intervals)     # 计算平均间隔
    std_interval = np.std(intervals)      # 计算标准差
    print(f"峰值波数(cm⁻¹):", ["%.2f" % x for x in peak_wavenumbers_])
    print(f"检测到 {len(peak_wavenumbers_)} 个峰值")
    print("相邻峰值间隔(cm⁻¹):", ["%.2f" % x for x in intervals])
    print(f"平均间隔: {avg_interval:.2f} ± {std_interval:.2f} cm⁻¹")
    

    n = 2.1
    thickness = 1 / (2 * n * avg_interval) * 1e7 
    print(f"计算得到的薄膜厚度: {thickness:.2f} nm")
else:
    print("检测到的峰值不足，无法计算间隔")

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(lmd, rsl1, 'b-', label='原始反射率')
plt.plot(peak_wavenumbers_, rsl1[np.isin(lmd, peak_wavenumbers_)], 'ro', label='检测到的峰值')

# 标注间隔
if len(peak_wavenumbers_) >= 2:
    for i in range(len(peak_wavenumbers_)-1):
        mid = (peak_wavenumbers_[i] + peak_wavenumbers_[i+1])/2
        plt.annotate(f'Δ={intervals[i]:.1f}', 
                    xy=(mid, np.mean(rsl1[np.isin(lmd, peak_wavenumbers_)])),
                    ha='center', va='bottom', fontsize=8)

plt.xlabel('波数 (cm-1)')
plt.ylabel('反射率 (%)')
plt.title(f'峰值间隔分析 (平均间隔: {avg_interval:.2f} cm⁻¹)' if len(peak_wavenumbers_)>=2 else '峰值检测')
plt.legend()
plt.grid(True)
plt.show()
