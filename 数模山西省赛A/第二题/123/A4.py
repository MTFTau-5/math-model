import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import zhplot

def compare(value, i):
    before_value = []
    after_value = []
    before = 0
    after = 0
    before_value_max = value[i+1]
    before_value_min = value[i+1]
    after_value_max = value[i+1]
    after_value_min = value[i+1]
    
    for j in range(i+1, i+8):
        after += value[j]
        after_value.append(value[j])
        if value[j] > after_value_max:
            after_value_max = value[j]
        if value[j] < after_value_min:
            after_value_min = value[j]
    
    for j in range(i-6, i+1):
        before += value[j]
        before_value.append(value[j])
        if value[j] > before_value_max:
            before_value_max = value[j]
        if value[j] < before_value_min:
            before_value_min = value[j]
    
    k = (after-after_value_max-after_value_min)/5 - (before-before_value_max-before_value_min)/5
    return (before-before_value_max-before_value_min)/5, (after-after_value_max-after_value_min)/5, k

# Load data
df = pd.read_excel('/home/mtftau-5/workplace/数模/输出文件/电站4最大发电量与辐照值比值（最小二乘法拟合）.xlsx')
df = df[['日期', '比值']]
df['日期'] = pd.to_datetime(df['日期'])
df['比值'] = df['比值'].astype(float)

day = df['日期']
value = df['比值']
new_value = []

for i in range(6, len(value)-7):
    new_value.append(compare(value, i)[2])

# output_path = '/home/mtftau-5/workplace/数模/输出文件/电站1最大发电量与辐照值（最小二乘拟合）.xlsx'
# df = pd.read_excel(output_path)
# new_value = np.array(new_value)

df1 = pd.DataFrame({'日期': day[6:len(value)-7], '比值': new_value})
print(df1)

plt.figure(figsize=(12, 6))
plt.plot(df1['日期'], df1['比值'], label='拟合结果')
plt.scatter(df1['日期'], df1['比值'], color='red', s=10, label='原始数据')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=df1['比值'].mean(), color='g', linestyle='--', label='平均值')

plt.legend(prop={'size': 12})
plt.xlabel('日期', fontsize=12)
plt.ylabel('比值', fontsize=12)
plt.title('清洁指数比值残差拟合结果', fontsize=14)
plt.tight_layout()
plt.show()