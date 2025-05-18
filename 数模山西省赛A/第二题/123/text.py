import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def compare(value,i):
    before_value = []
    after_value = []
    before = 0
    after = 0
    before_value_max0 = value[i+1]
    before_value_max1 = value[i+1]
    before_value_min0 = value[i+1]
    before_value_min1 = value[i+1]
    after_value_max0 = value[i+1]
    after_value_max1 = value[i+1]
    after_value_min0 = value[i+1]
    after_value_min1 = value[i+1]
    for j in range(i+1,i+10):
        after+=value[j]
        after_value.append(value[j])
        if value[j]>after_value_max0:
            after_value_max1 = after_value_max0
            after_value_max0 = value[j]
        elif value[j]>after_value_max1:
            after_value_max1 = value[j]
        if value[j]<after_value_min0:
            after_value_min1 = after_value_min0
            after_value_min0 = value[j]
        elif value[j]<after_value_min1:
            after_value_min1 = value[j]
    for j in range(i-9,i+1):
        before+=value[j]
        before_value.append(value[j])
        if value[j]>before_value_max0:
            before_value_max1 = before_value_max0
            before_value_max = value[j]
        elif value[j]>before_value_max1:
            before_value_max1 = value[j]
        if value[j]<before_value_min0:
            before_value_min1 = before_value_min0
            before_value_min = value[j]
        elif value[j]<before_value_min1:
            before_value_min1 = value[j]
    k = (after-after_value_max0-after_value_min0-after_value_max1-after_value_min1)/5-(before-before_value_max0-before_value_min0-before_value_max1-before_value_min1)/5
    return (after-after_value_max0-after_value_min0-after_value_max1-after_value_min1)/5, (before-before_value_max0-before_value_min0-before_value_max1-before_value_min1)/5,k
df = pd.read_excel('/home/mtftau-5/workplace/数模/输出文件/电站1最大发电量与辐照值（最小二乘拟合）.xlsx')
df = df[['日期', '比值']]
df['日期'] = pd.to_datetime(df['日期'])
df['比值'] = df['比值'].astype(float)

day = df['日期']
value = df['比值']
new_value = []
for i in range(9,len(value)-10):
    new_value.append(compare(value,i)[2])
    
output_path = '/home/mtftau-5/workplace/数模/输出文件/电站1最大发电量与辐照值（最小二乘拟合）.xlsx'
df = pd.read_excel(output_path)
new_value = np.array(new_value)

df1 = pd.DataFrame({'日期': day[9:len(value)-10], '比值': new_value})
print(df1)
plt.plot(df1['日期'], df1['比值'], label='拟合结果')
plt.scatter(df1['日期'], df1['比值'], color='red', label='原始数据')
plt.xticks(rotation=45)
plt.grid()
plt.axhline(y=df1['比值'].mean(), color='g', linestyle='--', label='平均值')
plt.legend()
plt.xlabel('日期')
plt.ylabel('比值')
plt.title('清洁指数比值拟合结果')
plt.show()