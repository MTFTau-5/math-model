import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('/home/mtftau-5/workplace/2025赛题/2025赛题/2025_A题/附件/电站1发电数据.xlsx')
print(df)

time_column = '当日累计发电量kwh'
# 假设该列是数值类型，计算该列的总和
a = df[time_column].sum()
print(a)