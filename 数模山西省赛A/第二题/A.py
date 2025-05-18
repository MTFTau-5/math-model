import pandas as pd

file_path1 = '/home/mtftau-5/workplace/2025赛题/2025赛题/2025_A题/附件/电站1发电数据.xlsx' 
file_path2 = ''
df = pd.read_excel(file_path1)
time_column = '时间'

df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
start_date = pd.to_datetime('2024-06-01')
end_date = pd.to_datetime('2024-06-30')

filtered_df = df[(df[time_column] >= start_date) & (df[time_column] <= end_date)]

output_path = 'filtered_data.xlsx'
filtered_df.to_excel(output_path, index=False)

print(f"数据已保存至：{output_path}")