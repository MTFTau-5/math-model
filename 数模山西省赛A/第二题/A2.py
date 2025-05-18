import pandas as pd

file_path1 = '/home/chenyifeng/workspace/2025_A题/附件/电站1发电数据.xlsx'
file_path2 = '/home/chenyifeng/workspace/数模/输出文件/辐照数据（已清洗）.xlsx'
df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)

time_column = '时间'
df1[time_column] = pd.to_datetime(df1[time_column], errors='coerce')
df2[time_column] = pd.to_datetime(df2[time_column], errors='coerce')

start_data = '2024-11-17 00:00:00'
end_data = '2024-11-17 23:59:59'

filtered_df1 = df1[(df1[time_column] >= start_data) & (df1[time_column] <= end_data)]
filtered_df2 = df2[(df2[time_column] >= start_data) & (df2[time_column] <= end_data)]
filtered_df1['来源'] = '电站1环境检测仪数据'
filtered_df2['来源'] = '辐照数据（已清洗）'

max_value = filtered_df1['当日累计发电量kwh'].max()
total_radiation = filtered_df2['辐照强度'].sum()

clean_index = max_value / total_radiation
print(f"日期: {start_data}, 最大发电量与辐照值比值: {clean_index}")