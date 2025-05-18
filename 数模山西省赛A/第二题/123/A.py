import pandas as pd
import matplotlib.pyplot as plt
import zhplot

end_year = 2025
end_month = 2
end_day = 15
start_year = 2024
start_month = 5
start_day = 1

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_days_in_month(year, month):
    year_list = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if month == 2 and is_leap_year(year):
        return 29
    return year_list[month]

file_path1 = '/home/mtftau-5/workplace/2025赛题/2025赛题/2025_A题/附件/电站4发电数据.xlsx'
file_path2 = '/home/mtftau-5/workplace/数模/2文件/电站4辐照值.xlsx'

df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)

time_column = '时间'

df1[time_column] = pd.to_datetime(df1[time_column], errors='coerce')
df2[time_column] = pd.to_datetime(df2[time_column], errors='coerce')

output_data = []

while True:
    start_date_str = f'{start_year}-{start_month:02d}-{start_day:02d} 00:00:00'
    start_date = pd.to_datetime(start_date_str)

    end_date_str = f'{start_year}-{start_month:02d}-{start_day:02d} 23:59:59'
    end_date = pd.to_datetime(end_date_str)

    filtered_df1 = df1[(df1[time_column] >= start_date) & (df1[time_column] <= end_date)]
    filtered_df2 = df2[(df2[time_column] >= start_date) & (df2[time_column] <= end_date)]

    filtered_df1['来源'] = '电站1环境检测仪数据'
    filtered_df2['来源'] = '辐照数据（已清洗）'

    max_value = filtered_df1['当日累计发电量kwh'].max()
    total_radiation = filtered_df2['辐照强度'].sum()

    if total_radiation != 0:
        ratio = max_value / total_radiation
        print(f"日期: {start_date_str}, 最大发电量与辐照值比值: {ratio}")
        output_data.append({'日期': start_date.date(), '比值': ratio})
    else:
        print(f"日期: {start_date_str}, 辐照值为零，无法计算比值")
        output_data.append({'日期': start_date.date(), '比值': None})

    start_day += 1
    if start_day > get_days_in_month(start_year, start_month):
        start_day = 1
        start_month += 1
        if start_month > 12:
            start_month = 1
            start_year += 1

    if start_year > end_year or (start_year == end_year and start_month > end_month) or (
            start_year == end_year and start_month == end_month and start_day > end_day):
        break


output_df = pd.DataFrame(output_data)

max_value = output_df['比值'].max()
filtered_df = output_df[output_df['比值'] != max_value]

output_file_path = '/home/mtftau-5/workplace/数模/输出文件/电站4最大发电量与辐照值比值.xlsx'
filtered_df.to_excel(output_file_path, index=False)

print(f"数据已保存到 {output_file_path}")


plt.xlabel('时间')
plt.ylabel('最大发电量与辐照值比值')
plt.title('清洁指数随时间变化图')
plt.plot(filtered_df['日期'], filtered_df['比值'], 'ro-')
plt.show()