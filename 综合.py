
import pandas as pd
# 88分组
csv_path = "/home/mtftau-5/workplace/数模校赛/segment_times.csv"
xlsx_path = "/home/mtftau-5/workplace/25中北校赛/2025B土壤普查/附件1：xx地区.xlsx"

df = pd.read_csv(csv_path)
df[['from_node', 'to_node']] = df['路段'].str.split('→', expand=True).astype(int)
df_work = pd.read_excel(
    xlsx_path, 
    usecols=[0, 3], 
    names=['node_id', 'work_min'],
    header=0
)
work_dict = (df_work.set_index('node_id')['work_min'] / 60).to_dict()


group_size = 8
skip_size = 1  
groups = []
current_group = []
for start_idx in range(0, len(df), group_size + skip_size):
    end_idx = start_idx + group_size
    group = df.iloc[start_idx:end_idx]
    

    if len(group) == 0:
        continue
    total_time = 0.0
    path_segments = []
    for _, row in group.iloc[:-1].iterrows():
        total_time += row['时间(小时)'] + work_dict.get(row['from_node'], 0)
        path_segments.append(f"{row['from_node']}→{row['to_node']}")
    

    if len(group) >= 1:
        last_row = group.iloc[-1]
        total_time += last_row['时间(小时)']
        total_time += work_dict.get(last_row['from_node'], 0) 
        total_time += work_dict.get(last_row['to_node'], 0)  
        path_segments.append(f"{last_row['from_node']}→{last_row['to_node']}")

    groups.append({
        '组ID': len(groups) + 1,
        '起始节点': group.iloc[0]['from_node'],
        '结束节点': group.iloc[-1]['to_node'],
        '实际行数': len(group),
        '总时间(小时)': round(total_time, 4),
        '路径': " → ".join(path_segments)
    })


result_df = pd.DataFrame(groups)
output_path = "skip_groups.csv"
result_df.to_csv(output_path, index=False, encoding='utf_8_sig')

print("分组结果示例：")
print(result_df.head().to_string(index=False))
print(f"\n完整结果已保存至: {output_path}")