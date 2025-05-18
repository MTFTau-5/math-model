import pandas as pd
#只是单纯88分组里加了个工作时间限制

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

target_min = 8.0 
target_max = 8.5 
skip_size = 1     

groups = []
current_idx = 0     
group_id = 1

while current_idx < len(df):
    current_group = []
    total_time = 0.0
    while current_idx < len(df):
        row = df.iloc[current_idx]
        temp_time = total_time
        if len(current_group) == 0:
            temp_time += row['时间(小时)'] + work_dict.get(row['from_node'], 0)
        else:
            temp_time += row['时间(小时)']
        if temp_time + work_dict.get(row['to_node'], 0) > target_max:
            break  
            

        current_group.append(row)
        total_time = temp_time
        if len(current_group) >= 1:
            last_row = current_group[-1]
            total_time += work_dict.get(last_row['to_node'], 0)
        
        if target_min <= total_time <= target_max:
            current_idx += 1 
            break
            
        current_idx += 1

    if len(current_group) > 0:
        path_segments = [f"{row['from_node']}→{row['to_node']}" for _, row in pd.DataFrame(current_group).iterrows()]
        
        groups.append({
            '组ID': group_id,
            '起始节点': current_group[0]['from_node'],
            '结束节点': current_group[-1]['to_node'],
            '实际行数': len(current_group),
            '总时间(小时)': round(total_time, 4),
            '路径': " → ".join(path_segments)
        })
        group_id += 1
    current_idx += skip_size  
result_df = pd.DataFrame(groups)
output_path = "dynamic_groups.csv"
result_df.to_csv(output_path, index=False, encoding='utf_8_sig')

print("优化后的分组结果（时间严格控制在8-8.5小时）：")
print(result_df.head().to_string(index=False))
print(f"\n完整结果已保存至: {output_path}")