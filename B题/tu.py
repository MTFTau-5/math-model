import pandas as pd
import numpy as np

# 修正路径
try:
    data = pd.read_excel(r'math-model-main\B题(1)\B题\附件\附件3.xlsx')
    print("数据加载成功，前5行数据：")
    print(data.head())
    
    # 假设波长数据在'波长'列，CCB数据在'CCB'列
    wavelength = data['波长'].values
    CCB = data['CCB'].values
    
    # 筛选条件 - 示例：筛选CCB值大于0.5的数据
    threshold = 0.5
    CCB_filtered = wavelength[CCB > threshold]
    
    if len(CCB_filtered) > 0:
        print(f"筛选后波长范围: {np.min(CCB_filtered):.2f} - {np.max(CCB_filtered):.2f} nm")
        print(f"筛选后数据点数: {len(CCB_filtered)}")
    else:
        print("警告：筛选后没有数据，请检查筛选条件或数据质量")
        print("原始数据统计：")
        print(data.describe())
        
except FileNotFoundError:
    print("错误：文件未找到，请检查路径")
except Exception as e:
    print(f"发生错误: {str(e)}")