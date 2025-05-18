import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zhplot
def least_squares_fit(filepath, output_filepath, degree=3):
    df = pd.read_excel(filepath)
    data = df[['日期', '比值']].copy()
    
    data['x'] = range(1, len(data)+1)
    
    known = data.dropna(subset=['比值'])
    missing = data[data['比值'].isnull()]
    
    if len(known) < degree + 1:
        raise ValueError(f"至少需要 {degree+1} 个数据点进行{degree}次多项式拟合")
    coeff = np.polyfit(known['x'], known['比值'], deg=degree)
    poly_func = np.poly1d(coeff)
    data.loc[missing.index, '比值'] = poly_func(missing['x'])

    mean_value = data['比值'].mean()
    print(f"填充后的比值平均值为: {mean_value}")
    data.drop(columns=['x']).to_excel(output_filepath, index=False)
    print(f"成功填充{len(missing)}个缺失值，保存至: {output_filepath}")

input_path = "/home/mtftau-5/workplace/数模/输出文件/电站2最大发电量与辐照值（最小二乘拟合）.xlsx"
output_path = "/home/mtftau-5/workplace/数模/输出文件/电站2最大发电量与辐照值（最小二乘拟合）.xlsx"
least_squares_fit(input_path, output_path, degree=3)

df = pd.read_excel(output_path)
plt.plot(df['日期'], df['比值'], label='拟合结果')
plt.scatter(df['日期'], df['比值'], color='red', label='原始数据')
plt.xticks(rotation=45)
plt.grid()
plt.axhline(y=df['比值'].mean(), color='g', linestyle='--', label='平均值')
plt.legend()
plt.xlabel('日期')
plt.ylabel('比值')
plt.title('清洁指数比值拟合结果')
plt.show()
