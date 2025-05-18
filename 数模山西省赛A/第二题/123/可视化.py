import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zhplot

df = pd.read_excel("/home/mtftau-5/workplace/数模/输出文件/电站2最大发电量与辐照值（最小二乘拟合）.xlsx")

df['日期'] = pd.to_datetime(df['日期'])
df['比值'] = pd.to_numeric(df['比值'], errors='coerce')
df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')
df['日期'] = pd.to_datetime(df['日期'])

plt.figure(figsize=(10, 6))
plt.plot(df['日期'], df['比值'], marker='o', linestyle='-', color='b')
plt.title('Power Output Ratio Over Time')
plt.xlabel('Date')
plt.ylabel('Power Output Ratio')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
