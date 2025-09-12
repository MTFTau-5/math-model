import pandas as pd
import matplotlib.pyplot as plt
import zhplot
data = pd.read_excel("math-model-main\B题(1)\B题\附件\附件2.xlsx")
lmd = data['波数 (cm-1)']
rlf = data['反射率 (%)']
plt.plot(lmd, rlf)
plt.xlabel('波数 (cm-1)')
plt.ylabel('反射率 (%)')
plt.title('附件2反射率随波数变化曲线')
plt.grid()
plt.show()