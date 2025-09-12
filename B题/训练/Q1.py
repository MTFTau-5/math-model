import pandas as pd
from matplotlib.pyplot import plot, show
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import zhplot
data = pd.read_excel("/home/mtftau-5/workplace/B题/附件/附件3.xlsx")
n = data['波数 (cm-1)'].values
x = 1e7 / n
y = data['反射率 (%)'].values
fft_y = ifft(y)
plt.plot(x, abs(fft_y))
plt.xlabel('波长 (nm)')
plt.ylabel('FFT幅值')
plt.title('反射率的FFT频谱')
plt.grid()
plt.show()


