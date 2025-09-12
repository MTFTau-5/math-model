import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zhplot

data = pd.read_excel('B题\B题\附件\附件3.xlsx')
lmd = data['波数 (cm-1)'].values
rsl = data['反射率 (%)'].values

# 应用波数范围过滤（1200 cm⁻¹到最大值）
# mask = (lmd >= 1200) & (lmd <= lmd.max())
# lmd = lmd[mask]
# rsl = rsl[mask]

# 计算波数间隔
delta_lmd = np.mean(np.diff(lmd))
N = len(lmd)

# 执行FFT
fft_result = np.fft.fft(rsl)
fft_freq = np.fft.fftfreq(N, d=delta_lmd)  # 频率单位：cm⁻¹

# 计算振幅和相位
amplitude = np.abs(fft_result) / N * 2
amplitude[0] /= 2  # DC分量
phase = np.angle(fft_result)

# 选择前5个主要频率成分（按振幅排序）
sorted_indices = np.argsort(amplitude[:N//2])[::-1]
top5_indices = sorted_indices[0:5]

# 计算各成分的波数周期（单位：cm⁻¹）
print("前5个主要正弦成分的波数周期：")
print("=" * 60)
for i, idx in enumerate(top5_indices, 1):
    freq = abs(fft_freq[idx])  # 频率 (cm⁻¹)
    period = 1 / freq if freq != 0 else np.inf  # 波数周期
    
    print(f"成分 {i}:")
    print(f"  频率: {freq:.4f} cm⁻¹")
    print(f"  波数周期: {period:.4f} cm⁻¹") 
    print(f"  振幅: {amplitude[idx]:.4f}")
    print(f"  相位: {phase[idx]:.4f} rad")
    print("-" * 40)

# 重建各正弦成分
reconstructed_components = []
for i in top5_indices:
    freq = fft_freq[i]
    amp = amplitude[i]
    phi = phase[i]
    component = amp * np.cos(2 * np.pi * freq * lmd + phi)
    reconstructed_components.append((freq, amp, phi, component))

# 绘制结果
plt.figure(figsize=(14, 10))

# 1. 原始反射率光谱
plt.subplot(3, 1, 1)
plt.plot(lmd, rsl, 'b-', label='原始信号', linewidth=1.5)
plt.title('原始反射率光谱')
plt.xlabel('波数 (cm⁻¹)')
plt.ylabel('反射率 (%)')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. 主要正弦成分（前5个）
plt.subplot(3, 1, 2)
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, (freq, amp, phi, component) in enumerate(reconstructed_components):
    plt.plot(lmd, component, color=colors[i], 
             label=f'频率: {abs(freq):.2f} cm⁻¹, 振幅: {amp:.2f}')
plt.title('主要正弦成分')
plt.xlabel('波数 (cm⁻¹)')
plt.ylabel('振幅')
plt.grid(True, alpha=0.3)
plt.legend()

# 3. 重建信号对比
plt.subplot(3, 1, 3)
reconstructed_signal = amplitude[0]/2  # DC分量
for _, _, _, component in reconstructed_components:
    reconstructed_signal += component

plt.plot(lmd, rsl, 'b-', alpha=0.5, label='原始信号', linewidth=2)
plt.plot(lmd, reconstructed_signal, 'r-', label='重建信号', linewidth=1.5)
plt.title('原始信号与重建信号对比')
plt.xlabel('波数 (cm⁻¹)')
plt.ylabel('反射率 (%)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
