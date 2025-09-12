import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal

data = pd.read_excel(r'math-model-main\B题(1)\B题\附件\附件3.xlsx')
lmd = data['波数 (cm-1)'].values
rsl = data['反射率 (%)'].values
mask = (lmd >= 1200) & (lmd <= lmd.max())
lmd = lmd[mask]
rsl = rsl[mask]
R = np.clip(rsl / 100, 1e-6, 0.999)
CCB = 1e7 / lmd  # 波长 (nm)
c = 2.99792458e8
omg = 2 * np.pi * c / (CCB * 1e-9)
sort_idx = np.argsort(omg)
omg = omg[sort_idx]
R = R[sort_idx]
CCB = CCB[sort_idx]

# 平滑反射率数据
R_smooth = savgol_filter(R, window_length=11, polyorder=3)
diangun = interp1d(omg, R_smooth, kind='cubic', bounds_error=False, fill_value=0)
luguan = np.log(np.sqrt(R_smooth))
theta = -np.imag(scipy.signal.hilbert(luguan))
sqrt_R = np.sqrt(R_smooth)
cos_theta = np.cos(theta)
n1 = (1 - R_smooth) / (1 + R_smooth - 2 * sqrt_R * cos_theta)
n2 = (1 + sqrt_R) / (1 - sqrt_R)
kappa = 2 * sqrt_R * np.sin(theta) / (1 + R_smooth - 2 * sqrt_R * cos_theta)

# 波长筛选（单晶硅透明区域）
mask = (CCB >= 400) & (CCB <= 1100)
CCB_filtered = CCB[mask]
n1_filtered = n1[mask]
n2_filtered = n2[mask]
kappa_filtered = kappa[mask]
print(f"筛选后波长范围: {np.min(CCB_filtered):.2f} - {np.max(CCB_filtered):.2f} nm")
print(f"筛选后数据点数: {len(CCB_filtered)}")
print(f"kappa 范围: {np.min(kappa_filtered):.4f} - {np.max(kappa_filtered):.4f}")

class CauchyDispersionValidator:
    def __init__(self, wavelength_nm, refractive_index, kappa=None):
        self.wavelength_nm = np.array(wavelength_nm)
        self.refractive_index = np.array(refractive_index)
        self.kappa = np.array(kappa) if kappa is not None else None
        self.wavelength_um = self.wavelength_nm / 1000
        self.results = {'2param': None, '3param': None}

    @staticmethod
    def _cauchy_2param(λ, A, B):
        return A + B / (λ**2)

    @staticmethod
    def _cauchy_3param(λ, A, B, C):
        return A + B / (λ**2) + C / (λ**4)

    def fit(self, max_order=3, transparency_threshold=0.01):
        if self.kappa is not None:
            mask = self.kappa < transparency_threshold
            λ_fit = self.wavelength_um[mask]
            n_fit = self.refractive_index[mask]
            print(f"使用透明区域数据点: {sum(mask)}/{len(mask)} (κ<{transparency_threshold})")
        else:
            λ_fit = self.wavelength_um
            n_fit = self.refractive_index

        p0_2param = [3.6, 0.12]
        p0_3param = [3.6, 0.12, 0.008]
        bounds_2param = ([3.0, 0.0], [4.0, 0.3])
        bounds_3param = ([3.0, 0.0, -0.03], [4.0, 0.3, 0.03])

        if max_order >= 2:
            try:
                popt, pcov = curve_fit(self._cauchy_2param, λ_fit, n_fit, p0=p0_2param, bounds=bounds_2param)
                perr = np.sqrt(np.diag(pcov))
                self.results['2param'] = {'coeffs': popt, 'errors': perr, 'model': self._cauchy_2param}
                print(f"2参数拟合成功: A={popt[0]:.4f}, B={popt[1]:.4f} μm²")
            except Exception as e:
                print(f"2参数拟合失败: {e}")

        if max_order >= 3:
            try:
                popt, pcov = curve_fit(self._cauchy_3param, λ_fit, n_fit, p0=p0_3param, bounds=bounds_3param)
                perr = np.sqrt(np.diag(pcov))
                self.results['3param'] = {'coeffs': popt, 'errors': perr, 'model': self._cauchy_3param}
                print(f"3参数拟合成功: A={popt[0]:.4f}, B={popt[1]:.4f} μm², C={popt[2]:.6f} μm⁴")
            except Exception as e:
                print(f"3参数拟合失败: {e}")

    def evaluate(self):
        if self.kappa is not None:
            mask = self.kappa < 0.01
            λ_eval = self.wavelength_um[mask]
            n_eval = self.refractive_index[mask]
        else:
            λ_eval = self.wavelength_um
            n_eval = self.refractive_index

        for order, result in self.results.items():
            if result is None:
                continue
            n_pred = result['model'](λ_eval, *result['coeffs'])
            ss_res = np.sum((n_eval - n_pred)**2)
            ss_tot = np.sum((n_eval - np.mean(n_eval))**2)
            r_squared = 1 - (ss_res / ss_tot)
            residuals = n_eval - n_pred
            std_dev = np.std(residuals)
            result['r_squared'] = r_squared
            result['residual_std'] = std_dev

            print(f"\n{order}模型结果:")
            print(f"柯西系数: {result['coeffs']}")
            print(f"系数误差: {result['errors']}")
            print(f"R² = {r_squared:.6f}")
            print(f"残差标准差 = {std_dev:.6f}")
            if r_squared > 0.95:
                print("模型质量: 优秀 (R² > 0.95)")
            elif r_squared > 0.9:
                print("模型质量: 良好 (0.9 ≤ R² ≤ 0.95)")
            else:
                print("模型质量: 一般或较差 (R² < 0.9)")

    def plot_results(self):
        plt.figure(figsize=(14, 6))
        ax1 = plt.subplot(1, 2, 1)
        if self.kappa is not None:
            sc = ax1.scatter(self.wavelength_nm, self.refractive_index, 
                            c=self.kappa, cmap='viridis', vmin=0, vmax=0.05, s=20, alpha=0.7, label='数据点(颜色=κ)')
            plt.colorbar(sc, label='消光系数 κ')
        else:
            ax1.scatter(self.wavelength_nm, self.refractive_index, s=20, alpha=0.7, label='数据点')

        colors = ['red', 'blue']
        for i, (order, result) in enumerate(self.results.items()):
            if result is None:
                continue
            n_fit = result['model'](self.wavelength_um, *result['coeffs'])
            ax1.plot(self.wavelength_nm, n_fit, color=colors[i], linewidth=2,
                     label=f'{order}模型 (R²={result["r_squared"]:.3f})')

        ax1.set_xlabel('波长 (nm)')
        ax1.set_ylabel('折射率 n')
        ax1.set_title('单晶硅 柯西色散拟合结果')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(1, 2, 2)
        for i, (order, result) in enumerate(self.results.items()):
            if result is None:
                continue
            n_pred = result['model'](self.wavelength_um, *result['coeffs'])
            residuals = self.refractive_index - n_pred
            ax2.scatter(self.wavelength_nm, residuals, color=colors[i], s=20, alpha=0.7,
                        label=f'{order}模型残差')

        ax2.axhline(0, color='black', linestyle='--')
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('残差 (n_data - n_fit)')
        ax2.set_title('拟合残差分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# 运行验证
print("\n" + "="*50)
print("柯西色散关系验证 - 算法1 (含相位信息)")
print("="*50)
validator1 = CauchyDispersionValidator(CCB_filtered, n1_filtered, kappa_filtered)
validator1.fit(max_order=3, transparency_threshold=0.01)
validator1.evaluate()
validator1.plot_results()

print("\n" + "="*50)
print("柯西色散关系验证 - 算法2 (透明近似)")
print("="*50)
validator2 = CauchyDispersionValidator(CCB_filtered, n2_filtered)
validator2.fit(max_order=3)
validator2.evaluate()
validator2.plot_results()