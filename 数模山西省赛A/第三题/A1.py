import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from scipy.integrate import quad
import zhplot

W_sun_max = 1000
sun_rise = 7
sun_set = 18
alpha = 0.0009
eta = 0.0004
q = 0.22
C = 2 * 5000
P = 0.5
w1 = 0.4
w2 = 0.4
w3 = 0.2

def calculate_total_energy(sun_rise, sun_set, W_sun_max):
    E_total, _ = quad(lambda t: W_sun_max * np.sin((t - sun_rise) * np.pi / (sun_set - sun_rise)), sun_rise, sun_set)
    return E_total / 1000

def calculate_log_Wk(log_Wk_prev, alpha, eta, q, E_total_kwh):
    Wk_prev = np.exp(log_Wk_prev)
    delta = (alpha * eta * Wk_prev**2) / (q * E_total_kwh)
    return np.log(max(Wk_prev - delta, 1e-10))

def generate_Wk_sequence(m_days, alpha, eta, q, E_total_kwh):
    log_Wk = np.log(20000)
    sequence = [np.exp(log_Wk)]
    for _ in range(m_days):
        log_Wk = calculate_log_Wk(log_Wk, alpha, eta, q, E_total_kwh)
        sequence.append(np.exp(log_Wk))
    return sequence

def objective(m, alpha, eta, q, C, P, w1, w2, w3, E_total_kwh):
    m_int = int(np.round(m[0]))
    Wk_sequence = generate_Wk_sequence(m_int, alpha, eta, q, E_total_kwh)
    total_revenue = P * sum(Wk_sequence[:-1])
    delta_A = P * (Wk_sequence[0] - Wk_sequence[-1])
    cycle_cost = C

    if delta_A <= 0:
        print("Error: delta_A <= 0, optimization may not be valid.")
        return np.inf

    K = cycle_cost / delta_A
    avg_profit = (total_revenue - cycle_cost) / m_int

    return -(
        w1 * (1/K) +
        w2 * (1/m_int) +
        w3 * avg_profit
    )

E_total_kwh = calculate_total_energy(sun_rise, sun_set, W_sun_max)

bounds = [(1, 365)]
result = differential_evolution(
    objective,
    bounds=bounds,
    args=(alpha, eta, q, C, P, w1, w2, w3, E_total_kwh)
)

if result.success:
    optimal_m = int(np.round(result.x[0]))
    Wk_sequence = generate_Wk_sequence(optimal_m, alpha, eta, q, E_total_kwh)
    delta_A = P * (Wk_sequence[0] - Wk_sequence[-1])
    K = C / delta_A
    avg_profit = (P * sum(Wk_sequence[:-1]) - C) / optimal_m
    total_profit = avg_profit * optimal_m

    print(f"最优清理周期: {optimal_m} 天")
    print(f"回本天数: {K:.1f} 天")
    print(f"周期平均日利润: {avg_profit:.2f} 元")
    print(f"总利润: {total_profit:.2f} 元")
    print(f"周期末效率: {Wk_sequence[-1]:.2f} W")

    plt.figure(figsize=(10, 4))
    plt.plot(Wk_sequence, 'r-', linewidth=2)
    plt.title(f'{optimal_m}天清理周期效率衰减曲线')
    plt.xlabel('天数')
    plt.ylabel('发电效率(W)')
    plt.grid(True)
    plt.show()
else:
    print("优化失败:", result.message)