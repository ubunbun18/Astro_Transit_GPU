import pandas as pd
import numpy as np
from scipy.stats import beta

def calculate_completeness_with_ci(k, n, alpha=0.05):
    """Jeffreys信頼区間の計算"""
    if n == 0: return 0, 0, 0
    lower = beta.ppf(alpha/2, k + 0.5, n - k + 0.5)
    upper = beta.ppf(1 - alpha/2, k + 0.5, n - k + 0.5)
    return k/n, lower, upper

def generate_rigorous_sensitivity_map():
    df = pd.read_csv("outputs/injection_results.csv")
    
    # 周期ビン (days)
    p_bins = [0.5, 2, 5, 10, 13.7]
    # 深さビン (ppm)
    d_bins = [500, 1000, 2500, 5000, 10000]
    
    print("# Rigorous Sensitivity Map (with 95% Jeffreys CI)")
    print("\n| Depth \\ Period | 0.5-2d | 2-5d | 5-10d | 10-13.7d |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for j in range(len(d_bins)-1):
        d_low, d_high = d_bins[j], d_bins[j+1]
        row_str = f"| {d_low}-{d_high} ppm |"
        
        for i in range(len(p_bins)-1):
            p_low, p_high = p_bins[i], p_bins[i+1]
            
            mask = (df['p_inj'] >= p_low) & (df['p_inj'] < p_high) & \
                   (df['depth_inj'] >= d_low) & (df['depth_inj'] < d_high)
            
            subset = df[mask]
            n = len(subset)
            k = subset['recovered'].sum()
            
            rate, low, high = calculate_completeness_with_ci(k, n)
            
            if n > 0:
                row_str += f" {rate:.1%} [{low:.1%}, {high:.1%}] |"
            else:
                row_str += " - |"
        print(row_str)

if __name__ == "__main__":
    generate_rigorous_sensitivity_map()
