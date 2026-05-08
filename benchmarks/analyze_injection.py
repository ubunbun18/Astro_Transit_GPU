import pandas as pd
import numpy as np

def generate_sensitivity_map():
    df = pd.read_csv("outputs/injection_results.csv")
    
    # 周期ビン (days)
    p_bins = [0.5, 2, 5, 10, 13.7]
    # 深さビン (ppm)
    d_bins = [500, 1000, 2500, 5000, 10000]
    
    print("# Sensitivity Map (Completeness by Period and Depth)")
    print("\n| Depth \\ Period | 0.5-2d | 2-5d | 5-10d | 10-13.7d |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for j in range(len(d_bins)-1):
        d_low, d_high = d_bins[j], d_bins[j+1]
        row_str = f"| {d_low}-{d_high} ppm |"
        
        for i in range(len(p_bins)-1):
            p_low, p_high = p_bins[i], p_bins[i+1]
            
            mask = (df['p_inj'] >= p_low) & (df['p_inj'] < p_high) & \
                   (df['depth_inj'] >= d_low) & (df['depth_inj'] < d_high)
            
            in_bin = df[mask]
            if len(in_bin) > 0:
                rate = in_bin['recovered'].mean()
                row_str += f" {rate:.1%} |"
            else:
                row_str += " - |"
        print(row_str)

    # 周期精度の再確認 (注入データ版)
    recovered = df[df['recovered']]
    p_err = np.abs(recovered['period'] - recovered['p_inj']) / recovered['p_inj'] * 100
    print(f"\n**Period Precision (on recovered injections)**:")
    print(f"- Median Error: {p_err.median():.4f}%")
    print(f"- 95th Percentile: {p_err.quantile(0.95):.4f}%")

if __name__ == "__main__":
    generate_sensitivity_map()
