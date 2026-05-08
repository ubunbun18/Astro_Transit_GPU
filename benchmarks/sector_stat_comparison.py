import numpy as np
from scipy.stats import beta

def jeffreys_interval(k, n, confidence=0.95):
    if n == 0: return 0, 0, 0
    alpha = 1 - confidence
    lower = beta.ppf(alpha/2, k + 0.5, n - k + 0.5)
    upper = beta.ppf(1 - alpha/2, k + 0.5, n - k + 0.5)
    return k/n, lower, upper

def compare_sectors_statistically():
    # Sector 1 Results (Confirmed)
    s1_n = 156
    s1_k = 27 # 17.31% * 156
    
    # Sector 2 Results (Confirmed with relaxed T0)
    s2_n = 20
    s2_k = 12 # 60%
    
    print("# Priority 2: Statistical Consistency (S1 vs S2)")
    print("\n| Sector | Sample (N) | Recovered (k) | Completeness | 95% CI (Jeffreys) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for label, n, k in [("Sector 1", s1_n, s1_k), ("Sector 2", s2_n, s2_k)]:
        rate, low, high = jeffreys_interval(k, n)
        print(f"| {label} | {n} | {k} | {rate:.1%} | [{low:.1%}, {high:.1%}] |")

    rate1, low1, high1 = jeffreys_interval(s1_k, s1_n)
    rate2, low2, high2 = jeffreys_interval(s2_k, s2_n)
    
    overlap = not (high1 < low2 or high2 < low1)
    
    print(f"\nOverlap Status: **{'Overlapping' if overlap else 'NOT Overlapping'}**")
    
    if not overlap:
        print("\n⚠️ 信頼区間が重なっていません。これは統計的な誤差だけでなく、")
        print("「セクター2のサンプルがセクター1よりも物理的に検出しやすい（深い・周期が短い）天体に偏っていた」")
        print("ことを示唆しています。")
    else:
        print("\n✅ 信頼区間が重なっており、両セクターの結果は統計的に矛盾しません。")

if __name__ == "__main__":
    compare_sectors_statistically()
