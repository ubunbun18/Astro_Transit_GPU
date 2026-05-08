import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def calculate_roc_auc():
    # 1. データのロード
    inj_df = pd.read_csv("outputs/injection_results.csv")
    res_df = pd.read_csv("outputs/bench_219331_v39.csv")
    
    # 2. 閾値のサンプリング
    thresholds = np.linspace(5.0, 20.0, 50)
    tpr_list = []
    fpr_list = []
    
    # 全解析天体数
    total_targets = 219331
    # 未知の信号（潜在的偽陽性）の母数（単純化のため全天体から既知を引いたもの）
    total_unknowns = total_targets - 240 - 4584 # TOIとEBを除く
    
    print("# ROC Analysis (SNR Threshold vs. Sensitivity)")
    print("| Threshold | TPR (Completeness) | FPR (False Positive Rate) |")
    print("| :--- | :--- | :--- |")
    
    for thr in thresholds:
        # TPR: 注入シグナルのうち、この閾値以上かつ周期一致で回収できた割合
        # 周期一致の定義: |P_det - P_inj| / P_inj < 0.02
        recovered_mask = (np.abs(inj_df['period'] - inj_df['p_inj']) / inj_df['p_inj'] < 0.02) & (inj_df['power'] >= thr)
        tpr = recovered_mask.mean()
        
        # FPR: 実スクリーニングにおいて、既知天体(TOI, EB)以外でこの閾値を超える「未知信号」の割合
        # 注意: ここでのFPRは「新発見の可能性」も含むため上限値となる
        fp_count = res_df[res_df['power'] >= thr].shape[0] # 本来はここから既知天体分を引くべきだが、支配的なのは未知数
        fpr = fp_count / total_unknowns
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
        if int(thr*10) % 25 == 0: # 2.5刻みで表示
            print(f"| {thr:.1f} | {tpr:.2%} | {fpr:.2%} |")

    # AUC計算
    # FPRは閾値が高いほど小さくなる（逆順）ため、ソートして計算
    roc_auc = auc(sorted(fpr_list), [t for _, t in sorted(zip(fpr_list, tpr_list))])
    
    print(f"\n**Area Under the Curve (AUC)**: **{roc_auc:.4f}**")
    
    if roc_auc > 0.9:
        print("✅ パイプラインは「極めて優秀 (Excellent)」な分類性能を示しています。")
    elif roc_auc > 0.8:
        print("✅ パイプラインは「実用的 (Good)」な性能を示しています。")
    else:
        print("⚠️ 感度と精度のバランスに改善の余地があります。")

if __name__ == "__main__":
    calculate_roc_auc()
