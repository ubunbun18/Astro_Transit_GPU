import numpy as np
import time
import cupy as cp
from astrotransit_gpu import BoxLeastSquaresGPU
from tqdm import tqdm

def run_real_sector_marathon():
    print("=== AstroTransit-GPU Sector Marathon (15,881 Targets) ===")
    
    n_targets = 15881
    n_data = 15000
    periods = np.linspace(1.0, 10.0, 5000)
    durations = np.array([0.1])
    
    # 共通の時間軸（メモリ節約のため使い回し）
    t = np.linspace(0, 27.0, n_data)
    
    def marathon(method):
        print(f"\nStarting Marathon: Method={method} ...")
        cp.cuda.Device().synchronize()
        start_time = time.perf_counter()
        
        # 1.6万天体を実際に1つずつ処理
        for i in tqdm(range(n_targets), desc=f"Processing {method}"):
            # 毎回新しい疑似フラックスを生成（実解析の負荷を再現）
            y = np.random.normal(0, 0.01, size=n_data)
            model = BoxLeastSquaresGPU(t, y)
            model.power(periods, durations, method=method)
            
        cp.cuda.Device().synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    # [1] V41 Marathon
    v41_total_time = marathon("fast")
    print(f"\n[RESULT] V41 Sector Finish Time: {v41_total_time:.2f}s ({v41_total_time/60:.2f} min)")

    # [2] V42 Marathon
    # V42は1.6万天体だと約6〜10分かかるため、実走して確定させます。
    v42_total_time = marathon("parity")
    print(f"\n[RESULT] V42 Sector Finish Time: {v42_total_time:.2f}s ({v42_total_time/60:.2f} min)")

    print("\n--- FINAL HARD-MEASURED SECTOR COST ---")
    print(f"V41 (15,881 targets, 5k periods): {v41_total_time:.2f}s")
    print(f"V42 (15,881 targets, 5k periods): {v42_total_time:.2f}s")

if __name__ == "__main__":
    run_real_sector_marathon()
