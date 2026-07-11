import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from ..search.gpu_bls import run_gpu_bls, _require_cupy

def _worker_process_data(target_id):
    """External worker function for ProcessPoolExecutor"""
    try:
        from ..data.lightkurve_client import LightkurveClient
        from ..preprocess.clean import clean_lightcurve, to_arrays
        client = LightkurveClient()
        lc = client.download_lightcurve(target_id)
        lc_clean = clean_lightcurve(lc)
        time_arr, flux_arr = to_arrays(lc_clean)
        return target_id, time_arr, flux_arr, None
    except Exception as e:
        return target_id, None, None, str(e)

class AsyncTransitSearchEngine:
    def __init__(self, n_workers=None, n_streams=4):
        cp = _require_cupy()
        self.n_workers = n_workers or (os.cpu_count() or 4)
        self.n_streams = n_streams
        self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]
        
    def run_batch(self, target_ids, periods, durations):
        results = []
        
        # Using ProcessPool to avoid GIL and library thread-safety issues
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_target = {executor.submit(_worker_process_data, tid): tid for tid in target_ids}
            
            stream_idx = 0
            for future in as_completed(future_to_target):
                tid, time_arr, flux_arr, error = future.result()
                
                if error:
                    results.append({'target_id': tid, 'error': error})
                    continue
                
                # Assign to CUDA stream for parallel GPU execution
                current_stream = self.streams[stream_idx % self.n_streams]
                with current_stream:
                    res = run_gpu_bls(time_arr, flux_arr, periods, durations)
                    res['target_id'] = tid
                    results.append(res)
                stream_idx += 1
                
        return results

def run_hyper_batch_search(target_ids, periods=None, durations=None, n_workers=8):
    """Entry point for hyper-speed batch search."""
    if periods is None:
        periods = np.linspace(0.5, 20.0, 5000)
    if durations is None:
        durations = np.linspace(0.01, 0.2, 5)
        
    engine = AsyncTransitSearchEngine(n_workers=n_workers)
    return engine.run_batch(target_ids, periods, durations)
