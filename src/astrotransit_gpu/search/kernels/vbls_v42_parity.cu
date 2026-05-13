#ifndef SCALAR_T
#define SCALAR_T double
#endif

// Vectorized type for combined count and flux
typedef struct {
    SCALAR_T counts;
    SCALAR_T flux;
} SCALAR_VALS;

// V42 Apex-Parity Kernel (Stabilized 20.83 LC/s version)
extern "C" __global__ void
vbls_v42_parity_kernel(
    const SCALAR_T* __restrict__ flux_matrix,    
    const SCALAR_T* __restrict__ dt_array,        
    const SCALAR_T* __restrict__ periods,      
    const SCALAR_T* __restrict__ durations,
    const SCALAR_T* __restrict__ weights_matrix,
    SCALAR_T* __restrict__ out_power,       
    SCALAR_T* __restrict__ out_best_t0,         
    SCALAR_T* __restrict__ out_best_dur,        
    SCALAR_T* __restrict__ out_best_depth,      
    const int n_data,
    const int n_durations,
    const int oversample,
    const int max_bins,
    const SCALAR_T t_start,
    const SCALAR_T* __restrict__ sum_y_totals,
    const SCALAR_T* __restrict__ sum_ivar_totals,
    const int total_targets
) {
    const int p_idx = blockIdx.x;
    const int target_idx = blockIdx.y;
    if (target_idx >= total_targets) return;

    const SCALAR_T period = periods[p_idx];
    const SCALAR_T inv_period = (SCALAR_T)1.0 / period;
    
    SCALAR_T min_duration = durations[0];
    for(int d = 1; d < n_durations; d++) {
        if(durations[d] < min_duration) min_duration = durations[d];
    }
    SCALAR_T bin_duration = min_duration / (SCALAR_T)oversample;
    const SCALAR_T inv_bin_dur = (SCALAR_T)1.0 / bin_duration;
    
    int n_bins = (int)(ceil(period * inv_bin_dur)) + oversample;
    if (n_bins >= max_bins) n_bins = max_bins - 1;

    extern __shared__ long long s_mem_raw_64[];
    SCALAR_VALS* s_bins = (SCALAR_VALS*)s_mem_raw_64;
    int* s_dur_bins_ptr = (int*)&s_bins[max_bins + 2];
    
    // 1. Parallel Initialization
    for (int i = threadIdx.x; i < (max_bins + 2); i += blockDim.x) {
        s_bins[i].counts = 0.0;
        s_bins[i].flux = 0.0;
    }
    if (threadIdx.x < n_durations) {
        s_dur_bins_ptr[threadIdx.x] = (int)round(durations[threadIdx.x] * inv_bin_dur);
    }
    __syncthreads();
    
    // 2. Sequential binning on thread 0
    const int target_offset = target_idx * n_data;
    if (threadIdx.x == 0) {
        for (int i = 0; i < n_data; i++) {
            SCALAR_T t = dt_array[i];
            SCALAR_T ph = t - period * floor(t * inv_period);
            int ind = (int)(ph * inv_bin_dur) + 1;
            if (ind > n_bins) ind = n_bins;
            
            SCALAR_T w = weights_matrix[target_offset + i];
            s_bins[ind].counts += w;
            s_bins[ind].flux += flux_matrix[target_offset + i] * w;
        }
        
        // Sequential Padding & Prefix Sum
        for (int n = 1, ind = n_bins - oversample; n <= oversample; ++n, ++ind) {
            s_bins[ind] = s_bins[n];
        }
        for (int n = 1; n <= n_bins; ++n) {
            s_bins[n].counts += s_bins[n - 1].counts;
            s_bins[n].flux += s_bins[n - 1].flux;
        }
    }
    __syncthreads();
    
    // 3. Parallel Search
    SCALAR_T best_snr = (SCALAR_T)-1.0;
    SCALAR_T best_t0_val = (SCALAR_T)0.0;
    SCALAR_T best_dur_val = (SCALAR_T)0.0;
    SCALAR_T best_dep_val = (SCALAR_T)0.0;
    
    const SCALAR_T sum_y_total = sum_y_totals[target_idx];
    const SCALAR_T sum_ivar_total = sum_ivar_totals[target_idx];
    
    for (int n = threadIdx.x; n < n_bins; n += blockDim.x) {
        SCALAR_VALS start_val = s_bins[n];
        
        #pragma unroll
        for (int k = 0; k < n_durations; ++k) {
            int dur_bins = s_dur_bins_ptr[k];
            if (n + dur_bins > n_bins) continue;
            
            SCALAR_VALS end_val = s_bins[n + dur_bins];
            SCALAR_T ivar_in = end_val.counts - start_val.counts;
            SCALAR_T y_in_sum = end_val.flux - start_val.flux;
            SCALAR_T ivar_out = sum_ivar_total - ivar_in;
            SCALAR_T y_out_sum = sum_y_total - y_in_sum;
            
            if (ivar_in < (SCALAR_T)1e-15 || ivar_out < (SCALAR_T)1e-15) continue;
            
            SCALAR_T y_in = y_in_sum / ivar_in;
            SCALAR_T y_out = y_out_sum / ivar_out;
            
            if (y_out >= y_in) {
                SCALAR_T depth = y_out - y_in;
                SCALAR_T depth_err = sqrt((SCALAR_T)1.0 / ivar_in + (SCALAR_T)1.0 / ivar_out);
                SCALAR_T snr = depth / depth_err;
                
                if (snr > best_snr) {
                    best_snr = snr;
                    best_dep_val = depth;
                    best_dur_val = (SCALAR_T)dur_bins * bin_duration;
                    best_t0_val = fmod(n * bin_duration + (SCALAR_T)0.5 * best_dur_val + t_start, period);
                }
            }
        }
    }
    
    // 4. Final Reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        SCALAR_T o_snr = __shfl_down_sync(0xffffffff, best_snr, offset);
        SCALAR_T o_t0 = __shfl_down_sync(0xffffffff, best_t0_val, offset);
        SCALAR_T o_dur = __shfl_down_sync(0xffffffff, best_dur_val, offset);
        SCALAR_T o_dep = __shfl_down_sync(0xffffffff, best_dep_val, offset);
        if (o_snr > best_snr) {
            best_snr = o_snr; best_t0_val = o_t0; best_dur_val = o_dur; best_dep_val = o_dep;
        }
    }
    
    __syncthreads();
    
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    if (lane == 0) {
        SCALAR_T* s_scratch = (SCALAR_T*)s_bins;
        s_scratch[warp] = best_snr;
        s_scratch[warp + 32] = best_t0_val;
        s_scratch[warp + 64] = best_dur_val;
        s_scratch[warp + 96] = best_dep_val;
    }
    __syncthreads();
    
    if (warp == 0) {
        SCALAR_T* s_scratch = (SCALAR_T*)s_bins;
        best_snr = (lane < blockDim.x / 32) ? s_scratch[lane] : (SCALAR_T)-1.0;
        best_t0_val = (lane < blockDim.x / 32) ? s_scratch[lane + 32] : (SCALAR_T)0.0;
        best_dur_val = (lane < blockDim.x / 32) ? s_scratch[lane + 64] : (SCALAR_T)0.0;
        best_dep_val = (lane < blockDim.x / 32) ? s_scratch[lane + 96] : (SCALAR_T)0.0;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            SCALAR_T o_snr = __shfl_down_sync(0xffffffff, best_snr, offset);
            SCALAR_T o_t0 = __shfl_down_sync(0xffffffff, best_t0_val, offset);
            SCALAR_T o_dur = __shfl_down_sync(0xffffffff, best_dur_val, offset);
            SCALAR_T o_dep = __shfl_down_sync(0xffffffff, best_dep_val, offset);
            if (o_snr > best_snr) {
                best_snr = o_snr; best_t0_val = o_t0; best_dur_val = o_dur; best_dep_val = o_dep;
            }
        }
        
        if (lane == 0) {
            int out_idx = target_idx * gridDim.x + p_idx;
            out_power[out_idx] = (best_snr > (SCALAR_T)0.0) ? best_snr : (SCALAR_T)0.0;
            out_best_t0[out_idx] = best_t0_val;
            out_best_dur[out_idx] = best_dur_val;
            out_best_depth[out_idx] = best_dep_val;
        }
    }
}
