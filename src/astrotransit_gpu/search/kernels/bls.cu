// Hyper-Optimized Tiled BLS CUDA kernel
// Support for variable precision (SCALAR_T)

#ifndef SCALAR_T
#define SCALAR_T float
#endif

#define N_TILE 4

extern "C" __global__ void bls_kernel(
    const SCALAR_T* __restrict__ time,
    const SCALAR_T* __restrict__ flux,
    const SCALAR_T* __restrict__ weights,
    int n_data,
    const SCALAR_T* __restrict__ inv_periods, 
    int n_periods,
    const SCALAR_T* __restrict__ durations,
    int n_durations,
    int n_bins,
    SCALAR_T t_start,
    SCALAR_T* power,
    SCALAR_T* best_t0,
    SCALAR_T* best_dur,
    SCALAR_T* best_depth
) {
    extern __shared__ char s_mem_raw[];
    SCALAR_T* s_mem = (SCALAR_T*)s_mem_raw;
    
    SCALAR_T* s_counts_base = s_mem;
    SCALAR_T* s_flux_sum_base = &s_mem[N_TILE * n_bins];
    SCALAR_T* s_red_max = &s_mem[N_TILE * n_bins * 2];
    SCALAR_T* s_red_t0  = &s_red_max[blockDim.x];
    SCALAR_T* s_red_dur = &s_red_max[blockDim.x * 2];
    SCALAR_T* s_red_dep = &s_red_max[blockDim.x * 3];

    int tile_base_idx = blockIdx.x * N_TILE;
    
    // 1. Initialize shared memory bins
    for (int i = threadIdx.x; i < N_TILE * n_bins; i += blockDim.x) {
        s_counts_base[i] = (SCALAR_T)0;
        s_flux_sum_base[i] = (SCALAR_T)0;
    }
    __syncthreads();

    // 2. Binning phase
    for (int i = threadIdx.x; i < n_data; i += blockDim.x) {
        SCALAR_T t = __ldg(&time[i]);
        SCALAR_T f = __ldg(&flux[i]);
        SCALAR_T w = __ldg(&weights[i]);
        SCALAR_T t_rel = t - t_start;
        
        #pragma unroll
        for (int t_idx = 0; t_idx < N_TILE; t_idx++) {
            int p_idx = tile_base_idx + t_idx;
            if (p_idx < n_periods) {
                SCALAR_T inv_p = inv_periods[p_idx];
                SCALAR_T phase = t_rel * inv_p;
                phase -= floor(phase);
                int bin = (int)(phase * (SCALAR_T)n_bins);
                if (bin >= 0 && bin < n_bins) {
                    atomicAdd(&s_counts_base[t_idx * n_bins + bin], w);
                    atomicAdd(&s_flux_sum_base[t_idx * n_bins + bin], f * w);
                }
            }
        }
    }
    __syncthreads();

    // 3. Search phase
    for (int t_idx = 0; t_idx < N_TILE; t_idx++) {
        int p_idx = tile_base_idx + t_idx;
        if (p_idx >= n_periods) continue;
        
        SCALAR_T* s_counts = &s_counts_base[t_idx * n_bins];
        SCALAR_T* s_flux_sum = &s_flux_sum_base[t_idx * n_bins];
        SCALAR_T inv_p = inv_periods[p_idx];
        SCALAR_T p = (SCALAR_T)1.0 / inv_p;

        // Prefix Sum (Hillis-Steele)
        // Note: Assumes blockDim.x >= n_bins
        for (int stride = 1; stride < n_bins; stride *= 2) {
            SCALAR_T val_c = 0, val_f = 0;
            if (threadIdx.x >= stride && threadIdx.x < n_bins) {
                val_c = s_counts[threadIdx.x - stride];
                val_f = s_flux_sum[threadIdx.x - stride];
            }
            __syncthreads();
            if (threadIdx.x >= stride && threadIdx.x < n_bins) {
                s_counts[threadIdx.x] += val_c;
                s_flux_sum[threadIdx.x] += val_f;
            }
            __syncthreads();
        }

        SCALAR_T total_counts = s_counts[n_bins - 1];
        SCALAR_T total_flux_sum = s_flux_sum[n_bins - 1];

        SCALAR_T max_score = (SCALAR_T)-1.0;
        SCALAR_T b_t0 = (SCALAR_T)0;
        SCALAR_T b_dur = (SCALAR_T)0;
        SCALAR_T b_depth = (SCALAR_T)0;

        if (threadIdx.x < n_bins) {
            int start_bin = threadIdx.x;
            for (int d_idx = 0; d_idx < n_durations; d_idx++) {
                SCALAR_T dur = durations[d_idx];
                int n_dur_bins = (int)(dur * inv_p * (SCALAR_T)n_bins + (SCALAR_T)0.5);
                if (n_dur_bins < 1) n_dur_bins = 1;
                if (n_dur_bins >= n_bins) continue;

                SCALAR_T cur_counts, cur_flux_sum;
                int end_bin = start_bin + n_dur_bins - 1;
                if (end_bin < n_bins) {
                    SCALAR_T prev_c = (start_bin > 0) ? s_counts[start_bin - 1] : (SCALAR_T)0;
                    SCALAR_T prev_f = (start_bin > 0) ? s_flux_sum[start_bin - 1] : (SCALAR_T)0;
                    cur_counts = s_counts[end_bin] - prev_c;
                    cur_flux_sum = s_flux_sum[end_bin] - prev_f;
                } else {
                    int wrap_end = end_bin % n_bins;
                    SCALAR_T prev_c = (start_bin > 0) ? s_counts[start_bin - 1] : (SCALAR_T)0;
                    SCALAR_T prev_f = (start_bin > 0) ? s_flux_sum[start_bin - 1] : (SCALAR_T)0;
                    cur_counts = (total_counts - prev_c) + s_counts[wrap_end];
                    cur_flux_sum = (total_flux_sum - prev_f) + s_flux_sum[wrap_end];
                }

                if (cur_counts > (SCALAR_T)0 && cur_counts < total_counts) {
                    SCALAR_T r = cur_counts / total_counts;
                    SCALAR_T s = cur_flux_sum - r * total_flux_sum;
                    SCALAR_T score = (s * s) / (r * ((SCALAR_T)1.0 - r));
                    if (score > max_score) {
                        max_score = score;
                        b_t0 = t_start + (SCALAR_T)start_bin / (SCALAR_T)n_bins * p;
                        b_dur = dur;
                        SCALAR_T in_mean = cur_flux_sum / cur_counts;
                        SCALAR_T out_mean = (total_flux_sum - cur_flux_sum) / (total_counts - cur_counts);
                        b_depth = out_mean - in_mean;
                    }
                }
            }
        }

        // Reduction
        __syncthreads();
        s_red_max[threadIdx.x] = max_score;
        s_red_t0[threadIdx.x] = b_t0;
        s_red_dur[threadIdx.x] = b_dur;
        s_red_dep[threadIdx.x] = b_depth;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                if (s_red_max[threadIdx.x + stride] > s_red_max[threadIdx.x]) {
                    s_red_max[threadIdx.x] = s_red_max[threadIdx.x + stride];
                    s_red_t0[threadIdx.x]  = s_red_t0[threadIdx.x + stride];
                    s_red_dur[threadIdx.x]  = s_red_dur[threadIdx.x + stride];
                    s_red_dep[threadIdx.x]  = s_red_dep[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            power[p_idx] = (SCALAR_T)sqrt(max((SCALAR_T)0, s_red_max[0]));
            best_t0[p_idx] = s_red_t0[0];
            best_dur[p_idx] = s_red_dur[0];
            best_depth[p_idx] = s_red_dep[0];
        }
        __syncthreads(); 
    }
}
