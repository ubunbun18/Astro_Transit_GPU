// AstroTransit-GPU V38 (Robust Blackwell Apex - Warp Cooperative)
#include <cuda_runtime.h>

#ifndef SCALAR_T
#define SCALAR_T float
#endif

// JIT Constants
#ifndef N_DATA
#define N_DATA 1312
#endif
#ifndef N_BINS
#define N_BINS 128
#endif

__constant__ float c_durations[32];

__device__ __forceinline__ void atomicMaxResult(float* global_power, float* global_t0, float* global_dur, float* global_dep, float* global_period,
                                float val, float t0, float dur, float depth, float period) {
    if (val <= 1e-6f) return;
    unsigned int* address_as_ui = (unsigned int*)global_power;
    unsigned int old = atomicOr(address_as_ui, 0);
    unsigned int assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;
        old = atomicCAS(address_as_ui, assumed, __float_as_uint(val));
        if (old == assumed) {
            *(volatile float*)global_t0 = t0;
            *(volatile float*)global_dur = dur;
            *(volatile float*)global_dep = depth;
            *(volatile float*)global_period = period;
            break;
        }
    } while (true);
}

extern "C" __global__ __launch_bounds__(512, 1)
void vbls_v38_robust_kernel(
    const SCALAR_T* __restrict__ flux_matrix,    
    const SCALAR_T* __restrict__ dt_array,        
    const float2* __restrict__ period_pairs,      
    const float* __restrict__ global_rms,
    float* __restrict__ global_max_power,       
    float* __restrict__ global_best_t0,       
    float* __restrict__ global_best_dur,
    float* __restrict__ global_best_depth,
    float* __restrict__ global_best_period,
    const int n_periods,
    const SCALAR_T t_start
) {
    // V38: Robust version with outlier clipping and SNR normalization
    const int target_idx = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int cur_p_idx = (blockIdx.x * 16) + warp_id;
    
    // Shared memory layouts (Dynamic)
    extern __shared__ char s_mem[];
    float* s_flux = (float*)s_mem;
    float* s_dt = (float*)&s_mem[N_DATA * sizeof(float)];
    
    float (*s_c)[N_BINS + 1] = (float (*)[N_BINS + 1])&s_mem[2 * N_DATA * sizeof(float)];
    float (*s_f)[N_BINS + 1] = (float (*)[N_BINS + 1])&s_mem[2 * N_DATA * sizeof(float) + 16 * (N_BINS + 1) * sizeof(float)];

    const int target_offset = target_idx * N_DATA;
    const float rms = global_rms[target_idx];
    const float clip_limit = 10.0f * (rms > 1e-6f ? rms : 0.01f);

    // Cooperative load with clipping
    #pragma unroll
    for (int i = threadIdx.x; i < N_DATA; i += 512) {
        float f = (float)flux_matrix[target_offset + i];
        // Outlier Clipping (Robustness)
        if (f > 1.0f + clip_limit) f = 1.0f + clip_limit;
        if (f < 1.0f - clip_limit) f = 1.0f - clip_limit;
        
        s_flux[i] = f;
        s_dt[i] = (float)dt_array[i];
    }
    
    // Zero out bins
    #pragma unroll
    for (int i = lane_id; i < N_BINS + 1; i += 32) {
        s_c[warp_id][i] = 0.0f;
        s_f[warp_id][i] = 0.0f;
    }
    __syncthreads();

    if (cur_p_idx < n_periods) {
        const float f_nbins = (float)N_BINS;
        const float2 pdata = period_pairs[cur_p_idx];
        const float p_val = pdata.x;
        const float inv_p = pdata.y;

        float total_f_lane = 0.0f;
        for (int i = lane_id; i < N_DATA; i += 32) {
            float ph = s_dt[i] * inv_p;
            float f = s_flux[i];
            ph -= (float)((int)ph);
            if (ph < 0) ph += 1.0f;
            int b = (int)(ph * f_nbins);
            b = (b >= N_BINS) ? N_BINS - 1 : b;
            
            atomicAdd(&s_c[warp_id][b + 1], 1.0f);
            atomicAdd(&s_f[warp_id][b + 1], f);
            total_f_lane += f;
        }
        __syncwarp();

        float total_f = total_f_lane;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) total_f += __shfl_down_sync(0xFFFFFFFF, total_f, off);
        total_f = __shfl_sync(0xFFFFFFFF, total_f, 0);

        // Parallel Prefix Sum
        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            int idx = chunk * 32 + lane_id + 1;
            float val_c = s_c[warp_id][idx];
            float val_f = s_f[warp_id][idx];
            #pragma unroll
            for (int j = 1; j < 32; j <<= 1) {
                float tmp_c = __shfl_up_sync(0xFFFFFFFF, val_c, j);
                float tmp_f = __shfl_up_sync(0xFFFFFFFF, val_f, j);
                if (lane_id >= j) { val_c += tmp_c; val_f += tmp_f; }
            }
            if (chunk > 0) {
                val_c += __shfl_sync(0xFFFFFFFF, s_c[warp_id][chunk * 32], 0);
                val_f += __shfl_sync(0xFFFFFFFF, s_f[warp_id][chunk * 32], 0);
            }
            s_c[warp_id][idx] = val_c;
            s_f[warp_id][idx] = val_f;
            __syncwarp();
        }

        const float total_c = (float)N_DATA;
        const float inv_n_bins_p = p_val / f_nbins;
        float max_snr = -1.0f;
        float b_t0 = 0, b_dur = 0, b_dep = 0;

        // V38: SNR-based Search
        #pragma unroll
        for (int s_bin = lane_id; s_bin < N_BINS; s_bin += 32) {
            const float s_c_base = s_c[warp_id][s_bin];
            const float s_f_base = s_f[warp_id][s_bin];
            #pragma unroll
            for (int d = 0; d < N_DURATIONS; d++) {
                int n_db = (int)(c_durations[d] * inv_p * f_nbins + 0.5f);
                if (n_db < 1 || n_db >= N_BINS) continue;
                int e_bin = s_bin + n_db;
                float cur_c, cur_f;
                if (e_bin <= N_BINS) {
                    cur_c = s_c[warp_id][e_bin] - s_c_base;
                    cur_f = s_f[warp_id][e_bin] - s_f_base;
                } else {
                    cur_c = (total_c - s_c_base) + s_c[warp_id][e_bin - N_BINS];
                    cur_f = (total_f - s_f_base) + s_f[warp_id][e_bin - N_BINS];
                }
                
                if (cur_c > 1.0f && cur_c < total_c - 1.0f) {
                    const float out_c = total_c - cur_c;
                    // numer = cur_f * total_c - cur_c * total_f
                    // Scaled to avoid overflow: (cur_f/total_c - total_f/total_c) * total_c * total_c ? 
                    // No, numer is fine with clipped data (~10^4 scale)
                    const float numer = cur_f * total_c - cur_c * total_f;
                    
                    // Normalization by RMS and binomial variance
                    // denom = rms * sqrt(cur_c * out_c * total_c)
                    float denom = rms * sqrtf(cur_c * out_c * total_c);
                    if (denom > 1e-12f) {
                        float snr = -numer / denom; // Positive for transit
                        if (snr > max_snr) {
                            max_snr = snr;
                            b_t0 = (float)s_bin * inv_n_bins_p + t_start;
                            b_dur = c_durations[d];
                            b_dep = (total_f - cur_f) / out_c - cur_f / cur_c;
                        }
                    }
                }
            }
        }

        // Warp-level reduction
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float o_snr = __shfl_down_sync(0xFFFFFFFF, max_snr, off);
            float o_t0  = __shfl_down_sync(0xFFFFFFFF, b_t0, off);
            float o_dur = __shfl_down_sync(0xFFFFFFFF, b_dur, off);
            float o_dep = __shfl_down_sync(0xFFFFFFFF, b_dep, off);
            if (o_snr > max_snr) {
                max_snr = o_snr; b_t0 = o_t0; b_dur = o_dur; b_dep = o_dep;
            }
        }

        if (lane_id == 0 && max_snr > 0) {
            atomicMaxResult(
                &global_max_power[target_idx],
                &global_best_t0[target_idx],
                &global_best_dur[target_idx],
                &global_best_depth[target_idx],
                &global_best_period[target_idx],
                max_snr,
                b_t0, b_dur, b_dep, p_val
            );
        }
    }
}
