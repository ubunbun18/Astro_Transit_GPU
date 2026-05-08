// AstroTransit-GPU V39 (Weight-aware Blackwell Apex - Warp Cooperative)
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
void vbls_v39_weighted_kernel(
    const SCALAR_T* __restrict__ flux_matrix,    
    const SCALAR_T* __restrict__ dt_array,        
    const float2* __restrict__ period_pairs,      
    const SCALAR_T* __restrict__ weights_matrix,
    float* __restrict__ global_max_power,       
    float* __restrict__ global_best_t0,       
    float* __restrict__ global_best_dur,
    float* __restrict__ global_best_depth,
    float* __restrict__ global_best_period,
    const int n_periods,
    const SCALAR_T t_start
) {
    // V39: Weight-aware version (Gap handling)
    const int target_idx = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int cur_p_idx = (blockIdx.x * 16) + warp_id;
    
    // Shared memory layouts (Dynamic)
    extern __shared__ char s_mem[];
    float* s_flux = (float*)s_mem;
    float* s_dt = (float*)&s_mem[N_DATA * sizeof(float)];
    float* s_weight = (float*)&s_mem[2 * N_DATA * sizeof(float)];
    
    // Binning buffers: s_w (weight sum) and s_wf (weighted flux sum)
    float (*s_w)[N_BINS + 1] = (float (*)[N_BINS + 1])&s_mem[3 * N_DATA * sizeof(float)];
    float (*s_wf)[N_BINS + 1] = (float (*)[N_BINS + 1])&s_mem[3 * N_DATA * sizeof(float) + 16 * (N_BINS + 1) * sizeof(float)];

    const int target_offset = target_idx * N_DATA;

    // Cooperative load (512 threads)
    #pragma unroll
    for (int i = threadIdx.x; i < N_DATA; i += 512) {
        float f = (float)flux_matrix[target_offset + i];
        float w = (float)weights_matrix[target_offset + i];
        
        // Outlier Clipping (Robustness) using local sigma if w > 0
        if (w > 0) {
            float sigma = 1.0f / sqrtf(w);
            float limit = 10.0f * sigma;
            if (f > limit) f = limit;
            if (f < -limit) f = -limit;
        } else {
            f = 0.0f;
        }

        s_flux[i] = f;
        s_weight[i] = w;
        s_dt[i] = (float)dt_array[i];
    }
    
    // Zero out bins
    #pragma unroll
    for (int i = lane_id; i < N_BINS + 1; i += 32) {
        s_w[warp_id][i] = 0.0f;
        s_wf[warp_id][i] = 0.0f;
    }
    __syncthreads();

    if (cur_p_idx < n_periods) {
        const float f_nbins = (float)N_BINS;
        const float2 pdata = period_pairs[cur_p_idx];
        const float p_val = pdata.x;
        const float inv_p = pdata.y;

        float total_wf_lane = 0.0f;
        float total_w_lane = 0.0f;
        for (int i = lane_id; i < N_DATA; i += 32) {
            float ph = s_dt[i] * inv_p;
            float f = s_flux[i];
            float w = s_weight[i];
            
            ph -= (float)((int)ph);
            if (ph < 0) ph += 1.0f;
            int b = (int)(ph * f_nbins);
            b = (b >= N_BINS) ? N_BINS - 1 : b;
            
            atomicAdd(&s_w[warp_id][b + 1], w);
            atomicAdd(&s_wf[warp_id][b + 1], w * f);
            total_wf_lane += w * f;
            total_w_lane += w;
        }
        __syncwarp();

        float total_wf = total_wf_lane;
        float total_w = total_w_lane;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            total_wf += __shfl_down_sync(0xFFFFFFFF, total_wf, off);
            total_w += __shfl_down_sync(0xFFFFFFFF, total_w, off);
        }
        total_wf = __shfl_sync(0xFFFFFFFF, total_wf, 0);
        total_w = __shfl_sync(0xFFFFFFFF, total_w, 0);

        // Parallel Prefix Sum
        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            int idx = chunk * 32 + lane_id + 1;
            float val_w = s_w[warp_id][idx];
            float val_wf = s_wf[warp_id][idx];
            #pragma unroll
            for (int j = 1; j < 32; j <<= 1) {
                float tmp_w = __shfl_up_sync(0xFFFFFFFF, val_w, j);
                float tmp_wf = __shfl_up_sync(0xFFFFFFFF, val_wf, j);
                if (lane_id >= j) { val_w += tmp_w; val_wf += tmp_wf; }
            }
            if (chunk > 0) {
                val_w += __shfl_sync(0xFFFFFFFF, s_w[warp_id][chunk * 32], 0);
                val_wf += __shfl_sync(0xFFFFFFFF, s_wf[warp_id][chunk * 32], 0);
            }
            s_w[warp_id][idx] = val_w;
            s_wf[warp_id][idx] = val_wf;
            __syncwarp();
        }

        const float inv_n_bins_p = p_val / f_nbins;
        float max_snr = -1.0f;
        float b_t0 = 0, b_dur = 0, b_dep = 0;

        // V39: Weight-aware SNR Search
        #pragma unroll
        for (int s_bin = lane_id; s_bin < N_BINS; s_bin += 32) {
            const float s_w_base = s_w[warp_id][s_bin];
            const float s_wf_base = s_wf[warp_id][s_bin];
            #pragma unroll
            for (int d = 0; d < N_DURATIONS; d++) {
                int n_db = (int)(c_durations[d] * inv_p * f_nbins + 0.5f);
                if (n_db < 1 || n_db >= N_BINS) continue;
                int e_bin = s_bin + n_db;
                float cur_w, cur_wf;
                if (e_bin <= N_BINS) {
                    cur_w = s_w[warp_id][e_bin] - s_w_base;
                    cur_wf = s_wf[warp_id][e_bin] - s_wf_base;
                } else {
                    cur_w = (total_w - s_w_base) + s_w[warp_id][e_bin - N_BINS];
                    cur_wf = (total_wf - s_wf_base) + s_wf[warp_id][e_bin - N_BINS];
                }
                
                if (cur_w > 1e-10f && cur_w < total_w - 1e-10f) {
                    const float out_w = total_w - cur_w;
                    // numer = total_W * S_WF - S_W * total_WF
                    const float numer = total_w * cur_wf - cur_w * total_wf;
                    // denom = sqrt(S_W * (total_W - S_W) * total_W)
                    float denom = sqrtf(cur_w * out_w * total_w);
                    if (denom > 1e-12f) {
                        float snr = -numer / denom; // Positive for transit
                        if (snr > max_snr) {
                            max_snr = snr;
                            b_t0 = (float)s_bin * inv_n_bins_p + t_start;
                            b_dur = c_durations[d];
                            b_dep = (total_wf - cur_wf) / out_w - cur_wf / cur_w;
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
