// AstroTransit-GPU Hyper-Fast V11 (Surgical Ultimate)
#ifndef SCALAR_T
#define SCALAR_T float
#endif

typedef long long index_t;

// Constant memory for durations
__constant__ float c_durations[32];

extern "C" __global__ void vbls_ultra_kernel(
    const SCALAR_T* __restrict__ flux_matrix,    
    const SCALAR_T* __restrict__ time,        
    const SCALAR_T* __restrict__ inv_periods,
    const SCALAR_T* durations_unused,
    SCALAR_T* __restrict__ power_matrix,       
    SCALAR_T* __restrict__ best_t0_matrix,       
    SCALAR_T* __restrict__ best_dur_matrix,
    SCALAR_T* __restrict__ best_depth_matrix,
    SCALAR_T* __restrict__ best_period_matrix,
    const int* __restrict__ params_i32,
    const SCALAR_T* __restrict__ params_scalar
) {
    if (params_i32 == NULL || params_scalar == NULL) return;
    
    const int n_data = params_i32[0];
    const int n_periods = params_i32[1];
    const int n_durations = params_i32[2];
    const int n_bins = params_i32[3];
    const SCALAR_T t_start = params_scalar[0];

    const index_t p_idx_base = (index_t)blockIdx.x * 16;
    const index_t target_idx = blockIdx.y;
    if (p_idx_base >= (index_t)n_periods) return;

    // Shared Memory - 16 Warps * 224 bins (28.6 KB) + Data (10.4 KB) = 39 KB. 
    // Fits perfectly in 48 KB.
    __shared__ float s_flux[1312]; 
    __shared__ float s_dt[1312];
    __shared__ float s_counts[16][224];
    __shared__ float s_flux_sum[16][224];

    // Cooperative Load
    const index_t target_offset = target_idx * (index_t)n_data;
    for (int i = threadIdx.x; i < n_data; i += blockDim.x) {
        s_flux[i] = __ldg(&flux_matrix[target_offset + i]);
        s_dt[i] = __ldg(&time[i]) - t_start;
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const index_t cur_p_idx = p_idx_base + warp_id;

    if (cur_p_idx < (index_t)n_periods) {
        #pragma unroll 7
        for (int i = lane_id; i < 224; i += 32) {
            s_counts[warp_id][i] = 0;
            s_flux_sum[warp_id][i] = 0;
        }

        const float inv_p = __ldg(&inv_periods[cur_p_idx]);

        // Binning (Highly Unrolled)
        #pragma unroll 8
        for (int i = lane_id; i < n_data; i += 32) {
            float phase = s_dt[i] * inv_p;
            phase -= __float2int_rd(phase);
            int bin = __float2int_rd(phase * (float)n_bins);
            bin = max(0, min(bin, n_bins - 1));
            atomicAdd(&s_counts[warp_id][bin], 1.0f);
            atomicAdd(&s_flux_sum[warp_id][bin], s_flux[i]);
        }
        __syncwarp();

        if (lane_id == 0) {
            #pragma unroll 10
            for (int i = 1; i < n_bins; i++) {
                s_counts[warp_id][i] += s_counts[warp_id][i-1];
                s_flux_sum[warp_id][i] += s_flux_sum[warp_id][i-1];
            }
        }
        __syncwarp();

        const float total_c = s_counts[warp_id][n_bins - 1];
        const float total_f = s_flux_sum[warp_id][n_bins - 1];
        const float p_val = 1.0f / inv_p;

        float local_max = -1.0f;
        float b_t0 = 0, b_dur = 0, b_dep = 0;

        for (int s_bin = lane_id; s_bin < n_bins; s_bin += 32) {
            const float s_c_prev = (s_bin > 0) ? s_counts[warp_id][s_bin - 1] : 0;
            const float s_f_prev = (s_bin > 0) ? s_flux_sum[warp_id][s_bin - 1] : 0;

            #pragma unroll
            for (int d = 0; d < n_durations; d++) {
                float dur = c_durations[d];
                int n_db = __float2int_rn(dur * inv_p * (float)n_bins);
                if (n_db < 1 || n_db >= n_bins) continue;

                int e_bin = s_bin + n_db - 1;
                float cur_c, cur_f;
                if (e_bin < n_bins) {
                    cur_c = s_counts[warp_id][e_bin] - s_c_prev;
                    cur_f = s_flux_sum[warp_id][e_bin] - s_f_prev;
                } else {
                    int wrap_end = e_bin % n_bins;
                    cur_c = (total_c - s_c_prev) + s_counts[warp_id][wrap_end];
                    cur_f = (total_f - s_f_prev) + s_flux_sum[warp_id][wrap_end];
                }

                if (cur_c > 1e-6f && cur_c < (total_c - 1e-6f)) {
                    // FAST MATH: __fdividef and FMA
                    float r = __fdividef(cur_c, total_c);
                    float s = __fmaf_rn(-r, total_f, cur_f);
                    float score = __fdividef(s * s, __fmaf_rn(-r, r, r)); // r*(1-r)
                    
                    if (score > local_max) {
                        local_max = score;
                        b_t0 = t_start + (float)s_bin / (float)n_bins * p_val;
                        b_dur = dur;
                        b_dep = __fdividef(total_f - cur_f, total_c - cur_c) - __fdividef(cur_f, cur_c);
                    }
                }
            }
        }

        // Warp Reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            if (other_max > local_max) {
                local_max = other_max;
                b_t0 = __shfl_down_sync(0xFFFFFFFF, b_t0, offset);
                b_dur = __shfl_down_sync(0xFFFFFFFF, b_dur, offset);
                b_dep = __shfl_down_sync(0xFFFFFFFF, b_dep, offset);
            }
        }

        if (lane_id == 0) {
            index_t out_idx = target_idx * (index_t)n_periods + cur_p_idx;
            power_matrix[out_idx] = (local_max > 0) ? sqrtf(local_max) : 0;
            best_t0_matrix[out_idx] = b_t0;
            best_dur_matrix[out_idx] = b_dur;
            best_depth_matrix[out_idx] = b_dep;
            best_period_matrix[out_idx] = p_val;
        }
    }
}
