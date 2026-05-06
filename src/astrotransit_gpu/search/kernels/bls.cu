// Hyper-Optimized Tiled BLS CUDA kernel
// Features: 
// 1. Division-free phase calculation (Multiplication by inverse period)
// 2. Read-only data cache optimization (__ldg)
// 3. Warp-level scan for prefix sums (Partial)
// 4. Memory Tiling (N_TILE=8)

#define N_TILE 8

extern "C" __global__ void bls_kernel(
    const float* __restrict__ time,
    const float* __restrict__ flux,
    int n_data,
    const float* __restrict__ inv_periods, // Passed 1/P for faster multiplication
    int n_periods,
    const float* __restrict__ durations,
    int n_durations,
    int n_bins,
    float t_start,
    float* power,
    float* best_t0,
    float* best_dur,
    float* best_depth
) {
    extern __shared__ float s_mem[];
    float* s_counts_base = s_mem;
    float* s_flux_sum_base = &s_mem[N_TILE * n_bins];
    float* s_red_max = &s_mem[N_TILE * n_bins * 2];
    float* s_red_t0  = &s_red_max[blockDim.x];
    float* s_red_dur = &s_red_max[blockDim.x * 2];
    float* s_red_dep = &s_red_max[blockDim.x * 3];

    int tile_base_idx = blockIdx.x * N_TILE;
    
    // 1. Initialize shared memory bins
    for (int i = threadIdx.x; i < N_TILE * n_bins; i += blockDim.x) {
        s_counts_base[i] = 0.0f;
        s_flux_sum_base[i] = 0.0f;
    }
    __syncthreads();

    // 2. Binning phase with __ldg (Read-only cache) and No-Division
    for (int i = threadIdx.x; i < n_data; i += blockDim.x) {
        // Force use of read-only data cache (__ldg)
        float t = __ldg(&time[i]);
        float f = __ldg(&flux[i]);
        float t_rel = t - t_start;
        
        #pragma unroll
        for (int t_idx = 0; t_idx < N_TILE; t_idx++) {
            int p_idx = tile_base_idx + t_idx;
            if (p_idx < n_periods) {
                // Multiplication is much faster than division
                float inv_p = inv_periods[p_idx];
                float phase = t_rel * inv_p;
                phase -= floorf(phase);
                int bin = (int)(phase * n_bins);
                if (bin >= 0 && bin < n_bins) {
                    atomicAdd(&s_counts_base[t_idx * n_bins + bin], 1.0f);
                    atomicAdd(&s_flux_sum_base[t_idx * n_bins + bin], f);
                }
            }
        }
    }
    __syncthreads();

    // 3. Search phase for each period in the tile
    for (int t_idx = 0; t_idx < N_TILE; t_idx++) {
        int p_idx = tile_base_idx + t_idx;
        if (p_idx >= n_periods) continue;
        
        float* s_counts = &s_counts_base[t_idx * n_bins];
        float* s_flux_sum = &s_flux_sum_base[t_idx * n_bins];
        float inv_p = inv_periods[p_idx];
        float p = 1.0f / inv_p;

        // Optimized Prefix Sum (Inclusive Scan)
        // Using Kogge-Stone style scan in shared memory
        for (int stride = 1; stride < n_bins; stride *= 2) {
            float val_c = 0, val_f = 0;
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

        float total_counts = s_counts[n_bins - 1];
        float total_flux_sum = s_flux_sum[n_bins - 1];

        float max_score = -1.0f;
        float b_t0 = 0.0f;
        float b_dur = 0.0f;
        float b_depth = 0.0f;

        if (threadIdx.x < n_bins) {
            int start_bin = threadIdx.x;
            for (int d_idx = 0; d_idx < n_durations; d_idx++) {
                float dur = durations[d_idx];
                // Use multiplication instead of division
                int n_dur_bins = (int)(dur * inv_p * n_bins + 0.5f);
                if (n_dur_bins < 1) n_dur_bins = 1;
                if (n_dur_bins >= n_bins) continue;

                float cur_counts, cur_flux_sum;
                int end_bin = start_bin + n_dur_bins - 1;
                if (end_bin < n_bins) {
                    float prev_c = (start_bin > 0) ? s_counts[start_bin - 1] : 0.0f;
                    float prev_f = (start_bin > 0) ? s_flux_sum[start_bin - 1] : 0.0f;
                    cur_counts = s_counts[end_bin] - prev_c;
                    cur_flux_sum = s_flux_sum[end_bin] - prev_f;
                } else {
                    int wrap_end = end_bin % n_bins;
                    float prev_c = (start_bin > 0) ? s_counts[start_bin - 1] : 0.0f;
                    float prev_f = (start_bin > 0) ? s_flux_sum[start_bin - 1] : 0.0f;
                    cur_counts = (total_counts - prev_c) + s_counts[wrap_end];
                    cur_flux_sum = (total_flux_sum - prev_f) + s_flux_sum[wrap_end];
                }

                if (cur_counts > 0 && cur_counts < total_counts) {
                    float r = cur_counts / total_counts;
                    float s = cur_flux_sum - r * total_flux_sum;
                    float score = (s * s) / (r * (1.0f - r));
                    if (score > max_score) {
                        max_score = score;
                        // Use precomputed p
                        b_t0 = t_start + (float)start_bin / (float)n_bins * p;
                        b_dur = dur;
                        float in_mean = cur_flux_sum / cur_counts;
                        float out_mean = (total_flux_sum - cur_flux_sum) / (total_counts - cur_counts);
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
            power[p_idx] = sqrtf(max(0.0f, s_red_max[0]));
            best_t0[p_idx] = s_red_t0[0];
            best_dur[p_idx] = s_red_dur[0];
            best_depth[p_idx] = s_red_dep[0];
        }
        __syncthreads(); 
    }
}
