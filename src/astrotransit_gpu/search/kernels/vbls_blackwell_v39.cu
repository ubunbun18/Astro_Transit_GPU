#include <cupy/complex.cuh>

// Constant memory for durations (must be defined in the .cu file)
__constant__ float c_durations[32];

// V41: Thread-safe metadata commit helper
__device__ __forceinline__ void atomicMaxResult(float* global_power, float* global_t0, float* global_dur, float* global_dep, float* global_period,
                                float val, float t0, float dur, float depth, float period) {
    int* address_as_ui = (int*)global_power;
    int old = *address_as_ui, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;
        old = atomicCAS(address_as_ui, assumed, __float_as_uint(val));
        if (old == assumed) {
            // Commit metadata first
            *(volatile float*)global_t0 = t0;
            *(volatile float*)global_dur = dur;
            *(volatile float*)global_dep = depth;
            *(volatile float*)global_period = period;
            __threadfence(); 
            // Commit final power to signal completion
            *(volatile float*)global_power = val; 
            break;
        }
    } while (true);
}

extern "C" __global__ void __launch_bounds__(512, 2)
vbls_v39_weighted_kernel(
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
    const float t_start) 
{
    const int warp_id = threadIdx.x >> 5; 
    const int lane_id = threadIdx.x & 31;
    const int cur_p_idx = (blockIdx.x * 16) + warp_id;
    const int target_idx = blockIdx.y;

    extern __shared__ char s_mem[];
    float* s_flux = (float*)s_mem;
    float* s_dt = (float*)&s_mem[N_DATA * sizeof(float)];
    float* s_weight = (float*)&s_mem[2 * N_DATA * sizeof(float)];
    
    float (*s_w)[N_BINS + 1] = (float (*)[N_BINS + 1])&s_mem[3 * N_DATA * sizeof(float)];
    float (*s_wf)[N_BINS + 1] = (float (*)[N_BINS + 1])&s_mem[3 * N_DATA * sizeof(float) + 16 * (N_BINS + 1) * sizeof(float)];

    const int target_offset = target_idx * N_DATA;

    for (int i = threadIdx.x; i < N_DATA; i += 512) {
        float f = (float)flux_matrix[target_offset + i];
        float w = (float)weights_matrix[target_offset + i];
        float limit = (w > 0.0f) ? 12.0f * rsqrtf(w) : 0.0f;
        f = fminf(fmaxf(f, -limit), limit);
        s_flux[i] = f;
        s_weight[i] = w;
        s_dt[i] = (float)dt_array[i];
    }

    for (int i = lane_id; i < N_BINS + 1; i += 32) {
        s_w[warp_id][i] = 0.0f;
        s_wf[warp_id][i] = 0.0f;
    }
    __syncthreads();

    if (cur_p_idx < n_periods) {
        const float2 pdata = period_pairs[cur_p_idx];
        const float p_val = pdata.x;
        const float inv_p = pdata.y;
        const float f_nbins = (float)N_BINS;

        float total_wf_lane = 0.0f;
        float total_w_lane = 0.0f;

        for (int i = lane_id; i < N_DATA; i += 32) {
            float ph = s_dt[i] * inv_p;
            ph -= floorf(ph);
            float f = s_flux[i];
            float w = s_weight[i];
            float wf = w * f;
            int b = (int)(ph * f_nbins);
            b = (b >= N_BINS) ? N_BINS - 1 : b;
            atomicAdd(&s_w[warp_id][b + 1], w);
            atomicAdd(&s_wf[warp_id][b + 1], wf);
            total_wf_lane += wf;
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

        float prev_w = 0.0f;
        float prev_wf = 0.0f;
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
            val_w += prev_w;
            val_wf += prev_wf;
            s_w[warp_id][idx] = val_w;
            s_wf[warp_id][idx] = val_wf;
            prev_w = __shfl_sync(0xFFFFFFFF, val_w, 31);
            prev_wf = __shfl_sync(0xFFFFFFFF, val_wf, 31);
            __syncwarp();
        }

        float max_snr2 = -1.0f;
        float b_t0 = 0.0f, b_dur = 0.0f, b_dep = 0.0f;
        const float inv_n_bins_p = p_val / f_nbins;

        int n_db_arr[N_DURATIONS];
        #pragma unroll
        for (int d = 0; d < N_DURATIONS; d++) {
            int ndb = (int)(c_durations[d] * inv_p * f_nbins + 0.5f);
            n_db_arr[d] = (ndb < 1) ? 1 : ndb;
        }

        for (int s_bin = lane_id; s_bin < N_BINS; s_bin += 32) {
            const float s_w_base = s_w[warp_id][s_bin];
            const float s_wf_base = s_wf[warp_id][s_bin];

            for (int d = 0; d < N_DURATIONS; d++) {
                const int n_db = n_db_arr[d];
                if (n_db >= N_BINS) continue;

                int e_bin = s_bin + n_db;
                int e_idx = (e_bin > N_BINS) ? e_bin - N_BINS : e_bin;
                float lw = s_w[warp_id][e_idx];
                float lwf = s_wf[warp_id][e_idx];
                float wrap = (float)(e_bin > N_BINS);
                
                float cur_w = fmaf(total_w, wrap, lw - s_w_base);
                float cur_wf = fmaf(total_wf, wrap, lwf - s_wf_base);

                const float out_w = total_w - cur_w;
                if (cur_w > 1e-10f && out_w > 1e-10f) {
                    float delta = fmaf(cur_w, total_wf, -cur_wf * total_w);
                    
                    if (delta > 0.0f) {
                        // IEEE 754 compliant division (rsqrtf is non-compliant, avoid for parity)
                        float snr2 = delta * delta / (cur_w * out_w * total_w);
                        if (snr2 > max_snr2) {
                            max_snr2 = snr2;
                            float this_dur = c_durations[d];
                            b_t0 = ((float)s_bin + 0.5f) * inv_n_bins_p + t_start + this_dur * 0.5f;
                            b_dur = this_dur;
                            b_dep = delta / (cur_w * out_w);
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float o_snr2 = __shfl_down_sync(0xFFFFFFFF, max_snr2, off);
            float o_t0  = __shfl_down_sync(0xFFFFFFFF, b_t0, off);
            float o_dur = __shfl_down_sync(0xFFFFFFFF, b_dur, off);
            float o_dep = __shfl_down_sync(0xFFFFFFFF, b_dep, off);
            if (o_snr2 > max_snr2) {
                max_snr2 = o_snr2; b_t0 = o_t0; b_dur = o_dur; b_dep = o_dep;
            }
        }

        if (lane_id == 0) {
            float max_snr = (max_snr2 > 0.0f) ? sqrtf(max_snr2) : 0.0f;
            atomicMaxResult(&global_max_power[target_idx], &global_best_t0[target_idx], 
                           &global_best_dur[target_idx], &global_best_depth[target_idx], 
                           &global_best_period[target_idx],
                           max_snr, b_t0, b_dur, b_dep, p_val);
        }
    }
}
