// AstroTransit-GPU V15 Phase 1 (Tensor-Blitz Prototype)
#include <cuda_runtime.h>
#include <cuda_fp16.h>

typedef long long index_t;

extern "C" __global__ void vbls_blitz_kernel(
    const float* __restrict__ flux_matrix,    
    const float* __restrict__ time,        
    const float* __restrict__ inv_periods,
    float* __restrict__ candidate_scores, // SNR-like value to filter
    const int n_data,
    const int n_periods,
    const float t_start
) {
    const index_t p_idx = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
    const index_t target_idx = blockIdx.y;
    
    if (p_idx >= (index_t)n_periods) return;

    // Blitz uses ONLY registers for speed (Fixed 50 bins)
    // 50 bins * float = 200 bytes. Well within register limit.
    float r_counts[50];
    float r_flux_sum[50];

    #pragma unroll
    for (int i = 0; i < 50; i++) {
        r_counts[i] = 0;
        r_flux_sum[i] = 0;
    }

    const float inv_p = inv_periods[p_idx];
    const index_t target_offset = target_idx * (index_t)n_data;

    // Binning into registers
    // We only use every 2nd data point to double the speed (Sub-sampling for Blitz)
    for (int i = 0; i < n_data; i += 2) {
        float f = __ldg(&flux_matrix[target_offset + i]);
        float dt = __ldg(&time[i]) - t_start;
        float phase = dt * inv_p;
        phase -= __float2int_rd(phase);
        int bin = __float2int_rd(phase * 50.0f);
        bin = max(0, min(bin, 49));
        r_counts[bin] += 1.0f;
        r_flux_sum[bin] += f;
    }

    // Coarse Search (only 1 typical duration to check candidates)
    float total_c = 0, total_f = 0;
    #pragma unroll
    for (int i = 0; i < 50; i++) {
        total_c += r_counts[i];
        total_f += r_flux_sum[i];
    }

    float max_score = 0;
    // Simple Box-car scan in registers
    #pragma unroll
    for (int s = 0; s < 50; s++) {
        // Just check a window of 3 bins (~6% duration)
        float cur_c = 0, cur_f = 0;
        #pragma unroll
        for (int w = 0; w < 3; w++) {
            int idx = (s + w) % 50;
            cur_c += r_counts[idx];
            cur_f += r_flux_sum[idx];
        }
        
        if (cur_c > 0 && cur_c < total_c) {
            float r = cur_c / total_c;
            float s_val = cur_f - r * total_f;
            float score = (s_val * s_val) / (r * (1.0f - r));
            max_score = max(max_score, score);
        }
    }

    // Atomic update global candidate score for this target
    // We want the maximum score across all periods for this target
    atomicMax((int*)&candidate_scores[target_idx], __float_as_int(sqrtf(max_score)));
}
