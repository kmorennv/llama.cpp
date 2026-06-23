#include "penalty.cuh"

#include "ggml-nvtx.h"

#include <algorithm>
#include <cstdint>

// use dense path when penalized tokens cover a large fraction of the vocab
static constexpr int32_t k_penalties_dense_ratio = 8;

static __device__ float ggml_cuda_penalty_apply(
        float           x,
        const int32_t   cnt,
        const float     penalty_repeat,
        const float     penalty_freq,
        const float     penalty_present) {
    if (penalty_repeat != 1.0f) {
        if (x <= 0.0f) {
            x *= penalty_repeat;
        } else {
            x /= penalty_repeat;
        }
    }

    x -= float(cnt) * penalty_freq;
    if (cnt > 0) {
        x -= penalty_present;
    }

    return x;
}

static __global__ void k_penalties_sparse(
        float * __restrict__ logits,
        const int32_t * __restrict__ token_ids,
        const int32_t * __restrict__ counts,
        const int32_t         n_active,
        const int32_t         vocab,
        const float           penalty_repeat,
        const float           penalty_freq,
        const float           penalty_present) {
    if (n_active <= 0) {
        return;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_active; i += blockDim.x * gridDim.x) {
        const int32_t tok = token_ids[i];
        const int32_t cnt = counts[i];

        if (tok < 0 || tok >= vocab) {
            continue;
        }

        logits[tok] = ggml_cuda_penalty_apply(
            logits[tok], cnt, penalty_repeat, penalty_freq, penalty_present);
    }
}

static __global__ void k_penalties_scatter(
        int32_t * __restrict__ d_counts,
        const int32_t * __restrict__ token_ids,
        const int32_t * __restrict__ counts,
        const int32_t         n_active,
        const int32_t         vocab) {
    if (n_active <= 0) {
        return;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_active; i += blockDim.x * gridDim.x) {
        const int32_t tok = token_ids[i];
        if (tok < 0 || tok >= vocab) {
            continue;
        }

        // token ids are unique in the production path (token_count map)
        d_counts[tok] = counts[i];
    }
}

static __global__ void k_penalties_dense(
        float * __restrict__ logits,
        const int32_t * __restrict__ counts,
        const int32_t         vocab,
        const float           penalty_repeat,
        const float           penalty_freq,
        const float           penalty_present) {
    for (int tok = blockIdx.x * blockDim.x + threadIdx.x; tok < vocab; tok += blockDim.x * gridDim.x) {
        const int32_t cnt = counts[tok];
        if (cnt == 0) {
            continue;
        }

        logits[tok] = ggml_cuda_penalty_apply(
            logits[tok], cnt, penalty_repeat, penalty_freq, penalty_present);
    }
}

static int ggml_cuda_penalties_grid_size(const int64_t n, const int block_size) {
    return (int) std::min<int64_t>(65535, (n + block_size - 1) / block_size);
}

static bool ggml_cuda_penalties_use_dense(const int32_t n_active, const int32_t vocab) {
    return n_active > vocab / k_penalties_dense_ratio;
}

#ifdef GGML_NVTX

// stream-ordered NVTX: begin runs after prior stream work, end runs after penalty kernels only
static uint64_t g_penalties_nvtx_id = 0;

static void ggml_cuda_penalties_nvtx_begin(void *) {
    g_penalties_nvtx_id = ggml_nvtx_range_start("penalties_gpu", GGML_NVTX_COLOR_SAMPLER_BACKEND);
}

static void ggml_cuda_penalties_nvtx_end(void *) {
    ggml_nvtx_range_stop(g_penalties_nvtx_id);
}

#endif // GGML_NVTX

void ggml_cuda_op_penalties(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits   = dst->src[0];
    const ggml_tensor * ids      = dst->src[1];
    const ggml_tensor * counts   = dst->src[2];
    const ggml_tensor * n_active = dst->src[3];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(counts->type == GGML_TYPE_I32);
    GGML_ASSERT(n_active->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(logits));
    GGML_ASSERT(dst->data != nullptr);

    const float penalty_repeat  = ggml_get_op_params_f32(dst, 0);
    const float penalty_freq    = ggml_get_op_params_f32(dst, 1);
    const float penalty_present = ggml_get_op_params_f32(dst, 2);

    if (penalty_repeat == 1.0f && penalty_freq == 0.0f && penalty_present == 0.0f) {
        return;
    }

    float *         logits_d = (float *) dst->data;
    const int32_t * ids_d    = (const int32_t *) ids->data;
    const int32_t * cnt_d    = (const int32_t *) counts->data;

    const int32_t vocab = (int32_t) logits->ne[0];

    int32_t n_active_h = 0;
    if (n_active->data != nullptr && ggml_nelements(n_active) > 0) {
        CUDA_CHECK(cudaMemcpy(&n_active_h, n_active->data, sizeof(int32_t), cudaMemcpyDeviceToHost));
    }

    n_active_h = std::min(n_active_h, (int32_t) ids->ne[0]);
    if (n_active_h <= 0) {
        return;
    }

    const int block_size = 256;
    cudaStream_t stream  = ctx.stream();

#ifdef GGML_NVTX
    CUDA_CHECK(cudaLaunchHostFunc(stream, ggml_cuda_penalties_nvtx_begin, nullptr));
#endif

    if (ggml_cuda_penalties_use_dense(n_active_h, vocab)) {
        ggml_cuda_pool & pool = ctx.pool();

        ggml_cuda_pool_alloc<int32_t> d_counts(pool, vocab);
        CUDA_CHECK(cudaMemsetAsync(d_counts.get(), 0, vocab * sizeof(int32_t), stream));

        const int scatter_grid = ggml_cuda_penalties_grid_size(n_active_h, block_size);
        k_penalties_scatter<<<scatter_grid, block_size, 0, stream>>>(
            d_counts.get(), ids_d, cnt_d, n_active_h, vocab);

        const int dense_grid = ggml_cuda_penalties_grid_size(vocab, block_size);
        k_penalties_dense<<<dense_grid, block_size, 0, stream>>>(
            logits_d, d_counts.get(), vocab, penalty_repeat, penalty_freq, penalty_present);
    } else {
        const int grid_size = ggml_cuda_penalties_grid_size(n_active_h, block_size);
        k_penalties_sparse<<<grid_size, block_size, 0, stream>>>(
            logits_d, ids_d, cnt_d, n_active_h, vocab, penalty_repeat, penalty_freq, penalty_present);
    }

#ifdef GGML_NVTX
    CUDA_CHECK(cudaLaunchHostFunc(stream, ggml_cuda_penalties_nvtx_end, nullptr));
#endif
}
