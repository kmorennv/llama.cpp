#pragma once

#include "common.cuh"

#if defined(GGML_CUDA_NVRTC) && !defined(__CUDACC_RTC__)
#include <string>
#include <vector>

void ggml_cuda_nvrtc_enqueue(int device, const char * header, const std::vector<std::string> & expressions);
void ggml_cuda_nvrtc_require_shared(int device, const char * header, const std::string & expression, int bytes);
bool ggml_cuda_nvrtc_wait(int device);
uint64_t ggml_cuda_nvrtc_compilation_count();
uint64_t ggml_cuda_nvrtc_cache_hit_count();
void ggml_cuda_nvrtc_launch_impl(const char * header, const std::string & expression,
        const ggml_cuda_kernel_launch_params & params, int max_dynamic_smem, bool pdl, void ** args);
int ggml_cuda_nvrtc_occupancy(const char * header, const std::string & expression, int block_size, size_t shared);

template<typename... Args>
static void ggml_cuda_nvrtc_launch(const char * header, const std::string & expression,
        const ggml_cuda_kernel_launch_params & params, int max_dynamic_smem, bool pdl, Args &&... args) {
    void * argv[] = { const_cast<void *>(static_cast<const void *>(&args))... };
    ggml_cuda_nvrtc_launch_impl(header, expression, params, max_dynamic_smem, pdl, argv);
}

void ggml_cuda_nvrtc_mmvq_prepare(int device, ggml_type type, int64_t ncols_x, bool moe);
#endif
