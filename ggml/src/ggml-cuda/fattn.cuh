#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

#ifdef GGML_CUDA_NVRTC
void ggml_cuda_nvrtc_fattn_prepare_mma(int device, int DKQ, int DV, bool use_softcap);
void ggml_cuda_nvrtc_fattn_prepare(int device, const ggml_tensor * dst);
#endif
