#pragma once

#include "ggml.h"

#include <cstdint>

#define GGML_NVTX_COLOR_PP         0xFF00AA00u // green  - prompt processing (prefill)
#define GGML_NVTX_COLOR_TG         0xFF0066CCu // blue   - token generation (decode)
#define GGML_NVTX_COLOR_SESSION_PP 0xFF44CC44u // light green - request-level prefill
#define GGML_NVTX_COLOR_SESSION_TG 0xFF4488FFu // light blue  - request-level decode
#define GGML_NVTX_COLOR_SAMPLER_BACKEND 0xFFFF8800u // orange - backend (GPU) samplers
#define GGML_NVTX_COLOR_SAMPLER_CPU     0xFFCC44CCu // purple - CPU samplers

#ifdef GGML_NVTX

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_nvtx_init();
GGML_API void ggml_nvtx_range_begin(const char * name, uint32_t color);
GGML_API void ggml_nvtx_range_end();
GGML_API uint64_t ggml_nvtx_range_start(const char * name, uint32_t color);
GGML_API void ggml_nvtx_range_stop(uint64_t id);
GGML_API void ggml_nvtx_mark_impl(const char * name, uint32_t color);
GGML_API void ggml_nvtx_self_test();

#ifdef __cplusplus
}
#endif

struct ggml_nvtx_range {
    ggml_nvtx_range(const char * name, uint32_t color = 0) {
        ggml_nvtx_range_begin(name, color);
    }

    ~ggml_nvtx_range() {
        ggml_nvtx_range_end();
    }

    ggml_nvtx_range(const ggml_nvtx_range &)            = delete;
    ggml_nvtx_range & operator=(const ggml_nvtx_range &) = delete;
};

inline void ggml_nvtx_mark(const char * name, uint32_t color = 0) {
    ggml_nvtx_mark_impl(name, color);
}

#else

struct ggml_nvtx_range {
    explicit ggml_nvtx_range(const char *, uint32_t = 0) {}
};

inline void ggml_nvtx_mark(const char *, uint32_t = 0) {}

inline void ggml_nvtx_init() {}

inline void ggml_nvtx_self_test() {}

#endif
