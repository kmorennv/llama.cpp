#include "ggml-nvtx.h"

#ifdef GGML_NVTX

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <nvtx3/nvToolsExt.h>

#include <mutex>

static std::once_flag g_nvtx_once;

static void ggml_nvtx_init_impl() {
    nvtxInitialize(nullptr);
}

static void ggml_nvtx_ensure_init() {
    std::call_once(g_nvtx_once, ggml_nvtx_init_impl);
}

static void ggml_nvtx_push(const char * name, uint32_t color) {
    ggml_nvtx_ensure_init();

    nvtxEventAttributes_t attr{};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = color;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_nvtx_init() {
    ggml_nvtx_ensure_init();
}

GGML_API void ggml_nvtx_range_begin(const char * name, uint32_t color) {
    ggml_nvtx_push(name, color);
}

GGML_API void ggml_nvtx_range_end() {
    ggml_nvtx_ensure_init();
    nvtxRangePop();
}

GGML_API uint64_t ggml_nvtx_range_start(const char * name, uint32_t color) {
    ggml_nvtx_ensure_init();

    nvtxEventAttributes_t attr{};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = color;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    return nvtxRangeStartEx(&attr);
}

GGML_API void ggml_nvtx_range_stop(uint64_t id) {
    ggml_nvtx_ensure_init();
    nvtxRangeEnd(id);
}

GGML_API void ggml_nvtx_mark_impl(const char * name, uint32_t color) {
    ggml_nvtx_ensure_init();

    nvtxEventAttributes_t attr{};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = color;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxMarkEx(&attr);
}

GGML_API void ggml_nvtx_self_test() {
    ggml_nvtx_push("ggml-nvtx-init", GGML_NVTX_COLOR_PP);
    nvtxRangePop();
}

#ifdef __cplusplus
}
#endif

#endif
