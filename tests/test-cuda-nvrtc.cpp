#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

static bool compare_output(int, ggml_tensor * actual, ggml_tensor * expected, void * user_data) {
    const size_t n = ggml_nelements(actual);
    std::vector<float> a(n), b(n);
    ggml_backend_tensor_get(actual, a.data(), 0, n*sizeof(float));
    ggml_backend_tensor_get(expected, b.data(), 0, n*sizeof(float));
    double error = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double difference = (double) a[i] - b[i];
        error += difference*difference;
        norm += (double) b[i]*b[i];
    }
    const double relative_error = error / (norm + 1e-30);
    const bool ok = std::isfinite(relative_error) && relative_error <= 5e-4;
    if (!ok) {
        fprintf(stderr, "prepared kernel: NMSE %.9g exceeds 5e-4\n", relative_error);
    }
    *(bool *) user_data = *(bool *) user_data && ok;
    return ok;
}

static bool run_case(ggml_backend_t backend, ggml_backend_t reference, ggml_type type, int64_t rows, int64_t columns, bool moe = false, int64_t k = 256, bool f32_acc = false) {
    const int64_t experts = moe ? 3 : 1;
    const ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }
    ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), type, k, rows, experts);
    ggml_tensor * input = moe ? ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, 2, columns) :
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, columns);
    ggml_tensor * ids = moe ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, columns) : nullptr;
    ggml_tensor * output = moe ? ggml_mul_mat_id(ctx.get(), weights, input, ids) : ggml_mul_mat(ctx.get(), weights, input);
    if (f32_acc) {
        ggml_prec_set_acc(output, GGML_PREC_F32);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        return false;
    }

    std::vector<float> weight_values(k*rows*experts), input_values(ggml_nelements(input));
    for (size_t i = 0; i < weight_values.size(); ++i) {
        weight_values[i] = sinf((float) i*0.13f);
    }
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = cosf((float) i*0.17f);
    }
    std::vector<uint8_t> quantized(ggml_nbytes(weights));
    const std::vector<float> importance(k, 1.0f);
    if (ggml_quantize_chunk(weights->type, weight_values.data(), quantized.data(), 0, rows*experts, k, importance.data()) != quantized.size()) {
        return false;
    }
    ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size()*sizeof(float));
    if (ids) {
        std::vector<int32_t> selected(2*columns);
        for (int64_t token = 0; token < columns; ++token) {
            selected[2*token] = token % experts;
            selected[2*token + 1] = (token + 1) % experts;
        }
        ggml_backend_tensor_set(ids, selected.data(), 0, selected.size()*sizeof(int32_t));
    }

    bool ok = true;
    for (int repetition = 0; repetition < 3 && ok; ++repetition) {
        if (!ggml_backend_compare_graph_backend(backend, reference, graph, compare_output, &ok, &output, 1)) {
            return false;
        }
    }
    printf("prepared %s %s: rows=%lld columns=%lld K=%lld acc=%s %s\n", ggml_type_name(type), moe ? "MoE" : "matmul",
        (long long) rows, (long long) columns, (long long) k, f32_acc ? "F32" : "default", ok ? "OK" : "FAIL");
    return ok;
}

static ggml_tensor * build_auxiliary_graph(ggml_context * ctx, bool attention, int64_t columns, bool softcap = false) {
    if (!attention) {
        ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 128);
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, columns);
        return ggml_mul_mat(ctx, weights, input);
    }
    const int64_t head_size = softcap ? 128 : 64;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_size, columns, 2, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_size, 128, 2, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_size, 128, 2, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, columns);
    ggml_tensor * output = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf((float) head_size), 0.0f, softcap ? 1.0f : 0.0f);
    ggml_prec_set_acc(output, GGML_PREC_F32);
    return output;
}

static bool prepare_auxiliary_graphs(ggml_backend_t backend, ggml_backend_t reference) {
    for (int variant = 0; variant < 3; ++variant) {
        const bool attention = variant != 0;
        const bool softcap = variant == 2;
        const ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
        ggml_context_ptr ctx(ggml_init(params));
        if (!ctx) {
            return false;
        }
        ggml_tensor * output = build_auxiliary_graph(ctx.get(), attention, 16, softcap);
        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, output);
        ggml_backend_t backends[] = { backend, reference };
        ggml_backend_sched_ptr scheduler(ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
        if (!scheduler) {
            return false;
        }
        ggml_backend_sched_set_tensor_backend(scheduler.get(), output, backend);
        if (!ggml_backend_sched_reserve(scheduler.get(), graph)) {
            return false;
        }
    }
    return true;
}

static bool prepare_float_graphs(ggml_backend_t backend, ggml_backend_t reference) {
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F8_E4M3 }) {
        for (bool moe : { false, true }) {
            const ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
            ggml_context_ptr ctx(ggml_init(params));
            ggml_tensor * weights = ggml_new_tensor_3d(ctx.get(), type, 512, 130, moe ? 3 : 1);
            ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 512, moe ? 2 : 8, moe ? 7 : 1);
            ggml_tensor * ids = moe ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, 7) : nullptr;
            ggml_tensor * output = moe ? ggml_mul_mat_id(ctx.get(), weights, input, ids) : ggml_mul_mat(ctx.get(), weights, input);
            ggml_cgraph * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, output);
            ggml_backend_t backends[] = { backend, reference };
            ggml_backend_sched_ptr scheduler(ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
            if (!scheduler) {
                return false;
            }
            ggml_backend_sched_set_tensor_backend(scheduler.get(), output, backend);
            if (!ggml_backend_sched_reserve(scheduler.get(), graph)) {
                return false;
            }
        }
    }
    return true;
}

static bool run_auxiliary_case(ggml_backend_t backend, ggml_backend_t reference, bool attention, int64_t columns, bool softcap = false) {
    const ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }
    ggml_tensor * output = build_auxiliary_graph(ctx.get(), attention, columns, softcap);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        return false;
    }
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx.get()); tensor; tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
        if (tensor->op != GGML_OP_NONE) {
            continue;
        }
        std::vector<float> values(ggml_nelements(tensor));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = attention && tensor == output->src[3] ? 0.0f : 0.1f*sinf((float) i*0.13f);
        }
        if (tensor->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(values.size());
            ggml_fp32_to_fp16_row(values.data(), half.data(), values.size());
            ggml_backend_tensor_set(tensor, half.data(), 0, half.size()*sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(float));
        }
    }
    bool ok = true;
    for (int repetition = 0; repetition < 3 && ok; ++repetition) {
        if (!ggml_backend_compare_graph_backend(backend, reference, graph, compare_output, &ok, &output, 1)) {
            return false;
        }
    }
    printf("prepared %s: columns=%lld softcap=%d %s\n", attention ? "attention" : "F16 matmul", (long long) columns, softcap, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("CUDA0");
    if (!device) {
        printf("NVRTC test skipped: CUDA0 is unavailable\n");
        return 77;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    using prepare_tensor_fn = bool (*)(ggml_backend_dev_t, const ggml_tensor *);
    using prepare_fattn_fn = bool (*)(ggml_backend_dev_t, int, int, bool);
    using prepare_wait_fn = bool (*)(ggml_backend_t);
    using prepare_begin_fn = void (*)(ggml_backend_t);
    using compilation_count_fn = uint64_t (*)();
    auto prepare_tensor = (prepare_tensor_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_prepare_tensor");
    auto prepare_fattn = (prepare_fattn_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_prepare_fattn");
    auto prepare_wait = (prepare_wait_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_prepare_wait");
    auto prepare_begin = (prepare_begin_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_prepare_begin");
    auto compilation_count = (compilation_count_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nvrtc_compilation_count");
    if (!prepare_tensor || !prepare_fattn || !prepare_wait || !prepare_begin || !compilation_count) {
        printf("NVRTC test skipped: CUDA backend has no preparation hooks\n");
        return 77;
    }

    try {
        const ggml_type quant_types[] = {
            GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_Q8_0, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S,
        };
        ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        ggml_backend_ptr reference(ggml_backend_init_by_name("CPU", nullptr));
        const ggml_init_params params = { ggml_tensor_overhead()*17, nullptr, true };
        ggml_context_ptr metadata(ggml_init(params));
        if (!backend || !reference || !metadata) {
            return 1;
        }
        prepare_begin(backend.get());
        for (bool softcap : { false, true }) {
            const int head_size = softcap ? 128 : 64;
            if (!prepare_fattn(device, head_size, head_size, softcap)) {
                return 1;
            }
        }
        for (ggml_type type : quant_types) {
            for (int64_t rows : { 128, 130 }) {
                const ggml_tensor * weights = ggml_new_tensor_2d(metadata.get(), type, 256, rows);
                if (!prepare_tensor(device, weights)) {
                    return 1;
                }
            }
        }
        const ggml_tensor * experts = ggml_new_tensor_3d(metadata.get(), GGML_TYPE_Q4_K, 256, 128, 3);
        if (!prepare_tensor(device, experts)) {
            return 1;
        }
        if (!prepare_auxiliary_graphs(backend.get(), reference.get()) || !prepare_float_graphs(backend.get(), reference.get())) {
            return 1;
        }
        if (!prepare_wait(backend.get())) {
            return 1;
        }
        const uint64_t prepared_compilations = compilation_count();
        prepare_begin(backend.get());
        for (bool softcap : { false, true }) {
            const int head_size = softcap ? 128 : 64;
            if (!prepare_fattn(device, head_size, head_size, softcap)) {
                return 1;
            }
        }
        if (!prepare_auxiliary_graphs(backend.get(), reference.get()) || !prepare_wait(backend.get())) {
            return 1;
        }
        if (compilation_count() != prepared_compilations) {
            fprintf(stderr, "NVRTC recompiled a prepared attention specialization\n");
            return 1;
        }

        // No preparation below: every decode, prefill and tail variant must already be loaded.
        for (ggml_type type : quant_types) {
            for (int64_t rows : { 128, 130 }) {
                for (int64_t columns : { 1, 7, 32, 129 }) {
                    if (!run_case(backend.get(), reference.get(), type, rows, columns)) {
                        return 1;
                    }
                }
            }
        }
        for (int64_t columns : { 1, 7, 32 }) {
            if (!run_case(backend.get(), reference.get(), GGML_TYPE_Q4_K, 128, columns, true)) {
                return 1;
            }
        }
        for (bool attention : { false, true }) {
            for (int64_t columns : { 1, 7, 16 }) {
                if (!run_auxiliary_case(backend.get(), reference.get(), attention, columns)) {
                    return 1;
                }
            }
        }
        for (int64_t columns : { 1, 7, 16 }) {
            if (!run_auxiliary_case(backend.get(), reference.get(), true, columns, true)) {
                return 1;
            }
        }
        for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F8_E4M3 }) {
            for (int64_t columns = 1; columns <= 8; ++columns) {
                if (!run_case(backend.get(), reference.get(), type, 130, columns)) {
                    return 1;
                }
            }
            for (int64_t k : { 64, 128, 192, 256, 320, 384, 448, 512 }) {
                if (!run_case(backend.get(), reference.get(), type, 130, 1, false, k, true)) {
                    return 1;
                }
            }
            for (int64_t columns : { 1, 7 }) {
                if (!run_case(backend.get(), reference.get(), type, 130, columns, true)) {
                    return 1;
                }
            }
        }
        if (compilation_count() != prepared_compilations) {
            fprintf(stderr, "NVRTC compiled a kernel after the preparation barrier\n");
            return 1;
        }
    } catch (const std::exception & error) {
        fprintf(stderr, "NVRTC preparation test failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
