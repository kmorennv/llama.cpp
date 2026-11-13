#include "nvrtc-kernels.cuh"

#ifdef GGML_CUDA_NVRTC

#include <nvrtc.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <process.h>
static inline int ggml_nvrtc_getpid() { return _getpid(); }
#else
#    include <unistd.h>
static inline int ggml_nvrtc_getpid() { return (int) getpid(); }
#endif

#include "nvrtc-sources.h"

namespace {

thread_local bool preparing = false;
std::atomic<uint64_t> compilation_count{0};
std::atomic<uint64_t> cache_hit_count{0};
std::atomic<uint64_t> compiler_time_us{0};

static bool verbose() {
    static const bool enabled = getenv("GGML_CUDA_NVRTC_VERBOSE") && atoi(getenv("GGML_CUDA_NVRTC_VERBOSE")) != 0;
    return enabled;
}

static void check_nvrtc(nvrtcResult result) {
    if (result != NVRTC_SUCCESS) {
        throw std::runtime_error(nvrtcGetErrorString(result));
    }
}

static void check_driver(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char * message = nullptr;
        cuGetErrorString(result, &message);
        throw std::runtime_error(message ? message : "CUDA driver error");
    }
}

struct compiled_kernel {
    std::vector<char> image;
    std::unordered_map<std::string, std::string> names;
};

struct kernel_job {
    std::string header;
    std::string target;
    std::vector<std::string> expressions;
    std::promise<std::shared_ptr<compiled_kernel>> result;
};

struct loaded_kernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    int shared = 0;
};

static std::string kernel_key(const char * header, const std::string & expression) {
    return std::string(header) + "\n" + expression;
}

static std::string target_for_device(int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    const int major = cc / 100;
    const int minor = (cc % 100) / 10;
    std::string target = "sm_" + std::to_string(major) + std::to_string(minor);
    if (major == 12) {
        target += "a";
    }
    return target;
}

static uint64_t hash_string(const std::string & value) {
    uint64_t result = 0xcbf29ce484222325ULL;
    for (unsigned char c : value) {
        result = (result ^ c) * 0x100000001b3ULL;
    }
    return result;
}

static std::filesystem::path cache_path(const std::string & id) {
    const char * directory = getenv("GGML_CUDA_NVRTC_CACHE_DIR");
    std::filesystem::path path;
    if (directory) {
        if (std::string(directory) == "0") {
            return {};
        }
        path = directory;
    } else if (const char * xdg = getenv("XDG_CACHE_HOME")) {
        path = std::filesystem::path(xdg) / "ggml-nvrtc-kernels";
    } else if (const char * home = getenv("HOME")) {
        path = std::filesystem::path(home) / ".cache/ggml-nvrtc-kernels";
#ifdef _WIN32
    } else if (const char * local = getenv("LOCALAPPDATA")) {
        path = std::filesystem::path(local) / "ggml-nvrtc-kernels";
#endif
    } else {
        return {};
    }
    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
        return {};
    }
    return path / (std::to_string(hash_string(id)) + ".cubin");
}

static std::shared_ptr<compiled_kernel> compile_job(const kernel_job & job) {
    int major = 0, minor = 0;
    check_nvrtc(nvrtcVersion(&major, &minor));
    std::string source = "#include \"" + job.header + "\"\n";
    std::string identity = std::string(nvrtc_source_digest) + "\n" + job.target + "\n" +
        std::to_string(major) + "." + std::to_string(minor) + "\n" + GGML_NVRTC_OPTIONS_ID + "\n" + source;
#ifdef NDEBUG
    identity += "NDEBUG\n";
#endif
    for (const auto & expression : job.expressions) {
        identity += expression + "\n";
    }
    const auto path = cache_path(identity);
    auto result = std::make_shared<compiled_kernel>();

    // The manifest accompanies the image so template names need no recompilation on a cache hit.
    if (!path.empty()) {
        std::ifstream manifest(path.string() + ".names");
        std::string cached_identity, expression, name;
        std::getline(manifest, cached_identity, '\0');
        if (cached_identity == identity) {
            while (std::getline(manifest, expression) && std::getline(manifest, name)) {
                result->names.emplace(expression, name);
            }
            std::ifstream image(path, std::ios::binary | std::ios::ate);
            const auto size = image.tellg();
            if (image && size > 0 && size < 512 * 1024 * 1024 && result->names.size() == job.expressions.size()) {
                result->image.resize((size_t) size);
                image.seekg(0);
                if (image.read(result->image.data(), size)) {
                    cache_hit_count.fetch_add(1, std::memory_order_relaxed);
                    if (verbose()) {
                        GGML_LOG_INFO("NVRTC: loaded %zu %s instances for %s from disk\n", job.expressions.size(), job.header.c_str(), job.target.c_str());
                    }
                    return result;
                }
            }
        }
        result->names.clear();
    }

    const char * shim_names[] = { "cstdint", "type_traits", "limits", "array", "algorithm", "utility", "cassert", "climits", "cfloat",
        "stdbool.h", "stddef.h", "stdint.h", "stdio.h" };
    const char * shims[] = {
        "#pragma once\n#include <cuda/std/cstdint>\nnamespace std { using namespace cuda::std; }\n",
        "#pragma once\n#include <cuda/std/type_traits>\nnamespace std { using cuda::std::is_same; using cuda::std::is_same_v; using cuda::std::true_type; using cuda::std::false_type; using cuda::std::conditional_t; }\n",
        "#pragma once\n#include <cuda/std/limits>\nnamespace std { using cuda::std::numeric_limits; }\n",
        "#pragma once\n#include <cuda/std/array>\nnamespace std { using cuda::std::array; }\n",
        "#pragma once\n#include <cuda/std/algorithm>\nnamespace std { using cuda::std::min; using cuda::std::max; }\n",
        "#pragma once\n#include <cuda/std/utility>\nnamespace std { using cuda::std::pair; }\n",
        "#pragma once\n#include <cuda/std/cassert>\n",
        "#pragma once\n#include <cuda/std/climits>\n",
        "#pragma once\n#include <cuda/std/cfloat>\n",
        "#pragma once\n",
        "#pragma once\n#include <cuda/std/cstddef>\n",
        "#pragma once\n#include <cuda/std/cstdint>\n",
        "#pragma once\nstruct FILE;\n",
    };
    std::vector<const char *> names(std::begin(nvrtc_source_names), std::end(nvrtc_source_names));
    std::vector<const char *> headers(std::begin(nvrtc_source_text), std::end(nvrtc_source_text));
    names.insert(names.end(), std::begin(shim_names), std::end(shim_names));
    headers.insert(headers.end(), std::begin(shims), std::end(shims));
    nvrtcProgram program = nullptr;
    check_nvrtc(nvrtcCreateProgram(&program, source.c_str(), "ggml-kernel.cu", (int) names.size(), headers.data(), names.data()));
    try {
        for (const auto & expression : job.expressions) {
            check_nvrtc(nvrtcAddNameExpression(program, expression.c_str()));
        }
        const std::string arch = "--gpu-architecture=" + job.target;
        const char * options[] = { arch.c_str(), "--std=c++17", "--use_fast_math", "--device-as-default-execution-space",
#ifdef NDEBUG
            "-DNDEBUG",
#endif
            "--modify-stack-limit=false", "-DGGML_CUDA_NVRTC", "-DCUDART_VERSION=" GGML_NVRTC_CUDART_VERSION,
            "-I" GGML_NVRTC_CUDA_INCLUDE, "-I" GGML_NVRTC_CUDA_INCLUDE "/cccl" };
        if (verbose()) {
            GGML_LOG_INFO("NVRTC: compiling %zu %s instances for %s at %.3f s\n", job.expressions.size(), job.header.c_str(), job.target.c_str(), ggml_time_us() / 1e6);
        }
        compilation_count.fetch_add(1, std::memory_order_relaxed);
        const auto compile_start = std::chrono::steady_clock::now();
        const auto status = nvrtcCompileProgram(program, (int) (sizeof(options)/sizeof(options[0])), options);
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - compile_start).count();
        compiler_time_us.fetch_add((uint64_t) elapsed_us, std::memory_order_relaxed);
        if (verbose()) {
            GGML_LOG_INFO("NVRTC: %s for %s finished in %.1f ms (%s) at %.3f s\n", job.header.c_str(), job.target.c_str(), elapsed_us / 1000.0, nvrtcGetErrorString(status), ggml_time_us() / 1e6);
        }
        if (status != NVRTC_SUCCESS) {
            size_t size = 0;
            nvrtcGetProgramLogSize(program, &size);
            std::string log(size, '\0');
            nvrtcGetProgramLog(program, log.data());
            throw std::runtime_error(job.header + " (" + job.target + "): " + log);
        }
        size_t size = 0;
        check_nvrtc(nvrtcGetCUBINSize(program, &size));
        result->image.resize(size);
        check_nvrtc(nvrtcGetCUBIN(program, result->image.data()));
        for (const auto & expression : job.expressions) {
            const char * name = nullptr;
            check_nvrtc(nvrtcGetLoweredName(program, expression.c_str(), &name));
            result->names.emplace(expression, name);
        }
        check_nvrtc(nvrtcDestroyProgram(&program));
    } catch (...) {
        nvrtcDestroyProgram(&program);
        throw;
    }
    if (!path.empty()) {
        // Publish the manifest last; a partial entry is treated as a miss.
        const std::string suffix = ".tmp." + std::to_string(ggml_nvrtc_getpid()) + "." + std::to_string((uintptr_t) result.get());
        const auto temp = path.string() + suffix;
        std::ofstream image(temp, std::ios::binary);
        image.write(result->image.data(), result->image.size());
        image.close();
        std::ofstream manifest(temp + ".names", std::ios::binary);
        manifest.write(identity.c_str(), identity.size() + 1);
        for (const auto & expression : job.expressions) {
            manifest << expression << '\n' << result->names.at(expression) << '\n';
        }
        manifest.close();
        std::error_code error;
        if (image && manifest) {
            std::filesystem::rename(temp, path, error);
            if (!error) {
                std::filesystem::rename(temp + ".names", path.string() + ".names", error);
            }
        }
        std::filesystem::remove(temp, error);
        std::filesystem::remove(temp + ".names", error);
    }
    return result;
}

class kernel_pool {
public:
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::shared_ptr<kernel_job>> queue;
    std::vector<std::thread> workers;
    std::unordered_map<std::string, std::shared_future<std::shared_ptr<compiled_kernel>>> artifacts;
    std::unordered_map<int, std::unordered_map<std::string, std::shared_future<std::shared_ptr<compiled_kernel>>>> required;
    std::unordered_map<CUcontext, std::unordered_map<std::string, loaded_kernel>> functions;
    std::unordered_map<int, std::unordered_map<std::string, int>> shared;
    std::unordered_set<int> prepared;
    bool stopping = false;

    kernel_pool() {
        unsigned count = std::max(1u, std::min(4u, std::thread::hardware_concurrency() / 2));
        if (const char * value = getenv("GGML_CUDA_NVRTC_THREADS")) {
            count = std::max(1, std::min(32, atoi(value)));
        }
        for (unsigned i = 0; i < count; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::shared_ptr<kernel_job> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        ready.wait(lock, [this] { return stopping || !queue.empty(); });
                        if (queue.empty()) {
                            return;
                        }
                        job = queue.front();
                        queue.pop_front();
                    }
                    try {
                        job->result.set_value(compile_job(*job));
                    } catch (...) {
                        job->result.set_exception(std::current_exception());
                    }
                }
            });
        }
    }

    ~kernel_pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto & worker : workers) {
            worker.join();
        }
        for (auto & context : functions) {
            if (cuCtxPushCurrent(context.first) == CUDA_SUCCESS) {
                cuCtxSynchronize();
                std::unordered_set<CUmodule> modules;
                for (auto & entry : context.second) {
                    modules.insert(entry.second.module);
                }
                for (auto module : modules) {
                    cuModuleUnload(module);
                }
                CUcontext previous;
                cuCtxPopCurrent(&previous);
            }
        }
    }
};

static kernel_pool & pool() {
    static kernel_pool instance;
    return instance;
}

static loaded_kernel resolve(const char * header, const std::string & expression, cudaStream_t stream = cudaStreamPerThread, int shared = 0) {
    auto & p = pool();
    CUcontext context = nullptr;
    check_driver(cuCtxGetCurrent(&context));
    int device = ggml_cuda_get_device();
    const std::string key = kernel_key(header, expression);
    thread_local std::unordered_map<CUcontext, std::unordered_map<std::string, loaded_kernel>> local_functions;
    const auto local = local_functions[context].find(key);
    if (local != local_functions[context].end()) {
        return local->second;
    }
    std::shared_future<std::shared_ptr<compiled_kernel>> future;
    {
        std::lock_guard<std::mutex> lock(p.mutex);
        const auto found = p.functions[context].find(key);
        if (found != p.functions[context].end()) {
            local_functions[context].emplace(key, found->second);
            return found->second;
        }
        if (p.prepared.count(device) && !preparing) {
            throw std::runtime_error("NVRTC kernel missing from prepared model: " + key);
        }
        const auto job = p.required[device].find(key);
        if (job != p.required[device].end()) {
            future = job->second;
        }
    }
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        throw std::runtime_error("NVRTC kernel was not loaded before capture: " + key);
    }
    if (!future.valid()) {
        ggml_cuda_nvrtc_enqueue(device, header, { expression });
        std::lock_guard<std::mutex> lock(p.mutex);
        future = p.required[device].at(key);
    }
    const auto artifact = future.get();
    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.functions[context].count(key)) {
        return p.functions[context].at(key);
    }
    p.shared[device][key] = std::max(p.shared[device][key], shared);
    CUmodule module = nullptr;
    check_driver(cuModuleLoadData(&module, artifact->image.data()));
    try {
        for (const auto & entry : artifact->names) {
            loaded_kernel function;
            function.module = module;
            check_driver(cuModuleGetFunction(&function.function, module, entry.second.c_str()));
            function.shared = p.shared[device][kernel_key(header, entry.first)];
            if (function.shared > 0) {
                check_driver(cuFuncSetAttribute(function.function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function.shared));
            } else {
                check_driver(cuFuncGetAttribute(&function.shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function.function));
            }
#if CUDA_VERSION >= 12020
            check_driver(cuFuncLoad(function.function));
#endif
            p.functions[context].emplace(kernel_key(header, entry.first), function);
        }
    } catch (...) {
        for (auto it = p.functions[context].begin(); it != p.functions[context].end();) {
            it = it->second.module == module ? p.functions[context].erase(it) : std::next(it);
        }
        cuModuleUnload(module);
        throw;
    }
    const auto function = p.functions[context].at(key);
    local_functions[context].emplace(key, function);
    return function;
}

static void set_shared(const loaded_kernel & kernel, int shared) {
    if (shared > kernel.shared) {
        throw std::runtime_error("NVRTC kernel exceeds device shared memory limit");
    }
}

} // namespace

uint64_t ggml_cuda_nvrtc_compilation_count() {
    return compilation_count.load(std::memory_order_relaxed);
}

uint64_t ggml_cuda_nvrtc_cache_hit_count() {
    return cache_hit_count.load(std::memory_order_relaxed);
}

void ggml_cuda_nvrtc_require_shared(int device, const char * header, const std::string & expression, int bytes) {
    auto & p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    int & required = p.shared[device][kernel_key(header, expression)];
    required = std::max(required, bytes);
}

void ggml_cuda_nvrtc_enqueue(int device, const char * header, const std::vector<std::string> & expressions) {
    if (expressions.empty()) {
        return;
    }
    const auto target = target_for_device(device);
    auto & p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    auto job = std::make_shared<kernel_job>();
    job->header = header;
    job->target = target;
    for (const auto & expression : expressions) {
        const auto key = kernel_key(header, expression);
        if (p.required[device].count(key)) {
            continue;
        }
        const auto artifact = p.artifacts.find(target + "\n" + key);
        if (artifact != p.artifacts.end()) {
            p.required[device].emplace(key, artifact->second);
        } else {
            job->expressions.push_back(expression);
        }
    }
    if (job->expressions.empty()) {
        return;
    }
    std::sort(job->expressions.begin(), job->expressions.end());
    job->expressions.erase(std::unique(job->expressions.begin(), job->expressions.end()), job->expressions.end());
    const auto future = job->result.get_future().share();
    for (const auto & expression : job->expressions) {
        const auto key = kernel_key(header, expression);
        p.required[device].emplace(key, future);
        p.artifacts.emplace(target + "\n" + key, future);
    }
    p.queue.push_back(job);
    if (verbose()) {
        GGML_LOG_INFO("NVRTC: queued %zu %s instances for %s at %.3f s\n", job->expressions.size(), job->header.c_str(), job->target.c_str(), ggml_time_us() / 1e6);
    }
    p.ready.notify_one();
}

bool ggml_cuda_nvrtc_wait(int device) {
    const int64_t wait_start = ggml_time_us();
    try {
        preparing = true;
        ggml_cuda_set_device(device);
        auto & p = pool();
        CUcontext context = nullptr;
        check_driver(cuCtxGetCurrent(&context));
        size_t ready_count = 0;
        for (;;) {
            std::vector<std::string> pending;
            {
                std::lock_guard<std::mutex> lock(p.mutex);
                for (const auto & job : p.required[device]) {
                    if (!p.functions[context].count(job.first)) {
                        pending.push_back(job.first);
                    }
                }
                if (pending.empty()) {
                    ready_count = p.required[device].size();
                    p.prepared.insert(device);
                    break;
                }
            }
            for (const auto & key : pending) {
                const auto split = key.find('\n');
                resolve(key.substr(0, split).c_str(), key.substr(split + 1));
            }
        }
        preparing = false;
        if (verbose()) {
            GGML_LOG_INFO("NVRTC: readiness wait on CUDA%d took %.1f ms at %.3f s\n", device, (ggml_time_us() - wait_start) / 1000.0, ggml_time_us() / 1e6);
        }
        GGML_LOG_INFO("NVRTC: %zu model kernel instances ready on CUDA%d (process totals: %llu compilations, %llu disk cache hits, %.1f ms compiler time)\n",
            ready_count, device, (unsigned long long) ggml_cuda_nvrtc_compilation_count(), (unsigned long long) ggml_cuda_nvrtc_cache_hit_count(),
            compiler_time_us.load(std::memory_order_relaxed) / 1000.0);
        return true;
    } catch (const std::exception & error) {
        preparing = false;
        GGML_LOG_ERROR("NVRTC preparation failed: %s\n", error.what());
        return false;
    }
}

int ggml_cuda_nvrtc_occupancy(const char * header, const std::string & expression, int block_size, size_t shared) {
    const auto kernel = resolve(header, expression, cudaStreamPerThread, (int) shared);
    set_shared(kernel, (int) shared);
    int blocks = 0;
    check_driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel.function, block_size, shared));
    return blocks;
}

void ggml_cuda_nvrtc_launch_impl(const char * header, const std::string & expression,
        const ggml_cuda_kernel_launch_params & params, int max_dynamic_smem, bool pdl, void ** args) {
    const auto kernel = resolve(header, expression, params.stream, max_dynamic_smem);
    set_shared(kernel, max_dynamic_smem);
#if CUDA_VERSION >= 12000
    static const bool enable_pdl = !getenv("GGML_CUDA_PDL") || atoi(getenv("GGML_CUDA_PDL")) != 0;
    if (pdl && enable_pdl && ggml_cuda_info().devices[ggml_cuda_get_device()].cc >= GGML_CUDA_CC_HOPPER) {
        CUlaunchAttribute attribute = {};
        attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attribute.value.programmaticStreamSerializationAllowed = 1;
        CUlaunchConfig config = {};
        config.gridDimX = params.block_nums.x;
        config.gridDimY = params.block_nums.y;
        config.gridDimZ = params.block_nums.z;
        config.blockDimX = params.block_dims.x;
        config.blockDimY = params.block_dims.y;
        config.blockDimZ = params.block_dims.z;
        config.sharedMemBytes = params.shmem;
        config.hStream = (CUstream) params.stream;
        config.attrs = &attribute;
        config.numAttrs = 1;
        check_driver(cuLaunchKernelEx(&config, kernel.function, args, nullptr));
        return;
    }
#endif
    check_driver(cuLaunchKernel(kernel.function, params.block_nums.x, params.block_nums.y, params.block_nums.z,
        params.block_dims.x, params.block_dims.y, params.block_dims.z, (unsigned) params.shmem,
        (CUstream) params.stream, args, nullptr));
}

#endif // GGML_CUDA_NVRTC
