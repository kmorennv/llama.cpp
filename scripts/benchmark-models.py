import os

path_unsloth = "/mnt/share/gguf/unsloth"
path_nvidia = "/mnt/share/gguf/nvidia"
path_ggml_org = "/mnt/share/gguf/ggml-org"

models_1 = [
f"{path_unsloth}/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
f"{path_unsloth}/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
f"{path_unsloth}/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf",
f"{path_unsloth}/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf",
f"{path_nvidia}/Qwen3.6-35B-A3B-2.06GB-per-token-CT/Qwen3.6-35B-A3B-2.06GB-per-token-CT_fp8_q8.gguf",
f"{path_nvidia}/Gemma-4-26B-A4B-NVFP4/Gemma-4-26B-A4B-NVFP4_fp8_q8.gguf",
f"{path_nvidia}/Qwen3.6-27B-NVFP4/Qwen3.6-27B-NVFP4_fp8_q8.gguf",
f"{path_nvidia}/Gemma-4-26B-A4B-NVFP4/Gemma-4-26B-A4B-NVFP4_fp8_q8.gguf",
f"{path_nvidia}/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4_fp8_q8.gguf",
f"{path_nvidia}/Gemma-4-31B-IT-NVFP4/Gemma-4-31B-IT-NVFP4_fp8_q8.gguf",
f"{path_ggml_org}/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf",
]

models = [
"D:/models/gpt-oss-20b-mxfp4.gguf",
"D:/models/Qwen3.6-27B-Q4_K_M.gguf",
]

args = "-dio 1 -fa 1 -ub 512,1024 -p 256,512,1024,2048,4096,8192,16384 -n 256,512,1024 -r 10"

benchmark_binary = "D:/Projects/llama.cpp/build/bin/Release/llama-bench.exe"
log_file_path = "benchmark.log"

with open(log_file_path, "w") as log_file:
    for model in models:
        if os.path.exists(model):
            os.system(f"{benchmark_binary} --model {model} {args} >> {log_file_path} 2>&1")
        else:
            log_file.write(f"Model does not exist: {model}\n")