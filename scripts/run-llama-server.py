#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_SERVER = "./build/bin/llama-server"
DEFAULT_MODEL = "/mnt/share/gguf/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def parse_args():
    parser = argparse.ArgumentParser(description="Start llama-server and wait until it is ready.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Path to llama-server.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to the GGUF model.")
    parser.add_argument("--port", type=int, default=8033, help="Port passed to llama-server.")
    parser.add_argument("--parallel", type=int, default=1, help="Value for -np.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Value for -b.")
    parser.add_argument("--ubatch-size", type=int, default=4096, help="Value for -ub.")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Value for --repeat-penalty.")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Value for --presence-penalty.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--backend-sampling",
        dest="backend_sampling",
        action="store_true",
        default=True,
        help="Pass -bs to llama-server.",
    )
    group.add_argument(
        "--no-backend-sampling",
        dest="backend_sampling",
        action="store_false",
        help="Do not pass -bs to llama-server.",
    )
    parser.add_argument("--log-file", default="server.log", help="File that receives llama-server output.")
    parser.add_argument("--append-log", action="store_true", help="Append to the log instead of replacing it.")
    parser.add_argument("--ready-timeout", type=float, default=120.0, help="Seconds to wait for the ready line.")
    parser.add_argument("--request-interval", type=float, default=0.0, help="Seconds to wait between chat requests.")
    parser.add_argument(
        "--curl-requests",
        "--requests",
        dest="requests",
        type=int,
        default=0,
        help="Number of curl chat requests to send. Use 0 for no limit.",
    )
    parser.add_argument(
        "--ready-text",
        default=None,
        help="Text to wait for. Defaults to the expected localhost listening message for --port.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra llama-server arguments. Use '--' before the extra arguments.",
    )
    return parser.parse_args()


def find_key(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_key(child, key)
            if found is not None:
                return found
    return None


def print_stats(values, stats_path):
    if not values:
        lines = ["no predicted_per_second values collected"]
        print(lines[0], file=sys.stderr)
        stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    values_sorted = sorted(values)
    mid = len(values_sorted) // 2
    if len(values_sorted) % 2 == 0:
        median = (values_sorted[mid - 1] + values_sorted[mid]) / 2
    else:
        median = values_sorted[mid]

    lines = [
        "predicted_per_second summary:",
        f"  calls:  {len(values)}",
        f"  min:    {min(values)}",
        f"  max:    {max(values)}",
        f"  avg:    {sum(values) / len(values)}",
        f"  median: {median}",
    ]
    for line in lines:
        print(line, flush=True)
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary written to {stats_path}", flush=True)


def run_chat_request(args):
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise and practical AI assistant.",
            },
            {
                "role": "user",
                "content": "Explain KV cache in llama.cpp in 5 bullet points.",
            },
        ],
        "stream": False,
    }
    result = subprocess.run(
        [
            "curl",
            "-sS",
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(payload),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"curl failed with exit code {result.returncode}: {result.stderr.strip()}", file=sys.stderr)
        return None

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"failed to parse chat response as JSON: {exc}", file=sys.stderr)
        return None

    predicted_per_second = find_key(response, "predicted_per_second")
    if predicted_per_second is None:
        print("predicted_per_second not found in chat response", file=sys.stderr)
        return None

    try:
        predicted_per_second = float(predicted_per_second)
    except (TypeError, ValueError):
        print(f"predicted_per_second is not numeric: {predicted_per_second}", file=sys.stderr)
        return None

    print(f"predicted_per_second: {predicted_per_second}", flush=True)
    return predicted_per_second


def stop_process(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    args = parse_args()
    ready_text = args.ready_text or f"listening on http://127.0.0.1:{args.port}"
    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    stats_suffix = "BS" if args.backend_sampling else "NO_BS"
    stats_path = Path(f"{Path(args.model).stem}_{stats_suffix}.log")

    cmd = [
        args.server,
        "-m",
        args.model,
        "-dio",
        "--port",
        str(args.port),
        "-np",
        str(args.parallel),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "--repeat-penalty",
        str(args.repeat_penalty),
        "--presence-penalty",
        str(args.presence_penalty),
    ]
    if args.backend_sampling:
        cmd.append("-bs")
    cmd.extend(extra_args)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.append_log else "w"
    ready = threading.Event()
    output_done = threading.Event()

    print(f"starting llama-server; writing output to {log_path}", flush=True)
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        print(f"failed to start llama-server: {exc}", file=sys.stderr)
        return 1

    def copy_output():
        assert process.stdout is not None
        with log_path.open(log_mode, encoding="utf-8") as log_file:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                if ready_text in line:
                    ready.set()
        output_done.set()

    output_thread = threading.Thread(target=copy_output, daemon=True)
    output_thread.start()

    start = time.monotonic()
    values = []
    try:
        while not ready.is_set():
            exit_code = process.poll()
            if exit_code is not None:
                output_done.wait(timeout=1)
                print(
                    f"llama-server exited before it was ready with exit code {exit_code}; see {log_path}",
                    file=sys.stderr,
                )
                return exit_code or 1

            if args.ready_timeout > 0 and time.monotonic() - start > args.ready_timeout:
                stop_process(process)
                print(f"timed out waiting for '{ready_text}'; see {log_path}", file=sys.stderr)
                return 1

            time.sleep(0.1)

        print(f"server is ready: http://127.0.0.1:{args.port}", flush=True)
        request_count = 0
        while process.poll() is None and (args.requests <= 0 or request_count < args.requests):
            request_count += 1
            predicted_per_second = run_chat_request(args)
            if predicted_per_second is not None:
                values.append(predicted_per_second)
            if args.request_interval > 0:
                time.sleep(args.request_interval)

        print_stats(values, stats_path)
        if process.poll() is None and args.requests > 0:
            stop_process(process)
            output_done.wait(timeout=1)
            return 0

        exit_code = process.wait()
        output_done.wait(timeout=1)
        return exit_code
    except KeyboardInterrupt:
        print("\nstopping llama-server...", file=sys.stderr)
        print_stats(values, stats_path)
        stop_process(process)
        output_done.wait(timeout=1)
        return 130


if __name__ == "__main__":
    sys.exit(main())
