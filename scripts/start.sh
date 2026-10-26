#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'EOF'
启动 Kitchen Inventory Agent 展示服务。

用法：
  ./scripts/start.sh [--port PORT] [--no-reload]

选项：
  --port PORT    监听端口，默认为 8000
  --no-reload    禁用 Uvicorn 自动重载
  -h, --help     显示帮助
EOF
}

validate_port() {
    if [[ ! "$1" =~ ^[0-9]+$ ]] || ((10#$1 < 1 || 10#$1 > 65535)); then
        echo "端口必须是 1 到 65535 之间的整数。" >&2
        exit 2
    fi
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/.." && pwd -P)
python_bin="$project_root/.venv/bin/python"
run_dir="$project_root/.run"
port=8000
reload=true

while (($# > 0)); do
    case "$1" in
        --port)
            if (($# < 2)); then
                echo "--port 需要一个端口值。" >&2
                exit 2
            fi
            port=$2
            shift 2
            ;;
        --no-reload)
            reload=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数：$1" >&2
            show_help >&2
            exit 2
            ;;
    esac
done

validate_port "$port"

if [[ ! -x "$python_bin" ]]; then
    echo "未找到 Linux 虚拟环境：$python_bin" >&2
    echo "请先运行：python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-xpu.txt" >&2
    exit 1
fi

mkdir -p -- "$run_dir"
pid_file="$run_dir/kitchen-agent-$port.pid"
log_file="$run_dir/kitchen-agent-$port.log"

if [[ -f "$pid_file" ]]; then
    existing_pid=$(<"$pid_file")
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Kitchen Inventory Agent 已在端口 $port 运行（PID $existing_pid）。" >&2
        exit 1
    fi
    rm -f -- "$pid_file"
fi

if ! "$python_bin" - "$port" 2>/dev/null <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
PY
then
    echo "端口 $port 已被占用。" >&2
    exit 1
fi

arguments=(
    -m uvicorn app.web:app
    --host 127.0.0.1
    --port "$port"
)
if [[ "$reload" == true ]]; then
    arguments+=(--reload)
fi

(
    cd -- "$project_root"
    nohup setsid -- "$python_bin" "${arguments[@]}" < /dev/null >"$log_file" 2>&1 &
    echo "$!" >"$pid_file"
)
pid=$(<"$pid_file")

for ((attempt = 0; attempt < 40; attempt++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f -- "$pid_file"
        echo "Kitchen Inventory Agent 启动期间退出，日志：$log_file" >&2
        tail -n 20 -- "$log_file" >&2 || true
        exit 1
    fi

    if "$python_bin" - "$port" <<'PY' 2>/dev/null
import sys
from urllib.request import urlopen

with urlopen(f"http://127.0.0.1:{sys.argv[1]}/api/status", timeout=0.5) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
    then
        echo "Kitchen Inventory Agent 已启动（PID $pid）。"
        echo "访问：http://127.0.0.1:$port"
        echo "日志：$log_file"
        exit 0
    fi
    sleep 0.25
done

kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
rm -f -- "$pid_file"
echo "Kitchen Inventory Agent 未能在端口 $port 就绪，日志：$log_file" >&2
tail -n 20 -- "$log_file" >&2 || true
exit 1
