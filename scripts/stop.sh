#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'EOF'
停止由 scripts/start.sh 启动的 Kitchen Inventory Agent 展示服务。

用法：
  ./scripts/stop.sh [--port PORT]

选项：
  --port PORT    服务端口，默认为 8000
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
port=8000

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

pid_file="$project_root/.run/kitchen-agent-$port.pid"
if [[ ! -f "$pid_file" ]]; then
    echo "Kitchen Inventory Agent 未在端口 $port 运行。"
    exit 0
fi

pid=$(<"$pid_file")
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    rm -f -- "$pid_file"
    echo "PID 文件无效，已清理：$pid_file" >&2
    exit 1
fi

if ! kill -0 "$pid" 2>/dev/null; then
    rm -f -- "$pid_file"
    echo "Kitchen Inventory Agent 未在端口 $port 运行；已清理失效 PID 文件。"
    exit 0
fi

cmdline_file="/proc/$pid/cmdline"
cmdline=""
if [[ -r "$cmdline_file" ]]; then
    cmdline=$(tr '\0' ' ' <"$cmdline_file")
fi
if [[ "$cmdline" != *"$project_root/.venv/bin/python"* || "$cmdline" != *"uvicorn app.web:app"* ]]; then
    echo "拒绝停止 PID $pid：该进程不是当前项目的 Kitchen Inventory Agent。" >&2
    exit 1
fi

kill "$pid"
for ((attempt = 0; attempt < 40; attempt++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f -- "$pid_file"
        echo "Kitchen Inventory Agent 已停止（PID $pid）。"
        exit 0
    fi
    sleep 0.15
done

kill -KILL "$pid" 2>/dev/null || true
rm -f -- "$pid_file"
echo "Kitchen Inventory Agent 已强制停止（PID $pid）。"
