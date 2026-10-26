#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'EOF'
在 Linux 上构建 FastText 0.9.2 CLI。

用法：
  ./classifier/build_fasttext.sh

构建产物位于 .run/fasttext-linux-build/fasttext。
EOF
}

if (($# > 0)); then
    case "$1" in
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
fi

for command_name in git cmake; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "缺少构建命令：$command_name" >&2
        exit 1
    fi
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd -- "$script_dir/.." && pwd -P)
source_dir="$project_root/.run/fasttext-src"
build_dir="$project_root/.run/fasttext-linux-build"

mkdir -p -- "$project_root/.run"
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --depth 1 --branch v0.9.2 \
        https://github.com/facebookresearch/fastText.git \
        "$source_dir"
fi

# FastText 0.9.2 uses uint64_t without including cstdint on newer GCC versions.
sed -i '/^#include <cstdint>$/d; /^#include <unordered_map>$/a #include <cstdint>' \
    "$source_dir/src/args.cc"

cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$build_dir" --config Release --target fasttext-bin

binary="$build_dir/fasttext"
if [[ ! -x "$binary" ]]; then
    echo "FastText 构建完成，但未找到二进制：$binary" >&2
    exit 1
fi

printf '%s\n' "$binary"
