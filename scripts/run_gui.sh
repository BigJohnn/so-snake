#!/usr/bin/env bash
#
# 一键起 so-snake GUI:装前端依赖、按需重新构建、起网关、开浏览器。
#
#     scripts/run_gui.sh                  # http://127.0.0.1:8770
#     scripts/run_gui.sh --port 9000      # 其余参数原样转给 serve_gui.py
#     scripts/run_gui.sh --skip-build     # 确定 dist 是新的,别等 npm
#     scripts/run_gui.sh --no-open        # 不动浏览器
#
# npm install / npm run build 只在真的过期时才跑(见下面的 needs_* 判断),所以
# 反复执行这个脚本的代价就是起一个进程。
#
# 解释器默认 .venv/bin/python。这台机器上的 `python` 是 miniconda base,mujoco 和
# PyOpenGL 都是旧版,别拿它跑。用 SO_SNAKE_PYTHON 可以覆盖。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND="$REPO_ROOT/tools/gui/frontend"
DIST="$FRONTEND/dist"

PYTHON="${SO_SNAKE_PYTHON:-$REPO_ROOT/.venv/bin/python}"

skip_build=0
open_browser=1
host="127.0.0.1"
port="8770"
server_args=()

# 本脚本自己的开关吃掉,其余照原样转给 serve_gui.py。--host/--port 两边都要:
# 转过去让服务用,同时留一份自己拼 URL。
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)  skip_build=1; shift ;;
        --no-open)     open_browser=0; shift ;;
        --host)        host="$2"; server_args+=("$1" "$2"); shift 2 ;;
        --host=*)      host="${1#*=}"; server_args+=("$1"); shift ;;
        --port)        port="$2"; server_args+=("$1" "$2"); shift 2 ;;
        --port=*)      port="${1#*=}"; server_args+=("$1"); shift ;;
        -h|--help)
            # 上面那段注释就是 usage,别写第二份。跳过 shebang,到第一个非注释行为止。
            awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
                "${BASH_SOURCE[0]}"
            if [[ -x "$PYTHON" ]]; then
                echo "serve_gui.py 自己的参数:"
                exec "$PYTHON" "$REPO_ROOT/scripts/serve_gui.py" --help
            fi
            exit 0
            ;;
        *)             server_args+=("$1"); shift ;;
    esac
done

die() { echo "run_gui: $*" >&2; exit 1; }

# --- 解释器 -----------------------------------------------------------------

[[ -x "$PYTHON" ]] || die "找不到解释器 $PYTHON
    python3 -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'
  或者用 SO_SNAKE_PYTHON=/path/to/python 指一个。"

PYTHONPATH="$REPO_ROOT/src" "$PYTHON" -c "import so_snake.gui.server" \
    || die "解释器缺依赖,见上面的 ImportError:
    $PYTHON -m pip install -e '.[dev]'    # 仿真 backend 还要 .[sim]"

# --- 前端 -------------------------------------------------------------------

needs_install() {
    [[ -d "$FRONTEND/node_modules" ]] || return 0
    # npm 装完会把 lock 抄进 node_modules/.package-lock.json,比对这个就知道
    # 依赖是不是这份 lock 装出来的。
    [[ "$FRONTEND/package-lock.json" -nt "$FRONTEND/node_modules/.package-lock.json" ]]
}

needs_build() {
    [[ -f "$DIST/index.html" ]] || return 0
    # 任何一个源文件比构建产物新就重建。-newer 比的是 mtime,git checkout 会更新它。
    local newer
    newer="$(find "$FRONTEND/src" "$FRONTEND/index.html" "$FRONTEND/vite.config.ts" \
                  "$FRONTEND/tsconfig.json" "$FRONTEND/package.json" \
                  -newer "$DIST/index.html" -print -quit 2>/dev/null || true)"
    [[ -n "$newer" ]]
}

if (( skip_build )); then
    [[ -f "$DIST/index.html" ]] || die "--skip-build 但 $DIST 里没有构建产物"
else
    command -v npm >/dev/null || die "没有 npm(前端是 React + Vite)。
  装 node:  brew install node
  或者先构建过一次,之后一直用 --skip-build。"

    if needs_install; then
        echo "==> npm install(首次,或 package-lock.json 变了;要几分钟)"
        (cd "$FRONTEND" && npm install)
    fi

    if needs_build; then
        echo "==> npm run build"
        (cd "$FRONTEND" && npm run build)
    else
        echo "==> 前端已是最新,跳过构建"
    fi
fi

# --- 起服务 -----------------------------------------------------------------

url="http://${host}:${port}"
[[ "$host" == "0.0.0.0" || "$host" == "::" ]] && url="http://127.0.0.1:${port}"

if (( open_browser )); then
    opener=""
    command -v open      >/dev/null && opener="open"
    [[ -z "$opener" ]] && command -v xdg-open >/dev/null && opener="xdg-open"

    if [[ -n "$opener" ]]; then
        # 等端口真的接上再开,否则浏览器抢在服务前面拿到连接拒绝。等不到就算了,
        # 服务在前台跑着,URL 上面也打了。
        (
            for _ in $(seq 1 60); do
                if "$PYTHON" -c "
import socket, sys
s = socket.socket()
s.settimeout(0.4)
sys.exit(0 if s.connect_ex(('127.0.0.1', $port)) == 0 else 1)
" 2>/dev/null; then
                    "$opener" "$url"
                    exit 0
                fi
                sleep 0.5
            done
        ) &
    fi
fi

echo "==> $url"
cd "$REPO_ROOT"
# ${a[@]+"${a[@]}"}:bash 3.2(macOS 自带)在 set -u 下展开空数组会报 unbound。
PYTHONPATH="$REPO_ROOT/src" exec "$PYTHON" scripts/serve_gui.py \
    ${server_args[@]+"${server_args[@]}"}
