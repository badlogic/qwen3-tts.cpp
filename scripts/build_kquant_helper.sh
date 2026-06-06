#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGML_BUILD_DIR="${GGML_BUILD_DIR:-$ROOT_DIR/ggml/build}"
UNAME="$(uname -s)"

case "$UNAME" in
  Darwin)
    LIB_EXT="dylib"
    ;;
  Linux)
    LIB_EXT="so"
    ;;
  *)
    echo "Unsupported platform: $UNAME" >&2
    exit 1
    ;;
esac

if [[ ! -f "$GGML_BUILD_DIR/src/libggml-base.$LIB_EXT" ]]; then
  cmake -S "$ROOT_DIR/ggml" -B "$GGML_BUILD_DIR" -DGGML_METAL=OFF
  cmake --build "$GGML_BUILD_DIR" --target ggml-base -j "${JOBS:-4}"
fi

cc -shared -fPIC -O3 \
  -I"$ROOT_DIR/ggml/include" \
  -I"$ROOT_DIR/ggml/src" \
  "$ROOT_DIR/scripts/kquant_helper.c" \
  "$ROOT_DIR/ggml/src/ggml-quants.c" \
  -L"$GGML_BUILD_DIR/src" \
  -lggml-base \
  -Wl,-rpath,"$GGML_BUILD_DIR/src" \
  -o "$ROOT_DIR/scripts/libqwen3_kquant.$LIB_EXT"

echo "$ROOT_DIR/scripts/libqwen3_kquant.$LIB_EXT"
