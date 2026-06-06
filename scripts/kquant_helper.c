#include <stdint.h>
#include <stddef.h>
#include "ggml-quants.h"

size_t qwen3_quantize_q4_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_q4_K(src, dst, nrows, n_per_row, NULL);
}

size_t qwen3_quantize_q5_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_q5_K(src, dst, nrows, n_per_row, NULL);
}

size_t qwen3_quantize_q6_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return quantize_q6_K(src, dst, nrows, n_per_row, NULL);
}
