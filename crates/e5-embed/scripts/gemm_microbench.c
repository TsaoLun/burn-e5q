/* Flex-style i32 GEMM vs packed AVX512-VNNI u8×i8→i32, e5 FFN/QKV shapes.
 *
 *   gcc -O3 -march=native -o /tmp/gemm_microbench crates/e5-embed/scripts/gemm_microbench.c
 *   /tmp/gemm_microbench
 *
 * The VNNI path is a packed vpdpbusd inner kernel (not production quality).
 * It exists to show the instruction-level gap vs burn-flex's transpose+i32-dot.
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int32_t dot_i32(const int32_t *a, const int32_t *b, int k) {
    int32_t s = 0;
    for (int i = 0; i < k; i++) s += a[i] * b[i];
    return s;
}

static void gemm_i32_flex(const int32_t *A, const int32_t *B, int32_t *C,
                          int M, int N, int K) {
    int32_t *Bt = (int32_t *)malloc((size_t)K * N * sizeof(int32_t));
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            Bt[j * K + i] = B[i * N + j];
    for (int i = 0; i < M; i++) {
        const int32_t *Ai = A + i * K;
        for (int j = 0; j < N; j++)
            C[i * N + j] = dot_i32(Ai, Bt + j * K, K);
    }
    free(Bt);
}

#if defined(__AVX512VNNI__)
/* Pack B[K,N] i8 into [N/16][K/4][16 lanes of 4 K-bytes] for vpdpbusd. */
static int8_t *pack_b_vnni(const int8_t *B, int N, int K) {
    int8_t *P = (int8_t *)aligned_alloc(64, (size_t)N * K);
    for (int j = 0; j < N; j += 16) {
        int8_t *block = P + (j / 16) * (K * 16);
        for (int k = 0; k < K; k += 4) {
            int8_t *dst = block + (k / 4) * 64;
            for (int lane = 0; lane < 16; lane++)
                for (int kk = 0; kk < 4; kk++)
                    dst[lane * 4 + kk] = B[(k + kk) * N + j + lane];
        }
    }
    return P;
}

static void gemm_u8i8_vnni_packed(const uint8_t *A, const int8_t *Bp, int32_t *C,
                                  int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 16) {
            const int8_t *bcol = Bp + (j / 16) * (K * 16);
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < K; k += 4) {
                uint32_t packed;
                memcpy(&packed, A + i * K + k, 4);
                __m512i a = _mm512_set1_epi32((int)packed);
                __m512i b = _mm512_load_si512(bcol + (k / 4) * 64);
                acc = _mm512_dpbusd_epi32(acc, a, b);
            }
            _mm512_storeu_si512(C + i * N + j, acc);
        }
    }
}
#endif

static void fill_i32(int32_t *p, int n, int seed) {
    for (int i = 0; i < n; i++) p[i] = (seed + i * 17) % 255 - 128;
}

int main(void) {
    struct {
        const char *name;
        int M, N, intK;
    } shapes[] = {
        {"ffn1  [512,384]x[384,1536]", 512, 1536, 384},
        {"ffn2  [512,1536]x[1536,384]", 512, 384, 1536},
        {"qkv   [512,384]x[384,384]", 512, 384, 384},
        {"short ffn1 [16,384]x[384,1536]", 16, 1536, 384},
    };

    printf("compiler native:"
#ifdef __AVX512VNNI__
           " AVX512-VNNI"
#endif
#ifdef __AVX512F__
           " AVX512F"
#endif
           "\n");

    for (int s = 0; s < 4; s++) {
        int M = shapes[s].M, N = shapes[s].N, K = shapes[s].intK;
        int32_t *A = (int32_t *)malloc((size_t)M * K * sizeof(int32_t));
        int32_t *B = (int32_t *)malloc((size_t)K * N * sizeof(int32_t));
        int32_t *C = (int32_t *)malloc((size_t)M * N * sizeof(int32_t));
        fill_i32(A, M * K, 1);
        fill_i32(B, K * N, 2);

        gemm_i32_flex(A, B, C, M, N, K);
        double t0 = now_ms();
        gemm_i32_flex(A, B, C, M, N, K);
        double i32_ms = now_ms() - t0;
        double gmac = (1.0 * M * N * K) / 1e9;
        printf("%-32s  i32-flex         %8.2f ms  %7.2f GMAC/s\n",
               shapes[s].name, i32_ms, gmac / (i32_ms / 1e3));

#if defined(__AVX512VNNI__)
        uint8_t *Au = (uint8_t *)malloc((size_t)M * K);
        int8_t *Bi = (int8_t *)malloc((size_t)K * N);
        int32_t *Cv = (int32_t *)malloc((size_t)M * N * sizeof(int32_t));
        for (int i = 0; i < M * K; i++) Au[i] = (uint8_t)(A[i] & 255);
        for (int i = 0; i < K * N; i++) Bi[i] = (int8_t)B[i];
        int8_t *Bp = pack_b_vnni(Bi, N, K);
        gemm_u8i8_vnni_packed(Au, Bp, Cv, M, N, K);
        t0 = now_ms();
        gemm_u8i8_vnni_packed(Au, Bp, Cv, M, N, K);
        double v_ms = now_ms() - t0;
        printf("%-32s  u8i8-vnni-packed %8.2f ms  %7.2f GMAC/s  (%.1fx vs i32)\n",
               shapes[s].name, v_ms, gmac / (v_ms / 1e3), i32_ms / v_ms);
        free(Au);
        free(Bi);
        free(Bp);
        free(Cv);
#endif
        free(A);
        free(B);
        free(C);
    }
    return 0;
}
