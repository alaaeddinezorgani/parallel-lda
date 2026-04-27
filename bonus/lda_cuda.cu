#define _POSIX_C_SOURCE 199309L

/*
 * lda_cuda.cu
 * Fisher's LDA — GPU implementation with CUDA
 *
 * Parallelisation strategy: across samples (same as OpenMP Option A).
 *   Each CUDA thread processes one sample row i:
 *     diff = x_i - mu_{c(i)}
 *     contributes diff * diff^T to S_W via atomic adds.
 *
 *   To avoid excessive atomics on the d×d matrix (d=561 → 314 721 elements),
 *   we use a two-level reduction:
 *     1. Each thread block accumulates a SHARED partial S_W in shared memory.
 *     2. One atomicAdd per (r,col) per block to the global S_W.
 *   This keeps atomic pressure low while staying fully parallel.
 *
 * Exported symbols (called via ctypes from Python):
 *   compute_sw_cuda()      — timed GPU S_W kernel
 *   compute_class_means()  — serial (same as serial/parallel versions)
 *   compute_sb()           — serial (same as serial/parallel versions)
 *   lda_project_cuda()     — GPU projection Y = X @ W
 *
 * Compile:
 *   nvcc -O3 -arch=sm_75 --compiler-options -fPIC \
 *        -shared -o lda_cuda.so lda_cuda.cu -lm
 *   (replace sm_75 with your GPU arch: sm_61 for GTX10xx, sm_86 for RTX30xx,
 *    sm_89 for RTX40xx — or use -arch=native if nvcc >= 12.x)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

/* ─── timing ──────────────────────────────────────────────────────────────── */

static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ─── CUDA error check macro ──────────────────────────────────────────────── */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

/* ─── kernel: within-class scatter S_W ───────────────────────────────────── */
/*
 * Each thread handles one sample i.
 * Shared memory holds a partial d×d S_W for the block.
 * After all threads in the block finish, one thread per (r,col) atomically
 * adds the block's partial to the global SW.
 *
 * Shared memory requirement: BLOCK_SIZE * d doubles for diff buffers
 * + d*d doubles for partial SW.  For d=561 that's 561*561*8 ≈ 2.5 MB —
 * too large for a single tile.  We therefore use a TILED approach:
 * process the d×d output in tiles of TILE × TILE columns at a time so
 * that shared memory stays under the 48 KB hardware limit.
 */

/* Tile size for the output matrix — adjust if your GPU has more shared mem */
#define TILE 16
#define BLOCK_DIM 128   /* threads per block (samples) */

/*
 * sw_kernel_tiled
 *
 * Grid:  (ceil(n/BLOCK_DIM), ceil(d/TILE), ceil(d/TILE))
 * Block: (BLOCK_DIM, 1, 1)
 *
 * Each (bx, by, bz) block handles:
 *   - samples   in [bx*BLOCK_DIM, (bx+1)*BLOCK_DIM)
 *   - rows      in [by*TILE,      (by+1)*TILE)       of S_W
 *   - cols      in [bz*TILE,      (bz+1)*TILE)       of S_W
 */
__global__ void sw_kernel_tiled(
        const double * __restrict__ X,       /* [n × d] */
        const int    * __restrict__ labels,  /* [n]     */
        const double * __restrict__ means,   /* [C × d] */
        double       * __restrict__ SW,      /* [d × d] */
        int n, int d,
        int row_start, int row_end,          /* tile row range */
        int col_start, int col_end           /* tile col range */
) {
    /* Shared partial S_W for this tile */
    int tile_rows = row_end - row_start;
    int tile_cols = col_end - col_start;

    /* Use dynamic shared memory: tile_rows * tile_cols doubles */
    extern __shared__ double shared_SW[];  /* [tile_rows × tile_cols] */

    int tid = threadIdx.x;
    int total_tile = tile_rows * tile_cols;

    /* Zero the shared tile */
    for (int k = tid; k < total_tile; k += blockDim.x)
        shared_SW[k] = 0.0;
    __syncthreads();

    /* Each thread processes one sample */
    int i = blockIdx.x * blockDim.x + tid;
    if (i < n) {
        int c = labels[i];
        const double *x_row = X      + (long long)i * d;
        const double *mu_c  = means  + (long long)c * d;

        /* Accumulate into shared tile */
        for (int r = row_start; r < row_end; r++) {
            double dr = x_row[r] - mu_c[r];
            if (dr == 0.0) continue;
            int local_r = r - row_start;
            for (int col = col_start; col < col_end; col++) {
                double dc = x_row[col] - mu_c[col];
                /* atomic add into shared memory */
                atomicAdd(&shared_SW[local_r * tile_cols + (col - col_start)],
                          dr * dc);
            }
        }
    }
    __syncthreads();

    /* Flush shared tile → global SW */
    for (int k = tid; k < total_tile; k += blockDim.x) {
        int local_r   = k / tile_cols;
        int local_col = k % tile_cols;
        int global_r   = row_start + local_r;
        int global_col = col_start + local_col;
        atomicAdd(&SW[(long long)global_r * d + global_col],
                  shared_SW[k]);
    }
}

/* ─── kernel: projection Y = X @ W ───────────────────────────────────────── */
/*
 * Each thread computes one element Y[i, kk].
 * Grid: (ceil(n/16), ceil(K/16))  Block: (16, 16)
 */
#define PROJ_TILE 16

__global__ void project_kernel(
        const double * __restrict__ X,   /* [n × d] */
        const double * __restrict__ W,   /* [d × K] */
        double       * __restrict__ Y,   /* [n × K] */
        int n, int d, int K
) {
    int i  = blockIdx.x * PROJ_TILE + threadIdx.x;
    int kk = blockIdx.y * PROJ_TILE + threadIdx.y;
    if (i >= n || kk >= K) return;

    double s = 0.0;
    const double *x_row = X + (long long)i * d;
    const double *w_col = W + kk;          /* W is [d × K] row-major */
    for (int j = 0; j < d; j++)
        s += x_row[j] * w_col[(long long)j * K];
    Y[(long long)i * K + kk] = s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Exported C functions (called via ctypes)
 * ═══════════════════════════════════════════════════════════════════════════ */

extern "C" {

/* ─── class means (serial — same as serial/openmp versions) ───────────────── */

void compute_class_means(
        const double *X, const int *labels,
        int n, int d, int C,
        double *means,
        int    *counts,
        double *global_mean
) {
    memset(means,       0, (size_t)C * d * sizeof(double));
    memset(counts,      0, (size_t)C     * sizeof(int));
    memset(global_mean, 0, (size_t)d     * sizeof(double));

    for (int i = 0; i < n; i++) {
        int c = labels[i];
        counts[c]++;
        for (int j = 0; j < d; j++)
            means[(long long)c * d + j] += X[(long long)i * d + j];
    }
    for (int c = 0; c < C; c++) {
        for (int j = 0; j < d; j++) {
            means[(long long)c * d + j] /= counts[c];
            global_mean[j] += means[(long long)c * d + j] * counts[c];
        }
    }
    for (int j = 0; j < d; j++)
        global_mean[j] /= n;
}

/* ─── between-class scatter S_B (serial) ─────────────────────────────────── */

void compute_sb(
        const double *means, const double *global_mean,
        const int *counts,
        int d, int C,
        double *SB
) {
    double *diff = (double *)malloc((size_t)d * sizeof(double));
    if (!diff) { fprintf(stderr, "compute_sb: OOM\n"); exit(1); }

    for (int c = 0; c < C; c++) {
        const double *mu_c = means + (long long)c * d;
        for (int j = 0; j < d; j++)
            diff[j] = mu_c[j] - global_mean[j];
        for (int r = 0; r < d; r++) {
            double dr = diff[r] * counts[c];
            if (dr == 0.0) continue;
            double *row_ptr = SB + (long long)r * d;
            for (int col = 0; col < d; col++)
                row_ptr[col] += dr * diff[col];
        }
    }
    free(diff);
}

/* ─── GPU within-class scatter S_W ───────────────────────────────────────── */
/*
 * compute_sw_cuda
 *
 * Allocates GPU memory, launches tiled kernel, copies result back.
 * Returns wall-clock seconds for the GPU work only
 * (H2D transfer + kernel + D2H transfer — the fair comparison point).
 *
 * If you want kernel-only time, pass time_transfers=0 via an env var
 * or add a separate export — but for HPC projects total GPU time is
 * the standard metric.
 */
double compute_sw_cuda(
        const double *X,       /* host [n × d] */
        const int    *labels,  /* host [n]     */
        const double *means,   /* host [C × d] */
        int n, int d, int C,
        double *SW             /* host [d × d] — caller allocates & zeroes */
) {
    size_t sz_X     = (size_t)n * d * sizeof(double);
    size_t sz_lab   = (size_t)n     * sizeof(int);
    size_t sz_means = (size_t)C * d * sizeof(double);
    size_t sz_SW    = (size_t)d * d * sizeof(double);

    /* ── Allocate device memory ───────────────────────────────────────────── */
    double *d_X, *d_means, *d_SW;
    int    *d_labels;

    CUDA_CHECK(cudaMalloc(&d_X,      sz_X));
    CUDA_CHECK(cudaMalloc(&d_labels, sz_lab));
    CUDA_CHECK(cudaMalloc(&d_means,  sz_means));
    CUDA_CHECK(cudaMalloc(&d_SW,     sz_SW));

    /* ── Start timing (includes transfers) ───────────────────────────────── */
    double t0 = get_time_seconds();

    /* ── H2D transfers ───────────────────────────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(d_X,      X,      sz_X,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, labels, sz_lab,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means,  means,  sz_means, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_SW, 0, sz_SW));

    /* ── Launch tiled kernel ─────────────────────────────────────────────── */
    int blocks_n = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    int tiles    = (d + TILE - 1) / TILE;
    size_t shared_bytes = (size_t)TILE * TILE * sizeof(double);

    for (int by = 0; by < tiles; by++) {
        int row_start = by * TILE;
        int row_end   = (row_start + TILE < d) ? row_start + TILE : d;
        int tile_rows = row_end - row_start;

        for (int bz = 0; bz < tiles; bz++) {
            int col_start = bz * TILE;
            int col_end   = (col_start + TILE < d) ? col_start + TILE : d;
            int tile_cols = col_end - col_start;
            size_t smem = (size_t)tile_rows * tile_cols * sizeof(double);

            sw_kernel_tiled<<<blocks_n, BLOCK_DIM, smem>>>(
                d_X, d_labels, d_means, d_SW,
                n, d,
                row_start, row_end,
                col_start, col_end
            );
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── D2H transfer ────────────────────────────────────────────────────── */
    CUDA_CHECK(cudaMemcpy(SW, d_SW, sz_SW, cudaMemcpyDeviceToHost));

    double elapsed = get_time_seconds() - t0;

    /* ── Free device memory ──────────────────────────────────────────────── */
    cudaFree(d_X);
    cudaFree(d_labels);
    cudaFree(d_means);
    cudaFree(d_SW);

    return elapsed;
}

/* ─── GPU projection Y = X @ W ───────────────────────────────────────────── */

void lda_project_cuda(
        const double *X,   /* host [n × d] */
        const double *W,   /* host [d × K] */
        int n, int d, int K,
        double *Y          /* host [n × K] — caller allocates */
) {
    size_t sz_X = (size_t)n * d * sizeof(double);
    size_t sz_W = (size_t)d * K * sizeof(double);
    size_t sz_Y = (size_t)n * K * sizeof(double);

    double *d_X, *d_W, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, sz_X));
    CUDA_CHECK(cudaMalloc(&d_W, sz_W));
    CUDA_CHECK(cudaMalloc(&d_Y, sz_Y));

    CUDA_CHECK(cudaMemcpy(d_X, X, sz_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, sz_W, cudaMemcpyHostToDevice));

    dim3 block(PROJ_TILE, PROJ_TILE);
    dim3 grid((n + PROJ_TILE - 1) / PROJ_TILE,
              (K + PROJ_TILE - 1) / PROJ_TILE);

    project_kernel<<<grid, block>>>(d_X, d_W, d_Y, n, d, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, sz_Y, cudaMemcpyDeviceToHost));

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);
}

/* ─── query GPU name (for notebook display) ──────────────────────────────── */

void get_gpu_name(char *buf, int buf_len) {
    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, 0);
    if (e == cudaSuccess)
        snprintf(buf, buf_len, "%s", prop.name);
    else
        snprintf(buf, buf_len, "unknown");
}

} /* extern "C" */
