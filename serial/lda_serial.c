#define _POSIX_C_SOURCE 199309

/*
 * lda_serial.c
 * Fisher's LDA — serial implementation
 * Computes within-class (S_W) and between-class (S_B) scatter matrices
 * from scratch, no parallel constructs.
 *
 * Exported symbols (called via ctypes from Python):
 *   compute_class_means()
 *   compute_sw()
 *   compute_sb()
 *   lda_project()
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ─── helpers ─────────────────────────────────────────────────────────────── */

static inline double *alloc_matrix(int rows, int cols) {
    double *m = (double *)calloc(rows * cols, sizeof(double));
    if (!m) { fprintf(stderr, "alloc_matrix: OOM\n"); exit(1); }
    return m;
}

/* ─── timing ──────────────────────────────────────────────────────────────── */

double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ─── class means ─────────────────────────────────────────────────────────── */

/*
 * compute_class_means
 *
 * Inputs:
 *   X          [n × d]  row-major data matrix
 *   labels     [n]      integer class labels  0 … C-1
 *   n, d, C    dimensions
 *
 * Outputs:
 *   means      [C × d]  per-class mean (caller allocates)
 *   counts     [C]      per-class sample count (caller allocates)
 *   global_mean[d]      global mean (caller allocates)
 */
void compute_class_means(
        const double *X, const int *labels,
        int n, int d, int C,
        double *means,          /* [C × d] */
        int    *counts,         /* [C]     */
        double *global_mean     /* [d]     */
) {
    memset(means,       0, C * d * sizeof(double));
    memset(counts,      0, C     * sizeof(int));
    memset(global_mean, 0,     d * sizeof(double));

    for (int i = 0; i < n; i++) {
        int c = labels[i];
        counts[c]++;
        for (int j = 0; j < d; j++) {
            means[c * d + j] += X[i * d + j];
        }
    }

    for (int c = 0; c < C; c++) {
        for (int j = 0; j < d; j++) {
            means[c * d + j] /= counts[c];
            global_mean[j]   += means[c * d + j] * counts[c];
        }
    }
    for (int j = 0; j < d; j++)
        global_mean[j] /= n;
}

/* ─── within-class scatter S_W ────────────────────────────────────────────── */

/*
 * compute_sw
 *
 * S_W = Σ_c Σ_{i in c} (x_i - μ_c)(x_i - μ_c)^T
 *
 * Returns wall-clock seconds spent in the core accumulation loop.
 */
double compute_sw(
        const double *X, const int *labels,
        const double *means,
        int n, int d, int C,
        double *SW              /* [d × d] — caller allocates & zeroes */
) {
    double *diff = (double *)malloc(d * sizeof(double));
    if (!diff) { fprintf(stderr, "compute_sw: OOM\n"); exit(1); }

    double t0 = get_time_seconds();

    for (int i = 0; i < n; i++) {
        int c = labels[i];
        /* diff = x_i - μ_c */
        for (int j = 0; j < d; j++)
            diff[j] = X[i * d + j] - means[c * d + j];

        /* SW += diff * diff^T  (outer product) */
        for (int r = 0; r < d; r++) {
            for (int col = 0; col < d; col++) {
                SW[r * d + col] += diff[r] * diff[col];
            }
        }
    }

    double elapsed = get_time_seconds() - t0;
    free(diff);
    return elapsed;
}

/* ─── between-class scatter S_B ───────────────────────────────────────────── */

/*
 * compute_sb
 *
 * S_B = Σ_c n_c (μ_c - μ)(μ_c - μ)^T
 */
void compute_sb(
        const double *means, const double *global_mean,
        const int *counts,
        int d, int C,
        double *SB              /* [d × d] — caller allocates & zeroes */
) {
    double *diff = (double *)malloc(d * sizeof(double));
    if (!diff) { fprintf(stderr, "compute_sb: OOM\n"); exit(1); }

    for (int c = 0; c < C; c++) {
        for (int j = 0; j < d; j++)
            diff[j] = means[c * d + j] - global_mean[j];

        for (int r = 0; r < d; r++)
            for (int col = 0; col < d; col++)
                SB[r * d + col] += counts[c] * diff[r] * diff[col];
    }

    free(diff);
}

/* ─── projection ──────────────────────────────────────────────────────────── */

/*
 * lda_project
 *
 * Projects X [n × d] onto W [d × k] → Y [n × k]
 * Y = X @ W
 */
void lda_project(
        const double *X, const double *W,
        int n, int d, int k,
        double *Y               /* [n × k] — caller allocates */
) {
    for (int i = 0; i < n; i++)
        for (int kk = 0; kk < k; kk++) {
            double s = 0.0;
            for (int j = 0; j < d; j++)
                s += X[i * d + j] * W[j * k + kk];
            Y[i * k + kk] = s;
        }
}

/* ─── standalone smoke-test (optional) ────────────────────────────────────── */
#ifdef SERIAL_MAIN
int main(void) {
    printf("lda_serial.c compiled OK — use via ctypes from Python.\n");
    return 0;
}
#endif
