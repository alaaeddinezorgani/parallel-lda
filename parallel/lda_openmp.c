#define _POSIX_C_SOURCE 199309L

/*
 * lda_openmp.c
 * Fisher's LDA — parallel implementation with OpenMP
 *
 * Parallelisation strategy: Option A — across samples.
 *   Each thread processes a contiguous chunk of the n rows,
 *   accumulates a PRIVATE partial S_W matrix, then all
 *   partial matrices are reduced (summed) into the final S_W.
 *   This avoids false sharing and costly critical sections on
 *   the inner loop.
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o lda_openmp.so lda_openmp.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* ─── timing ──────────────────────────────────────────────────────────────── */

double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ─── class means (serial — not the bottleneck) ───────────────────────────── */

void compute_class_means(
        const double *X, const int *labels,
        int n, int d, int C,
        double *means,
        int    *counts,
        double *global_mean
) {
    memset(means,       0, C * d * sizeof(double));
    memset(counts,      0, C     * sizeof(int));
    memset(global_mean, 0,     d * sizeof(double));

    for (int i = 0; i < n; i++) {
        int c = labels[i];
        counts[c]++;
        for (int j = 0; j < d; j++)
            means[c * d + j] += X[i * d + j];
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

/* ─── parallel within-class scatter S_W ──────────────────────────────────── */

/*
 * compute_sw_parallel
 *
 * Optimized version:
 *   - All memory allocated BEFORE timing starts
 *   - Per-thread diff buffers pre-allocated
 *   - Row-pointer optimization for inner loop
 *   - Uses num_threads directly (no omp_get_num_threads inside timed section)
 *
 * Returns wall-clock seconds for the parallel region only.
 */
double compute_sw_parallel(
        const double *X, const int *labels,
        const double *means,
        int n, int d, int C,
        double *SW,             /* [d × d] output — caller allocates & zeroes */
        int num_threads
) {
    long long dd = (long long)d * d;
    int actual_threads = num_threads;
    
    /* ── Pre-allocate per-thread buffers BEFORE timing ────────────────────── */
    double **SW_private = (double **)malloc(actual_threads * sizeof(double *));
    double **diff_buffers = (double **)malloc(actual_threads * sizeof(double *));
    
    for (int t = 0; t < actual_threads; t++) {
        SW_private[t] = (double *)calloc(dd, sizeof(double));
        diff_buffers[t] = (double *)malloc(d * sizeof(double));
        if (!SW_private[t] || !diff_buffers[t]) {
            fprintf(stderr, "compute_sw_parallel: OOM thread %d\n", t);
            exit(1);
        }
    }
    
    /* ── Start timing ─────────────────────────────────────────────────────── */
    double t0 = get_time_seconds();
    
    omp_set_num_threads(actual_threads);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *my_SW = SW_private[tid];
        double *diff = diff_buffers[tid];
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int c = labels[i];
            const double *x_row = X + (i * d);
            const double *mu_c = means + (c * d);
            
            /* diff = x_i - μ_c */
            for (int j = 0; j < d; j++) {
                diff[j] = x_row[j] - mu_c[j];
            }
            
            /* Outer product accumulation into private buffer */
            for (int r = 0; r < d; r++) {
                double dr = diff[r];
                if (dr == 0.0) continue;  /* Skip zero rows */
                double *row_ptr = my_SW + (r * d);
                for (int col = 0; col < d; col++) {
                    row_ptr[col] += dr * diff[col];
                }
            }
        }
    }
    
    double elapsed = get_time_seconds() - t0;
    
    /* ── Serial reduction of private matrices → SW ───────────────────────── */
    for (int t = 0; t < actual_threads; t++) {
        for (long long k = 0; k < dd; k++) {
            SW[k] += SW_private[t][k];
        }
        free(SW_private[t]);
        free(diff_buffers[t]);
    }
    free(SW_private);
    free(diff_buffers);
    
    return elapsed;
}

/* ─── between-class scatter S_B (serial — C terms only) ──────────────────── */

void compute_sb(
        const double *means, const double *global_mean,
        const int *counts,
        int d, int C,
        double *SB
) {
    double *diff = (double *)malloc(d * sizeof(double));
    if (!diff) { fprintf(stderr, "compute_sb: OOM\n"); exit(1); }

    for (int c = 0; c < C; c++) {
        const double *mu_c = means + (c * d);
        for (int j = 0; j < d; j++)
            diff[j] = mu_c[j] - global_mean[j];
        for (int r = 0; r < d; r++) {
            double dr = diff[r] * counts[c];
            if (dr == 0.0) continue;
            double *row_ptr = SB + (r * d);
            for (int col = 0; col < d; col++)
                row_ptr[col] += dr * diff[col];
        }
    }
    free(diff);
}

/* ─── projection ──────────────────────────────────────────────────────────── */

void lda_project(
        const double *X, const double *W,
        int n, int d, int k,
        double *Y
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *x_row = X + (i * d);
        double *y_row = Y + (i * k);
        for (int kk = 0; kk < k; kk++) {
            double s = 0.0;
            for (int j = 0; j < d; j++) {
                s += x_row[j] * W[j * k + kk];
            }
            y_row[kk] = s;
        }
    }
}
