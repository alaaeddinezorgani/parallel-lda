#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ─── class means ───────────────────────────────────────── */

void compute_class_means(
        const double *X, const int *labels,
        int n, int d, int C,
        double *means,
        int    *counts,
        double *global_mean
) {
    memset(means, 0, C * d * sizeof(double));
    memset(counts, 0, C * sizeof(int));
    memset(global_mean, 0, d * sizeof(double));

    for (int i = 0; i < n; i++) {
        int c = labels[i];
        counts[c]++;
        for (int j = 0; j < d; j++)
            means[c * d + j] += X[i * d + j];
    }

    for (int c = 0; c < C; c++) {
        for (int j = 0; j < d; j++) {
            means[c * d + j] /= counts[c];
            global_mean[j] += means[c * d + j] * counts[c];
        }
    }

    for (int j = 0; j < d; j++)
        global_mean[j] /= n;
}

/* ─── S_W parallel ─────────────────────────────────────── */

double compute_sw_parallel(
        const double *X, const int *labels,
        const double *means,
        int n, int d, int C,
        double *SW,
        int num_threads
) {
    long long dd = (long long)d * d;

    double **SW_private = malloc(num_threads * sizeof(double *));
    double **diff_buffers = malloc(num_threads * sizeof(double *));

    for (int t = 0; t < num_threads; t++) {
        SW_private[t] = calloc(dd, sizeof(double));
        diff_buffers[t] = malloc(d * sizeof(double));
    }

    double t0 = get_time_seconds();

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        double *my_SW = SW_private[tid];
        double *diff = diff_buffers[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int c = labels[i];
            const double *x = X + i * d;
            const double *mu = means + c * d;

            for (int j = 0; j < d; j++)
                diff[j] = x[j] - mu[j];

            for (int r = 0; r < d; r++) {
                double dr = diff[r];
                double *row = my_SW + r * d;
                for (int col = 0; col < d; col++)
                    row[col] += dr * diff[col];
            }
        }
    }

    double elapsed = get_time_seconds() - t0;

    for (int t = 0; t < num_threads; t++) {
        for (long long i = 0; i < dd; i++)
            SW[i] += SW_private[t][i];
        free(SW_private[t]);
        free(diff_buffers[t]);
    }

    free(SW_private);
    free(diff_buffers);

    return elapsed;
}

/* ─── S_B parallel (NEW) ───────────────────────────────── */

void compute_sb(
        const double *means, const double *global_mean,
        const int *counts,
        int d, int C,
        double *SB
) {
    #pragma omp parallel
    {
        double *local_SB = calloc(d * d, sizeof(double));
        double *diff = malloc(d * sizeof(double));

        #pragma omp for
        for (int c = 0; c < C; c++) {
            const double *mu = means + c * d;

            for (int j = 0; j < d; j++)
                diff[j] = mu[j] - global_mean[j];

            for (int r = 0; r < d; r++) {
                double dr = diff[r] * counts[c];
                double *row = local_SB + r * d;
                for (int col = 0; col < d; col++)
                    row[col] += dr * diff[col];
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < d * d; i++)
                SB[i] += local_SB[i];
        }

        free(local_SB);
        free(diff);
    }
}

/* ─── projection ───────────────────────────────────────── */

void lda_project(
        const double *X, const double *W,
        int n, int d, int k,
        double *Y
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double *x = X + i * d;
        double *y = Y + i * k;

        for (int kk = 0; kk < k; kk++) {
            double s = 0.0;
            for (int j = 0; j < d; j++)
                s += x[j] * W[j * k + kk];
            y[kk] = s;
        }
    }
}
