#define _POSIX_C_SOURCE 199309

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
        const double *x = X + i * d;
        double *mu = means + c * d;
        for (int j = 0; j < d; j++)
            mu[j] += x[j];
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

/* ─── S_W ───────────────────────────────────────────────── */

double compute_sw(
        const double *X, const int *labels,
        const double *means,
        int n, int d, int C,
        double *SW
) {
    double *diff = malloc(d * sizeof(double));
    if (!diff) exit(1);

    double t0 = get_time_seconds();

    for (int i = 0; i < n; i++) {
        int c = labels[i];
        const double *x = X + i * d;
        const double *mu = means + c * d;

        for (int j = 0; j < d; j++)
            diff[j] = x[j] - mu[j];

        for (int r = 0; r < d; r++) {
            double dr = diff[r];
            double *row = SW + r * d;
            for (int col = 0; col < d; col++)
                row[col] += dr * diff[col];
        }
    }

    double elapsed = get_time_seconds() - t0;
    free(diff);
    return elapsed;
}

/* ─── S_B ───────────────────────────────────────────────── */

void compute_sb(
        const double *means, const double *global_mean,
        const int *counts,
        int d, int C,
        double *SB
) {
    double *diff = malloc(d * sizeof(double));
    if (!diff) exit(1);

    for (int c = 0; c < C; c++) {
        const double *mu = means + c * d;

        for (int j = 0; j < d; j++)
            diff[j] = mu[j] - global_mean[j];

        for (int r = 0; r < d; r++) {
            double dr = diff[r] * counts[c];
            double *row = SB + r * d;
            for (int col = 0; col < d; col++)
                row[col] += dr * diff[col];
        }
    }

    free(diff);
}

/* ─── projection ───────────────────────────────────────── */

void lda_project(
        const double *X, const double *W,
        int n, int d, int k,
        double *Y
) {
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
