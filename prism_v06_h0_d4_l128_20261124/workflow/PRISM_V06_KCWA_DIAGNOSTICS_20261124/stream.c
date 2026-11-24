#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "model.h"

static uint64_t now_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (uint64_t)value.tv_sec * 1000000000 + value.tv_nsec;
}

static double cube_positive(double x) {
    return x > 0.0 ? x * x * x : 0.0;
}

static double w_adjustment(double q) {
    double z = (q - w_mean) * w_inv_scale;
    if (z < w_edges[0]) z = w_edges[0];
    if (z > w_edges[W_SEGMENTS]) z = w_edges[W_SEGMENTS];
    size_t piece = 0;
    while (piece + 1 < W_SEGMENTS && z >= w_edges[piece + 1]) ++piece;
    double dx = z - w_edges[piece];
    const double *p = w_poly[piece];
    return (((p[0] * dx + p[1]) * dx + p[2]) * dx + p[3]) * dx + p[4];
}

static void extend_prefix(double prefix[L + 1][X_WIDTH], const double *row, size_t t) {
    size_t slot = (t + 1) % (L + 1);
    size_t previous = t % (L + 1);
    for (size_t j = 0; j < X_WIDTH; ++j)
        prefix[slot][j] = prefix[previous][j] + row[j+1];
}

static double bin_mean(const double prefix[L + 1][X_WIDTH], size_t t, size_t channel,
                       size_t near, size_t far) {
    size_t right = (t + 1 - near) % (L + 1);
    size_t left = (t + 1 - far) % (L + 1);
    return (prefix[right][channel] - prefix[left][channel]) / (double)(far - near);
}

static double representation(const double prefix[L + 1][X_WIDTH], size_t t, size_t channel) {
    double prediction = k_intercept[channel];
    for (size_t bin = 0; bin < k_bins[channel]; ++bin) {
        size_t near = k_near[channel][bin], far = k_far[channel][bin];
        double value = bin_mean(prefix, t, k_input[channel], near, far);
        if (k_joint[channel]) {
            double second = bin_mean(prefix, t, 2, near, far);
            value = (value - joint_mean[0]) * joint_weight[0] +
                    (second - joint_mean[1]) * joint_weight[1];
        }
        double z = (value - k_mean[channel]) * k_inv_scale[channel];
        prediction += (z - k_feature_mean[channel][0]) * k_theta[channel][bin][0];
        if (k_width[channel] > 1) {
            double tail = cube_positive(z - k_knots[channel][k_knot_count[channel] - 1]);
            for (size_t j = 1; j < k_width[channel]; ++j) {
                double phi = cube_positive(z - k_knots[channel][j - 1]) - tail;
                prediction += (phi - k_feature_mean[channel][j]) * k_theta[channel][bin][j];
            }
        }
    }
    return prediction;
}

static double time_kcwa(const double *raw, size_t start, size_t length, double *out) {
    double prefix[L + 1][X_WIDTH] = {{0.0}};
    double issued[DELAY] = {0.0};
    double residual[A_RING] = {0.0};
    for (size_t t = 0; t < start; ++t) extend_prefix(prefix, raw + t * STRIDE, t);
    uint64_t begin = now_ns();
    for (size_t i = 0; i < length; ++i) {
        size_t t = start + i;
        extend_prefix(prefix, raw + t * STRIDE, t);
        double k[K_CHANNELS] = {0.0};
        for (size_t channel = 0; channel < K_CHANNELS; ++channel) {
            if (C_ACTIVE || (channel == K_REFERENCE && (K_ACTIVE || W_ACTIVE)))
                k[channel] = representation(prefix, t, channel);
        }
        double q = K_ACTIVE ? k[K_REFERENCE] : 0.0;
        if (C_ACTIVE) {
            q += c_intercept;
            for (size_t j = 0; j < K_CHANNELS; ++j)
                q += (k[j] - c_mean[j]) * c_weight[j];
        }
        if (W_ACTIVE) q += w_adjustment(C_ACTIVE ? q : k[K_REFERENCE]);
        double anchor = raw[(t - DELAY) * STRIDE];
        double adjustment = 0.0;
        if (A_COUNT > 0) {
            size_t slot = i % DELAY;
            if (i >= DELAY) {
                residual[i % A_RING] = anchor - raw[(t - 2 * DELAY) * STRIDE] - issued[slot];
            }
            issued[slot] = q;
            if (i >= A_READY) {
                adjustment = a_intercept;
                for (size_t j = 0; j < a_count; ++j)
                    adjustment += (residual[(i - a_offset[j]) % A_RING] - a_mean[j]) * a_weight[j];
            }
        }
        out[i] = (q + adjustment) + anchor;
    }
    return (double)(now_ns() - begin) / (double)length;
}

static double time_linear(int mode, const double *raw, size_t start, size_t length, double *out) {
    uint64_t begin = now_ns();
    for (size_t i = 0; i < length; ++i) {
        size_t t = start + i;
        double anchor = raw[(t - DELAY) * STRIDE];
        double value = linear_intercept[mode];
        for (size_t j = 0; j < L; ++j)
            value += (raw[(t - DELAY - j) * STRIDE] - linear_mean[mode][j]) * linear_weight[mode][j];
        if (mode == 1) {
            for (size_t j = 0; j < L; ++j) {
                for (size_t c = 0; c < X_WIDTH; ++c) {
                    size_t f = L + j * X_WIDTH + c;
                    value += (raw[(t - j) * STRIDE + c + 1] - linear_mean[mode][f]) * linear_weight[mode][f];
                }
            }
        }
        out[i] = value + (linear_delta[mode] ? anchor : 0.0);
    }
    return (double)(now_ns() - begin) / (double)length;
}

double stream(int model, const double *raw, size_t start, size_t length, double *out) {
    return model == 0 ? time_kcwa(raw, start, length, out) : time_linear(model - 1, raw, start, length, out);
}
