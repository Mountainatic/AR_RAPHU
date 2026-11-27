#include <immintrin.h>
#include <math.h>
#include <stddef.h>

enum { BASIS = 4, BINS = 11, TERMS = 2 + BASIS, EMA = 7, LAGS = 16,
       FEEDBACK_RING = 64, REANCHOR = 65536 };

typedef struct {
    size_t column0, column1;
    double mean0, mean1, weight0, weight1;
} Channel;

typedef struct {
    size_t source, difference, bins, width;
    double mean, inv_scale, intercept, upper, reference_knot;
    double center[BASIS], knot[BASIS], knot_inv_span[BASIS], ref_inv_span;
    size_t near[BINS], far[BINS];
    double inv_width[BINS], lo[BINS], hi[BINS], inv_span[BINS], theta[BINS][BASIS];
    double sigma_max[BINS], inv_sigma_scale[BINS];
} View;

typedef struct {
    size_t recent, ema, shift;
    size_t offset[LAGS];
    double alpha[EMA], weight[LAGS], bias;
} Head;

#include "model.h"

_Static_assert((PREFIX_RING & (PREFIX_RING-1)) == 0 && PREFIX_RING > HISTORY,
               "prefix ring must be a power of two and cover all endpoints");
_Static_assert(REANCHOR > HISTORY && ANCHOR_DELAY <= DELAY,
               "invalid window or prediction clock");

typedef struct {
    double errors[FEEDBACK_RING], ema[FEEDBACK_RING][EMA];
} Feedback;

typedef struct {
    double prefix[2][VIEWS][PREFIX_RING][TERMS];
    unsigned main_ring;
    double history[CHANNELS][DELAY];
    double anchors[DELAY], issued_q[DELAY];
    double rolling[TRUST_WINDOW], trust_sum;
    Feedback assisted, fallback;
} State;

static double cube(double z) {
    z = fmax(z, 0.0);
    return z*z*z;
}

static void point(const View *v, double raw, double *out) {
    double z = (raw-v->mean)*v->inv_scale;
    out[0] = z;
    out[1] = z*z;
    out[2] = z-v->center[0];
    for (size_t m = 3; m < TERMS; ++m)
        out[m] = 0.0;
    if (v->width == 4) {
        double clipped = fmin(z, v->upper);
        double common = cube(clipped-v->reference_knot)*v->ref_inv_span;
        __m256d knots = _mm256_loadu_pd(v->knot);
        __m256d x = _mm256_max_pd(_mm256_setzero_pd(), _mm256_sub_pd(_mm256_set1_pd(clipped), knots));
        __m256d cubic = _mm256_mul_pd(_mm256_mul_pd(x, x), x);
        __m256d slopes = _mm256_mul_pd(_mm256_set1_pd(3.0),
                                      _mm256_sub_pd(_mm256_set1_pd(v->reference_knot), knots));
        __m256d value = _mm256_fmadd_pd(cubic, _mm256_loadu_pd(v->knot_inv_span), _mm256_set1_pd(-common));
        value = _mm256_fmadd_pd(_mm256_set1_pd(z-clipped), slopes, value);
        value = _mm256_blend_pd(value, _mm256_set1_pd(z), 1);
        _mm256_storeu_pd(out+2, _mm256_sub_pd(value, _mm256_loadu_pd(v->center)));
    } else if (v->width > 1) {
        double clipped = fmin(z, v->upper);
        double common = cube(clipped-v->reference_knot)*v->ref_inv_span;
        for (size_t m = 1; m < v->width; ++m)
            out[2+m] = cube(clipped-v->knot[m])*v->knot_inv_span[m]-common
                       + 3.0*(z-clipped)*(v->reference_knot-v->knot[m])-v->center[m];
    }
}

static void advance(double *next, const double *prior, const double *value) {
    next[0] = prior[0]+value[0];
    next[1] = prior[1]+value[1];
    _mm256_storeu_pd(next+2, _mm256_add_pd(_mm256_loadu_pd(prior+2), _mm256_loadu_pd(value+2)));
}

static void ingest(State *s, size_t t, const double *row) {
    size_t phase = t % REANCHOR;
    if (phase == 0 && t != 0)
        s->main_ring ^= 1u;
    unsigned live = s->main_ring, shadow = live ^ 1u;
    size_t prev = t & (PREFIX_RING-1), next = (t+1) & (PREFIX_RING-1);
    double level[CHANNELS], difference[CHANNELS];
    size_t slot = t % DELAY;
    for (size_t j = 0; j < CHANNELS; ++j) {
        const Channel *c = &channels[j];
        double value = (row[c->column0]-c->mean0)*c->weight0+(row[c->column1]-c->mean1)*c->weight1;
        level[j] = value;
        difference[j] = t < DELAY ? 0.0 : value-s->history[j][slot];
        s->history[j][slot] = value;
    }
    for (size_t j = 0; j < VIEWS; ++j) {
        if (c_mass[j] == 0.0)
            continue;
        const View *v = &views[j];
        double values[TERMS];
        point(v, v->difference ? difference[v->source] : level[v->source], values);
        advance(s->prefix[live][j][next], s->prefix[live][j][prev], values);
        if (phase >= REANCHOR-HISTORY) {
            if (phase == REANCHOR-HISTORY) {
                for (size_t k = 0; k < TERMS; ++k)
                    s->prefix[shadow][j][prev][k] = 0.0;
            }
            advance(s->prefix[shadow][j][next], s->prefix[shadow][j][prev], values);
        }
    }
}

static double read_view(const State *s, size_t j, size_t head, double *q) {
    const View *v = &views[j];
    const double (*prefix)[TERMS] = s->prefix[s->main_ring][j];
    double outside = 0.0;
    __m256d accum = _mm256_setzero_pd();
    for (size_t b = 0; b < v->bins; ++b) {
        const double *right = prefix[(head-v->near[b]) & (PREFIX_RING-1)];
        const double *left = prefix[(head-v->far[b]) & (PREFIX_RING-1)];
        double mu = (right[0]-left[0])*v->inv_width[b];
        outside = fmax(outside, fmax(v->lo[b]-mu, mu-v->hi[b])*v->inv_span[b]);
        if (USE_SIGMA && v->far[b]-v->near[b] > 1) {
            double second = (right[1]-left[1])*v->inv_width[b];
            double sigma = sqrt(fmax(0.0, second-mu*mu));
            outside = fmax(outside, (sigma-v->sigma_max[b])*v->inv_sigma_scale[b]);
        }
        if (outside >= 1.0)
            return 0.0;
        __m256d diff = _mm256_sub_pd(_mm256_loadu_pd(right+2), _mm256_loadu_pd(left+2));
        accum = _mm256_fmadd_pd(diff, _mm256_loadu_pd(v->theta[b]), accum);
    }
    double sums[BASIS];
    _mm256_storeu_pd(sums, accum);
    *q = v->intercept+sums[0]+sums[1]+sums[2]+sums[3];
    double u = 1.0-outside;
    return u*u*(1.0+2.0*outside);
}

static double correct(double fc) {
    double z = fmin(w_edges[W_SEGMENTS], fmax(w_edges[0], (fc-w_mean)*w_inv_scale));
    size_t i = 0;
    while (i+1 < W_SEGMENTS && z >= w_edges[i+1])
        ++i;
    double t = (z-w_edges[i])*w_inv_span[i];
    const double *c = w_poly[i];
    return (((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4];
}

static double exogenous(const State *s, size_t t, double *rho) {
    double fc = 0.0;
    *rho = 0.0;
    for (size_t j = 0; j < VIEWS; ++j) {
        if (c_mass[j] == 0.0)
            continue;
        double q = 0.0, trust = read_view(s, j, t+1, &q);
        fc += trust*((q-c_mean[j])*c_scaled_w[j]+c_bias[j]);
        *rho += trust*c_mass[j];
    }
    *rho = fmin(1.0, fmax(0.0, *rho));
    return fc + (W_ACTIVE ? (*rho)*correct(fc) : 0.0);
}

static double feedback(Feedback *f, const Head *head, size_t n, double residual) {
    size_t slot = n & (FEEDBACK_RING-1), prev = (n-1) & (FEEDBACK_RING-1);
    f->errors[slot] = residual;
    double delta = head->bias;
    for (size_t k = 0; k < head->recent; ++k) {
        size_t offset = head->offset[k];
        if (n >= offset)
            delta += head->weight[k]*f->errors[(n-offset) & (FEEDBACK_RING-1)];
    }
    for (size_t k = 0; k < head->ema; ++k) {
        f->ema[slot][k] = head->alpha[k]*residual + (1.0-head->alpha[k])*f->ema[prev][k];
        if (n >= head->shift)
            delta += head->weight[head->recent+k]*f->ema[(n-head->shift) & (FEEDBACK_RING-1)][k];
    }
    return delta;
}

void replay_v2(const double *data, size_t first, size_t count, double *out) {
    State s = {0};
    for (size_t t = 0; t < first + count; ++t) {
        ingest(&s, t, data + t*STRIDE);
        if (t < first)
            continue;
        size_t n = t-first, slot = t % DELAY;
        double anchor = data[(t-ANCHOR_DELAY)*STRIDE];
        double a = 0.0, a0 = 0.0;
        if (n >= DELAY) {
            double z = anchor-s.anchors[slot];
            if (A_ACTIVE)
                a = feedback(&s.assisted, &assisted_head, n-DELAY, z-s.issued_q[slot]);
            if (A0_ACTIVE)
                a0 = feedback(&s.fallback, &fallback_head, n-DELAY, z);
        }
        double rho, q = exogenous(&s, t, &rho);
        s.anchors[slot] = anchor;
        s.issued_q[slot] = q;
        s.trust_sum += rho-s.rolling[n % TRUST_WINDOW];
        s.rolling[n % TRUST_WINDOW] = rho;
        double weight = fmin(1.0, fmax(0.0, s.trust_sum/fmin((double)(n+1), TRUST_WINDOW)));
        out[n] = anchor+q+weight*a+(1.0-weight)*a0;
    }
}

