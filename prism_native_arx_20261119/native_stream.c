#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include "model_constants.h"
#ifdef __AVX512F__
#include <immintrin.h>
#endif

enum { WIDTH = 5, MAX_WIDTH = 10, Q_RING = 64, A_PAD=(A_DIM+7)&~7 };

typedef struct {
    double x[WINDOW][MAX_WIDTH], y[WINDOW];
    double sums[BIN_COUNT][MAX_WIDTH];
    double q[Q_RING], pending[DELAY][A_PAD];
    double theta[A_DIM], covariance[A_DIM][A_PAD];
    unsigned position, issued, updates, failed;
} PrismState;

typedef struct {
    double x[2*WINDOW*WIDTH], y[2*WINDOW];
    unsigned position;
} LinearState;

static unsigned index_at(unsigned position, unsigned lag) {
    return (position + lag) & (WINDOW - 1);
}

static void spline(const double *input, double *out) {
    for (int i=0; i<WIDTH; ++i) {
        int segment = input[i] >= SPLINE_KNOTS[i][1];
        double delta = input[i] - SPLINE_KNOTS[i][segment];
        for (int j=0; j<2; ++j)
            out[2*i+j] = SPLINE_VALUES[i][segment][j] + SPLINE_SLOPES[i][segment][j] * delta;
    }
}

/* History is oldest to newest and excludes the first input at issue time. */
void init_prism(PrismState *state, const double *x, const double *y, const double *q) {
    memset(state, 0, sizeof(*state));
    memcpy(state->theta, A_THETA, sizeof(state->theta));
    for (int i=0; i<A_DIM; ++i) {
        state->covariance[i][i]=A_COVARIANCE[i][i];
        for (int j=i+1; j<A_DIM; ++j)
            state->covariance[i][j]=state->covariance[j][i]=0.5*(A_COVARIANCE[i][j]+A_COVARIANCE[j][i]);
    }
    memcpy(state->q, q, sizeof(state->q));
    for (unsigned lag=0; lag<WINDOW-1; ++lag) {
        const double *row = x + (WINDOW-2-lag)*WIDTH;
        state->y[lag] = y[WINDOW-2-lag];
        if (SPLINE_BEFORE_BINS) spline(row, state->x[lag]);
        else memcpy(state->x[lag], row, WIDTH*sizeof(double));
    }
    for (unsigned bin=0; bin<BIN_COUNT; ++bin)
        for (unsigned lag=BIN_LO[bin]; lag<BIN_HI[bin]; ++lag)
            for (int c=0; c<HISTORY_WIDTH; ++c)
                state->sums[bin][c] += state->x[lag][c];
}

static double upper(PrismState *state, const double *x, double y) {
    double transformed[MAX_WIDTH];
    if (SPLINE_BEFORE_BINS) spline(x, transformed);
    else memcpy(transformed, x, WIDTH*sizeof(double));
    unsigned position = index_at(state->position, WINDOW-1);
    for (unsigned bin=0; bin<BIN_COUNT; ++bin) {
        unsigned lo = BIN_LO[bin], hi = BIN_HI[bin];
        const double *incoming = lo ? state->x[index_at(position, lo)] : transformed;
        const double *outgoing = state->x[index_at(position, hi)];
        for (int c=0; c<HISTORY_WIDTH; ++c)
            state->sums[bin][c] += incoming[c] - outgoing[c];
    }
    memcpy(state->x[position], transformed, HISTORY_WIDTH*sizeof(double));
    state->y[position] = y;
    state->position = position;

    /* Recompute periodically to bound floating-point drift in the rolling sums. */
    if ((state->issued & 255u) == 0) {
        memset(state->sums, 0, sizeof(state->sums));
        for (unsigned bin=0; bin<BIN_COUNT; ++bin)
            for (unsigned lag=BIN_LO[bin]; lag<BIN_HI[bin]; ++lag)
                for (int c=0; c<HISTORY_WIDTH; ++c)
                    state->sums[bin][c] += state->x[index_at(position,lag)][c];
    }

    double score = y*KCW_WEIGHTS[0] + state->y[index_at(position,WINDOW-1)]*KCW_WEIGHTS[1];
    unsigned offset = 2;
    for (unsigned bin=0; bin<BIN_COUNT; ++bin) {
        double mean[MAX_WIDTH], basis[MAX_WIDTH];
        double inverse_count = 1.0 / (BIN_HI[bin]-BIN_LO[bin]);
        for (int c=0; c<HISTORY_WIDTH; ++c) mean[c] = state->sums[bin][c] * inverse_count;
        const double *features = mean;
        if (SPLINE_AFTER_BINS) {
            spline(mean, basis);
            features = basis;
        }
        for (int c=0; c<BIN_WIDTH; ++c) score += features[c] * KCW_WEIGHTS[offset++];
    }
    score += KCW_INTERCEPT;
    double prediction = LINK_ASINH ? LINK_SCALE*sinh(score) : copysign(expm1(fabs(score)), score);
    return prediction + (DELTA_TARGET ? y : 0);
}

static int positive_definite(const double p[A_DIM][A_PAD]) {
    double lower[A_DIM][A_DIM];
    for (int i=0; i<A_DIM; ++i) {
        for (int j=0; j<=i; ++j) {
            double v = p[i][j];
            for (int k=0; k<j; ++k) v -= lower[i][k]*lower[j][k];
            if (i==j) {
                if (!(v>0) || !isfinite(v)) return 0;
                lower[i][j] = sqrt(v);
            } else lower[i][j] = v / lower[j][j];
        }
    }
    return 1;
}

static void update(PrismState *state, const double *phi, double residual) {
    double projected[A_PAD]={0};
    double denominator = A_FORGET, fit = 0, norm2 = 0;
    for (int i=0; i<A_DIM; ++i) {
        double dot = 0;
#ifdef __AVX512F__
        __m512d products=_mm512_setzero_pd();
        for (int j=0; j<A_PAD; j+=8)
            products=_mm512_add_pd(products,_mm512_mul_pd(_mm512_loadu_pd(state->covariance[i]+j),_mm512_loadu_pd(phi+j)));
        dot=_mm512_reduce_add_pd(products);
#else
        #pragma omp simd reduction(+:dot)
        for (int j=0; j<A_DIM; ++j) dot += state->covariance[i][j] * phi[j];
#endif
        projected[i] = dot;
        denominator += phi[i] * dot;
        fit += state->theta[i] * phi[i];
    }
    if (!(denominator>0) || !isfinite(denominator)) { state->failed=1; return; }
    double reciprocal = 1.0/denominator;
    double scaled_innovation = ((residual - A_YM)/A_YS - fit)*reciprocal;
    for (int i=0; i<A_DIM; ++i) {
        state->theta[i] += projected[i] * scaled_innovation;
        norm2 += state->theta[i]*state->theta[i];
    }
    double covariance_norm2=0;
#ifdef __AVX512F__
    __m512d squares=_mm512_setzero_pd();
    const __m512d inv_forget=_mm512_set1_pd(1.0/A_FORGET), inv_denominator=_mm512_set1_pd(reciprocal);
    for (int i=0; i<A_DIM; ++i) {
        const __m512d row_value=_mm512_set1_pd(projected[i]);
        for (int j=0; j<A_PAD; j+=8) {
            __m512d pair=_mm512_mul_pd(row_value,_mm512_loadu_pd(projected+j));
            __m512d p=_mm512_loadu_pd(state->covariance[i]+j);
            __m512d v=_mm512_mul_pd(_mm512_sub_pd(p,_mm512_mul_pd(pair,inv_denominator)),inv_forget);
            _mm512_storeu_pd(state->covariance[i]+j,v);
            squares=_mm512_add_pd(squares,_mm512_mul_pd(v,v));
        }
    }
    covariance_norm2=_mm512_reduce_add_pd(squares);
#else
    for (int i=0; i<A_DIM; ++i) {
        /* Use the symmetric outer product, with contiguous rows for SIMD. */
        #pragma omp simd reduction(+:covariance_norm2)
        for (int j=0; j<A_DIM; ++j) {
            double v=(state->covariance[i][j] - projected[i]*projected[j]*reciprocal)*(1.0/A_FORGET);
            state->covariance[i][j]=v;
            covariance_norm2+=v*v;
        }
    }
#endif
    if (!isfinite(norm2) || norm2>1e12 || !isfinite(covariance_norm2)) state->failed=1;
    if ((++state->updates & 255u)==0 && !positive_definite(state->covariance)) state->failed=1;
}

double prism_step(PrismState *state, const double *x, double y) {
    unsigned t=state->issued, qindex=t & (Q_RING-1);
    double q=upper(state,x,y), residual=y-state->q[(t-DELAY) & (Q_RING-1)];
    double *phi=state->pending[t%DELAY];
    if (t>=DELAY) update(state,phi,residual);
    double features[A_DIM-1];
    features[0]=q;
    if (DRIVER_EX) {
        features[1]=residual;
        for (int c=0; c<WIDTH; ++c) {
            features[2+c]=x[c];
            features[7+c]=(state->sums[1][c]+state->sums[2][c]+state->sums[3][c])/7.0;
            features[12+c]=(state->sums[4][c]+state->sums[5][c])/24.0;
            features[17+c]=(state->sums[6][c]+state->sums[7][c])/96.0;
        }
    } else {
        const unsigned lags[5]={1,4,8,16,32};
        int count=CONDITIONAL ? 5 : 3;
        for (int i=0; i<count; ++i) features[1+i]=q-state->q[(t-lags[i]) & (Q_RING-1)];
        if (CONDITIONAL) {
            features[6]=q*features[4]; features[7]=q*features[5]; features[8]=residual;
            features[9]=residual*features[0]; features[10]=residual*features[2]; features[11]=residual*features[5];
        } else { features[4]=q*features[3]; features[5]=residual; }
    }
    phi[0]=1;
    double correction=state->theta[0];
    for (int j=1; j<A_DIM; ++j) {
        phi[j]=(features[j-1]-A_MEAN[j-1])/A_SCALE[j-1];
        correction+=phi[j]*state->theta[j];
    }
    state->q[qindex]=q;
    state->issued++;
    double prediction=q+A_GAIN*(A_YM+A_YS*correction);
    return state->failed || !isfinite(prediction) ? NAN : prediction;
}

static void init_linear(LinearState *state, const double *x, const double *y, int exogenous) {
    memset(state, 0, sizeof(*state));
    for (unsigned i=0; i<WINDOW-1; ++i) {
        state->y[i]=state->y[i+WINDOW]=y[WINDOW-2-i];
        if (exogenous) {
            memcpy(state->x+WIDTH*i,x+WIDTH*(WINDOW-2-i),WIDTH*sizeof(double));
            memcpy(state->x+WIDTH*(i+WINDOW),state->x+WIDTH*i,WIDTH*sizeof(double));
        }
    }
}

static double dot(const double *u, const double *v, unsigned length) {
    double sum=0;
    #pragma omp simd reduction(+:sum)
    for (unsigned i=0; i<length; ++i) sum+=u[i]*v[i];
    return sum;
}

static double ar_step(LinearState *state, double y) {
    unsigned i=state->position=index_at(state->position, WINDOW-1);
    state->y[i]=state->y[i+WINDOW]=y;
    return AR_INTERCEPT + dot(state->y+i,AR_WEIGHTS,WINDOW) + (AR_DELTA ? y : 0);
}

static double arx_step(LinearState *state, const double *x, double y) {
    unsigned i=state->position=index_at(state->position, WINDOW-1);
    state->y[i]=state->y[i+WINDOW]=y;
    memcpy(state->x+WIDTH*i,x,WIDTH*sizeof(double));
    memcpy(state->x+WIDTH*(i+WINDOW),x,WIDTH*sizeof(double));
    return ARX_INTERCEPT + dot(state->y+i,ARX_WEIGHTS,WINDOW)
         + dot(state->x+WIDTH*i,ARX_WEIGHTS+WINDOW,WIDTH*WINDOW) + (ARX_DELTA ? y : 0);
}

size_t prism_state_bytes(void) { return sizeof(PrismState); }

static uint64_t nanoseconds(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC_RAW,&stamp);
    return (uint64_t)stamp.tv_sec*1000000000u + stamp.tv_nsec;
}

/* One streaming call per sample in each branch. Timing starts after warmup. */
double run(int kind, const double *xp, const double *yp, const double *qp,
           const double *x, const double *y, size_t count, size_t skip, double *output) {
    uint64_t start=0;
    if (kind==0) {
        PrismState state;
        init_prism(&state,xp,yp,qp);
        for (size_t i=0; i<count; ++i) {
            if (i==skip) start=nanoseconds();
            output[i]=prism_step(&state,x+i*WIDTH,y[i]);
        }
    } else {
        LinearState state;
        init_linear(&state,xp,yp,kind==2);
        if (kind==1) {
            for (size_t i=0; i<count; ++i) {
                if (i==skip) start=nanoseconds();
                output[i]=ar_step(&state,y[i]);
            }
        } else {
            for (size_t i=0; i<count; ++i) {
                if (i==skip) start=nanoseconds();
                output[i]=arx_step(&state,x+i*WIDTH,y[i]);
            }
        }
    }
    return (double)(nanoseconds()-start)/(count-skip);
}
