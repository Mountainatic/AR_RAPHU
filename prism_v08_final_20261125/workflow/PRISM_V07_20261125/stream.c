#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

enum { DELAY = 4, PREFIX_RING = 256, RESIDUAL_RING = 128, WIDTH = 4, BINS = 11, A_WIDTH = 16 };

typedef struct {
    size_t column0, column1;
    double mean0, mean1, weight0, weight1;
} Channel;

typedef struct {
    size_t source, difference, bins, width;
    double mean, inv_scale, intercept, upper, reference_knot;
    double center[WIDTH], knot[WIDTH], knot_inv_span[WIDTH];
    double ref_inv_span;
    size_t near[BINS], far[BINS];
    double inv_width[BINS], low[BINS], high[BINS], inv_support[BINS], theta[BINS][WIDTH];
} View;

#include "model.h"

typedef struct {
    double prefix[VIEWS][PREFIX_RING][WIDTH+1];
    double process_delay[CHANNELS][DELAY];
    double anchors[DELAY], issued[DELAY], residuals[RESIDUAL_RING], ema[7];
} State;

static uint64_t ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}

static double cube_pos(double x) {
    return x > 0.0 ? (x*x)*x : 0.0;
}

static void update_process(State *s, const double *x, size_t t) {
    double level[CHANNELS], delta[CHANNELS];
    for (size_t j=0; j<CHANNELS; ++j) {
        const Channel *m=&channels[j];
        double value=(x[m->column0]-m->mean0)*m->weight0;
        value+=(x[m->column1]-m->mean1)*m->weight1;
        level[j]=value;
        delta[j]=t<DELAY ? 0.0 : value-s->process_delay[j][t%DELAY];
        s->process_delay[j][t%DELAY]=value;
    }
    size_t current=(t+1)%PREFIX_RING, previous=t%PREFIX_RING;
    for (size_t j=0; j<VIEWS; ++j) {
        const View *v=&views[j];
        const double raw=v->difference ? delta[v->source] : level[v->source];
        double *out=s->prefix[j][current];
        const double *old=s->prefix[j][previous];
        out[0]=old[0]+raw;
        const double z=(raw-v->mean)*v->inv_scale;
        out[1]=old[1]+(z-v->center[0]);
        if (v->width>1) {
            const double b=z<v->upper ? z : v->upper;
            const double shared=cube_pos(b-v->reference_knot)*v->ref_inv_span;
            const double beyond=z>b ? z-b : 0.0;
            for (size_t k=1; k<v->width; ++k) {
                double col=cube_pos(b-v->knot[k])*v->knot_inv_span[k]-shared;
                col+=beyond*(3.0*(v->reference_knot-v->knot[k]));
                out[k+1]=old[k+1]+(col-v->center[k]);
            }
        }
    }
}

static void view_prediction(const State *s, size_t t, size_t j, double *out, double *trust) {
    const View *v=&views[j];
    double outside=0.0, k=v->intercept;
    for (size_t bin=0; bin<v->bins; ++bin) {
        size_t right=(t+1-v->near[bin])%PREFIX_RING, left=(t+1-v->far[bin])%PREFIX_RING;
        const double *a=s->prefix[j][right], *b=s->prefix[j][left];
        const double raw=(a[0]-b[0])*v->inv_width[bin];
        const double low=v->low[bin]-raw, high=raw-v->high[bin];
        const double deviation=(low>high ? low : high)*v->inv_support[bin];
        if (deviation>outside) outside=deviation;
        for (size_t m=0; m<v->width; ++m) {
            const double means=(a[m+1]-b[m+1])*v->inv_width[bin];
            k+=means*v->theta[bin][m];
        }
    }
    const double u=outside<1.0 ? outside : 1.0;
    *out=k;
    *trust=(1.0-u)*(1.0-u)*(1.0+2.0*u);
}

static double apply_w(double coordinate) {
    double z=(coordinate-w_mean)*w_inv_scale;
    if (z<w_edges[0]) z=w_edges[0];
    if (z>w_edges[W_SEGMENTS]) z=w_edges[W_SEGMENTS];
    size_t i=0;
    while (i+1<W_SEGMENTS && z>=w_edges[i+1]) ++i;
    const double dx=(z-w_edges[i])*w_inv_span[i];
    const double *p=w_poly[i];
    return (((p[0]*dx+p[1])*dx+p[2])*dx+p[3])*dx+p[4];
}

static double upstream(const State *s, size_t t) {
    double p[VIEWS], w[VIEWS];
    for (size_t j=0; j<VIEWS; ++j) view_prediction(s,t,j,&p[j],&w[j]);
    double out=K_ACTIVE ? p[REFERENCE]*w[REFERENCE] : 0.0;
    if (C_ACTIVE) {
        double max_trust=0.0, weighted_trust=0.0, shift=0.0;
        for (size_t j=0; j<VIEWS; ++j) {
            if (w[j]>max_trust) max_trust=w[j];
            shift+=(p[j]-c_mean[j])*c_inv_scale[j]*w[j]*c_coef[j];
            weighted_trust+=w[j]*c_abs_coef[j];
        }
        out+=shift+c_intercept*max_trust;
        if (W_ACTIVE) out+=(c_sum_abs>0.0 ? weighted_trust/c_sum_abs : max_trust)*apply_w(out);
    } else if (W_ACTIVE) {
        out+=w[REFERENCE]*apply_w(p[REFERENCE]);
    }
    return out;
}

static double residual_correction(State *s, size_t t, size_t count, double maturity, double delta) {
    if (!A_ACTIVE) return 0.0;
    const size_t slot=t%DELAY;
    double correction=0.0;
    if (count>=DELAY) {
        const double r=(maturity-s->anchors[slot])-s->issued[slot];
        s->residuals[t%RESIDUAL_RING]=r;
        correction=a_bias;
        for (size_t j=0; j<A_RECENT; ++j) {
            const double feature=s->residuals[(t-a_offsets[j])%RESIDUAL_RING];
            correction+=a_weight[j]*feature;
        }
        const double older=s->residuals[(t-A_EMA_SHIFT)%RESIDUAL_RING];
        for (size_t j=0; j<A_EMA; ++j) {
            s->ema[j]+=a_alpha[j]*(older-s->ema[j]);
            correction+=a_weight[A_RECENT+j]*s->ema[j];
        }
    }
    s->anchors[slot]=maturity;
    s->issued[slot]=delta;
    return correction;
}

double replay(const double *data,size_t start,size_t n,double *out) {
    State state;
    memset(&state,0,sizeof(state));
    for (size_t t=0; t<start; ++t) update_process(&state,data+t*STRIDE,t);
    const uint64_t begin=ns();
    for (size_t i=0; i<n; ++i) {
        const size_t t=start+i;
        update_process(&state,data+t*STRIDE,t);
        const double matured=data[(t-DELAY)*STRIDE];
        const double q=upstream(&state,t);
        const double a=residual_correction(&state,t,i,matured,q);
        out[i]=matured+(q+a);
    }
    return (double)(ns()-begin)/(double)n;
}
