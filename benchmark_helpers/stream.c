/* FP64 stream readouts. The exporter provides the sealed constants in model.h. */
#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "model.h"
#include "stream.h"

static int next(size_t slot, int size) {
    return slot + 1 == (size_t)size ? 0 : (int)slot + 1;
}

static int back(int slot, int size) {
    return slot ? slot - 1 : size - 1;
}

static double horizontal(__m256d v) {
    double a[4];
    _mm256_storeu_pd(a, v);
    return (a[0] + a[1]) + (a[2] + a[3]);
}

static double dot(const double *x, const double *y, int n) {
    __m256d sum = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= n; i += 4)
        sum = _mm256_fmadd_pd(_mm256_loadu_pd(x+i), _mm256_loadu_pd(y+i), sum);
    double v = horizontal(sum);
    for (; i < n; ++i)
        v += x[i] * y[i];
    return v;
}

#ifdef PRISM_MODEL

#if ACTIVE
typedef struct {
    int raw;
    int diff;
    int pieces;
    double mean, scale, limit;
    double weight, center;
    double knots[5];
    double cubic[4][4][2];
    double mu_low[POLES], mu_high[POLES], sigma_high[POLES], taper[POLES];
    double theta[XTAPS][4];
} View;
#endif
#include "parameters.h"

typedef struct {
    size_t tick;
    int at_y, at_delta;
    double y[SPAN], delta[YTAPS];
#if ACTIVE
    int at_x, at_raw;
    double raw[RAW_HISTORY][CHANNELS];
    double near[ACTIVE][XNEAR][4];
    double tail[ACTIVE][XTAPS-XNEAR][4];
    double mu[ACTIVE][POLES], variance[ACTIVE][POLES];
#endif
} State;

#if ACTIVE
static double hermite(double distance) {
    if (distance <= 0.) return 1.;
    if (distance >= 1.) return 0.;
    double z = 1. - distance;
    return (z*z)*(1. + 2.*distance);
}

static __m256d point(const View *p, double z) {
    double x = fabs(z);
    if (x > p->limit) {
        double excess = x-p->limit;
        z = copysign(p->limit + excess/(1.+excess), z);
    }
    if (!p->pieces)
        return _mm256_setr_pd(z,0.,0.,0.);
    double bound = fmin(p->knots[p->pieces], fmax(p->knots[0],z));
    int k = 0;
    while (k+1 < p->pieces && bound >= p->knots[k+1]) ++k;
    const double (*c)[2] = p->cubic[k];
    double w = bound-p->knots[k], q[2];
    for (int j=0; j<2; ++j) {
        double value = ((c[0][j]*w+c[1][j])*w+c[2][j])*w+c[3][j];
        double slope = (3.*c[0][j]*w+2.*c[1][j])*w+c[2][j];
        q[j] = value + (z-bound)*slope;
    }
    return _mm256_setr_pd(z,q[0],q[1],0.);
}

static double integrate(State *s, int g, const View *p, __m256d u) {
    int ix=s->at_x;
    __m256d old=_mm256_loadu_pd(s->near[g][ix]);
    _mm256_storeu_pd(s->near[g][ix],u);
    __m256d sum=_mm256_setzero_pd();
    for (int k=0; k<XNEAR; ++k) {
        sum=_mm256_fmadd_pd(_mm256_loadu_pd(s->near[g][ix]),_mm256_loadu_pd(p->theta[k]),sum);
        ix=next((size_t)ix,XNEAR);
    }
    int base=0;
    for (int b=0; b<BANKS; ++b) {
        const int n=bank_order[b];
        for (int i=n-1; i>=0; --i) {
            double *slot=s->tail[g][base+i];
            __m256d v=_mm256_mul_pd(_mm256_set1_pd(bank_b[b][i]),old);
            for (int j=0; j<=i; ++j)
                v=_mm256_fmadd_pd(_mm256_set1_pd(bank_a[b][i][j]),
                                  _mm256_loadu_pd(s->tail[g][base+j]),v);
            _mm256_storeu_pd(slot,v);
            sum=_mm256_fmadd_pd(v,_mm256_loadu_pd(p->theta[XNEAR+base+i]),sum);
        }
        base+=n;
    }
    return horizontal(sum);
}
#endif

static void step(State *s, const double *row, double *out) {
    double anchor=row[0], e=0., level=0.;
    if (s->tick >= ANCHOR_DELAY)
        level=anchor-anchor_center;
    if (s->tick >= SPAN+ANCHOR_DELAY)
        e=anchor-s->y[s->at_y];
    s->y[s->at_y]=anchor;
    s->at_y=next((size_t)s->at_y,SPAN);
    int ix=s->at_delta;
    s->delta[ix]=e;
    double endo=safe_bias, adj=joint_bias;
    for (int k=0; k<YTAPS; ++k) {
        endo+=safe_beta[k]*s->delta[ix];
        adj+=joint_beta[k]*s->delta[ix];
        ix=next((size_t)ix,YTAPS);
    }
    endo=(endo+safe_beta[YTAPS]*level)+anchor;
    adj+=joint_beta[YTAPS]*level;
    s->at_delta=back(s->at_delta,YTAPS);
    double rho=0., exo=0.;
#if ACTIVE
    const double *x=row+1;
    if (DUAL && !s->tick)
        for (int i=0; i<RAW_HISTORY; ++i) memcpy(s->raw[i],x,CHANNELS*sizeof(double));
    for (int g=0; g<ACTIVE; ++g) {
        const View *v=&views[g];
        double raw=x[v->raw]-(v->diff ? s->raw[s->at_raw][v->raw] : 0.);
        double z=fmin(1e6,fmax(-1e6,(raw-v->mean)/v->scale));
        double trust=1.;
        for (int b=0; b<POLES; ++b) {
            double a=moment_poles[b], d=z-s->mu[g][b];
            double var=a*(s->variance[g][b]+(1.-a)*(d*d));
            double mu=s->mu[g][b]+(1.-a)*d;
            s->variance[g][b]=var;
            s->mu[g][b]=mu;
            trust*=hermite((v->mu_low[b]-mu)/v->taper[b])
                  *hermite((mu-v->mu_high[b])/v->taper[b])
                  *hermite((sqrt(var)-v->sigma_high[b])/v->taper[b]);
        }
        double value=integrate(s,g,v,point(v,z));
        rho+=trust*v->weight;
        exo+=trust*(value-v->center);
    }
    if (DUAL) {
        memcpy(s->raw[s->at_raw],x,CHANNELS*sizeof(double));
        s->at_raw=next((size_t)s->at_raw,RAW_HISTORY);
    }
    s->at_x=back(s->at_x,XNEAR);
#endif
    ++s->tick;
    rho=fmin(1.,fmax(0.,rho));
    out[0]=endo+rho*adj+exo;
    out[1]=endo;
    out[2]=rho;
}

#else

typedef struct {
    int at_y, at_x;
    double y[YHIST];
    double x[XHIST][CHANNELS];
} State;

static void step(State *s, const double *row, double *out) {
    int yi=s->at_y, xi=s->at_x;
    s->y[yi]=row[0];
    memcpy(s->x[xi],row+1,CHANNELS*sizeof(double));
    double v=baseline_bias;
    for (int k=0; k<YHIST; ++k) {
        v+=s->y[yi]*baseline_y[k];
        yi=next((size_t)yi,YHIST);
    }
    for (int k=0; k<XHIST; ++k) {
        v+=dot(s->x[xi],baseline_x[k],CHANNELS);
        xi=next((size_t)xi,XHIST);
    }
    s->at_y=back(s->at_y,YHIST);
    s->at_x=back(s->at_x,XHIST);
    out[0]=v+(BASE_DELTA ? row[0] : 0.);
    out[1]=out[2]=0.;
}
#endif

size_t stream_channels(void) { return CHANNELS; }
size_t stream_state_bytes(void) { return sizeof(State); }
void *stream_create(void) { return calloc(1,sizeof(State)); }
void stream_reset(void *state) { memset(state,0,sizeof(State)); }
void stream_free(void *state) { free(state); }

__attribute__((noinline))
void stream_tick(void *state, const double *row, double out[3]) {
    step(state,row,out);
}

/* Interleaved anchor and raw sensors, chronological from row zero. */
int replay(const double *data, size_t n, double *out) {
    void *s=stream_create();
    if (!s) return -1;
    for (size_t i=0; i<n; ++i)
        stream_tick(s,data+i*(CHANNELS+1),out+3*i);
    stream_free(s);
    return 0;
}

#ifndef STREAM_LIBRARY
static uint64_t timestamp(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW,&ts)) abort();
    return (uint64_t)ts.tv_sec*1000000000u+(uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc!=7) return 2;
    size_t n=strtoull(argv[3],NULL,10), first=strtoull(argv[4],NULL,10);
    size_t repeats=strtoull(argv[5],NULL,10);
    int timer_only=atoi(argv[6]);
    if (!n || n<=first || !repeats) return 2;
    size_t count=n*(CHANNELS+1);
    double *x=malloc(count*sizeof(*x));
    void *s=stream_create();
    uint64_t *timing=malloc((n-first)*repeats*sizeof(*timing));
    double *y=malloc(n*3*sizeof(*y));
    FILE *input=fopen(argv[1],"rb");
    if (!x || !s || !timing || !y || !input) return 3;
    if (fread(x,sizeof(*x),count,input)!=count) return 4;
    fclose(input);
    if (replay(x,n,y)) return 3;
    size_t j=0;
    for (size_t r=0; r<repeats; ++r) {
        stream_reset(s);
        if (!timer_only)
            for (size_t i=0; i<first; ++i) stream_tick(s,x+i*(CHANNELS+1),y+3*i);
        for (size_t i=first; i<n; ++i) {
            uint64_t t=timestamp();
            if (!timer_only) stream_tick(s,x+i*(CHANNELS+1),y+3*i);
            timing[j++]=timestamp()-t;
        }
    }
    FILE *file=fopen(argv[2],"wb");
    if (!file) return 5;
    uint64_t bytes=stream_state_bytes();
    if (fwrite(&bytes,sizeof(bytes),1,file)!=1 || fwrite(y,sizeof(*y),n*3,file)!=n*3
        || fwrite(timing,sizeof(*timing),j,file)!=j)
        return 6;
    if (fclose(file)) return 6;
    free(x);free(timing);free(y);stream_free(s);
    return 0;
}
#endif
