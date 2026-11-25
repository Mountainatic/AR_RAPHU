#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

enum { DELAY=4, PREFIX_RING=256, RESIDUAL_RING=128, WIDTH=4, BINS=11, RECENT=8 };

typedef struct {
    size_t column0, column1;
    double mean0, mean1, weight0, weight1;
} Channel;

typedef struct {
    size_t source, difference, bins, width;
    double mean, inv_scale, intercept, upper, reference_knot;
    double center[WIDTH], knot[WIDTH], knot_inv_span[WIDTH], ref_inv_span;
    size_t near[BINS], far[BINS];
    double inv_width[BINS], low[BINS], high[BINS], inv_support[BINS], theta[BINS][WIDTH];
} View;

typedef struct {
    size_t near, width, use;
    double lo[RECENT], hi[RECENT], inv[RECENT], beta[WIDTH][RECENT];
    double far_beta[BINS][WIDTH];
} Packed;

#include "model.h"

typedef struct {
    // The mirrored point buffer exposes a contiguous vector for the exact near lags.
    double recent[VIEWS][WIDTH+1][2*RECENT];
    double prefix[VIEWS][PREFIX_RING][WIDTH+1];
    double level_delay[CHANNELS][DELAY];
    double anchors[DELAY], issued[DELAY], residual[RESIDUAL_RING], ema[7];
} State;

static uint64_t ns(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC_RAW,&stamp);
    return (uint64_t)stamp.tv_sec*1000000000+stamp.tv_nsec;
}

static double cube(double x) {
    const double p=x>0.0 ? x : 0.0;
    return (p*p)*p;
}

static double max4(__m256d x) {
    const __m128d pair=_mm_max_pd(_mm256_castpd256_pd128(x),_mm256_extractf128_pd(x,1));
    return _mm_cvtsd_f64(_mm_max_sd(pair,_mm_unpackhi_pd(pair,pair)));
}

static double sum4(__m256d x) {
    const __m128d pair=_mm_add_pd(_mm256_castpd256_pd128(x),_mm256_extractf128_pd(x,1));
    return _mm_cvtsd_f64(_mm_add_sd(pair,_mm_unpackhi_pd(pair,pair)));
}

static inline __attribute__((always_inline)) void ingest_view(State *s,size_t j,size_t recent,
        size_t head,size_t previous,const double *level,const double *difference) {
    if (!packed[j].use) return;
    const View *v=&views[j];
    const size_t width=packed[j].width;
    const double raw=v->difference ? difference[v->source] : level[v->source];
    s->recent[j][0][recent]=raw;
    s->recent[j][0][recent+RECENT]=raw;
    s->prefix[j][head][0]=s->prefix[j][previous][0]+raw;
    if (!width) return;
    const double z=(raw-v->mean)*v->inv_scale;
    double basis[WIDTH]={z-v->center[0],0.0,0.0,0.0};
    if (width==4) {
        const double capped=z<v->upper ? z : v->upper;
        const double common=cube(capped-v->reference_knot)*v->ref_inv_span;
        const __m256d knots=_mm256_loadu_pd(v->knot);
        const __m256d x=_mm256_max_pd(_mm256_setzero_pd(),_mm256_sub_pd(_mm256_set1_pd(capped),knots));
        const __m256d cubic=_mm256_mul_pd(_mm256_mul_pd(x,x),x);
        const __m256d slopes=_mm256_mul_pd(_mm256_set1_pd(3.0),
                                          _mm256_sub_pd(_mm256_set1_pd(v->reference_knot),knots));
        const __m256d nonlinear=_mm256_fmadd_pd(cubic,_mm256_loadu_pd(v->knot_inv_span),_mm256_set1_pd(-common));
        const __m256d values=_mm256_fmadd_pd(_mm256_set1_pd(z-capped),slopes,nonlinear);
        const __m256d merged=_mm256_blend_pd(values,_mm256_set1_pd(z),1);
        const __m256d centered=_mm256_sub_pd(merged,_mm256_loadu_pd(v->center));
        _mm256_storeu_pd(basis,centered);
    } else if (width>1) {
        const double capped=z<v->upper ? z : v->upper, tail=z-capped;
        const double common=cube(capped-v->reference_knot)*v->ref_inv_span;
        for (size_t m=1;m<width;++m)
            basis[m]=cube(capped-v->knot[m])*v->knot_inv_span[m]-common+
                tail*(3.0*(v->reference_knot-v->knot[m]))-v->center[m];
    }
    if (width==4) {
        const __m256d accum=_mm256_add_pd(_mm256_loadu_pd(&s->prefix[j][previous][1]),
                                          _mm256_loadu_pd(basis));
        _mm256_storeu_pd(&s->prefix[j][head][1],accum);
    }
    for (size_t m=0;m<width;++m) {
        const double value=basis[m];
        s->recent[j][m+1][recent]=value;
        s->recent[j][m+1][recent+RECENT]=value;
        if (width!=4) s->prefix[j][head][m+1]=s->prefix[j][previous][m+1]+value;
    }
}

static void ingest(State *s,const double *data,size_t t) {
    const size_t delay=t&(DELAY-1), recent=t&(RECENT-1);
    const size_t head=(t+1)&(PREFIX_RING-1), previous=t&(PREFIX_RING-1);
    double level[CHANNELS],difference[CHANNELS];
    for (size_t j=0;j<CHANNELS;++j) {
        const Channel *c=&channels[j];
        level[j]=(data[c->column0]-c->mean0)*c->weight0+(data[c->column1]-c->mean1)*c->weight1;
        difference[j]=t<DELAY ? 0.0 : level[j]-s->level_delay[j][delay];
        s->level_delay[j][delay]=level[j];
    }
    #define INGEST(j) ingest_view(s,j,recent,head,previous,level,difference);
    FOR_EACH_VIEW(INGEST)
    #undef INGEST
}

static double near_outside(const State *s,size_t j,size_t point) {
    const Packed *m=&packed[j];
    __m256d e=_mm256_setzero_pd();
    for (size_t b=0;b<m->near;b+=4) {
        const __m256d x=_mm256_loadu_pd(&s->recent[j][0][point+b]);
        const __m256d below=_mm256_sub_pd(_mm256_loadu_pd(&m->lo[b]),x);
        const __m256d above=_mm256_sub_pd(x,_mm256_loadu_pd(&m->hi[b]));
        const __m256d d=_mm256_mul_pd(_mm256_max_pd(below,above),_mm256_loadu_pd(&m->inv[b]));
        e=_mm256_max_pd(e,d);
    }
    return max4(e);
}

static double update_outside(double outside,const View *v,size_t b,const double *a,const double *z) {
    const double x=(a[0]-z[0])*v->inv_width[b];
    const double lower=v->low[b]-x,higher=x-v->high[b];
    const double d=(lower>higher ? lower : higher)*v->inv_support[b];
    return outside>d ? outside : d;
}

static double trust_weight(double outside) {
    const double u=outside<1.0 ? outside : 1.0, complement=1.0-u;
    return complement*complement*(1.0+2.0*u);
}

static double view_trust(const State *s,size_t j,size_t head,size_t point) {
    const View *v=&views[j];
    double outside=near_outside(s,j,point);
    if (outside>=1.0) return 0.0;
    for (size_t b=packed[j].near;b<v->bins;++b) {
        const size_t r=(head-v->near[b])&(PREFIX_RING-1), l=(head-v->far[b])&(PREFIX_RING-1);
        outside=update_outside(outside,v,b,s->prefix[j][r],s->prefix[j][l]);
    }
    return trust_weight(outside);
}

static inline __attribute__((always_inline)) double view_prediction(const State *s,size_t j,size_t head,
        size_t point,double *value) {
    const View *v=&views[j];
    const Packed *m=&packed[j];
    double outside=near_outside(s,j,point);
    if (outside>=1.0) {
        *value=0.0;
        return 0.0;
    }
    __m256d p=_mm256_setzero_pd();
    for (size_t b=0;b<m->near;b+=4)
        for (size_t basis=0;basis<m->width;++basis)
            p=_mm256_fmadd_pd(_mm256_loadu_pd(&s->recent[j][basis+1][point+b]),
                             _mm256_loadu_pd(&m->beta[basis][b]),p);
    double q=v->intercept+sum4(p);
    if (m->width==1) {
        for (size_t b=m->near;b<v->bins;++b) {
            const size_t r=(head-v->near[b])&(PREFIX_RING-1),l=(head-v->far[b])&(PREFIX_RING-1);
            const double *a=s->prefix[j][r],*z=s->prefix[j][l];
            outside=update_outside(outside,v,b,a,z);
            q+=(a[1]-z[1])*m->far_beta[b][0];
        }
    } else {
        p=_mm256_setzero_pd();
        for (size_t b=m->near;b<v->bins;++b) {
            const size_t r=(head-v->near[b])&(PREFIX_RING-1),l=(head-v->far[b])&(PREFIX_RING-1);
            const double *a=s->prefix[j][r],*z=s->prefix[j][l];
            outside=update_outside(outside,v,b,a,z);
            const __m256d d=_mm256_sub_pd(_mm256_loadu_pd(a+1),_mm256_loadu_pd(z+1));
            p=_mm256_fmadd_pd(d,_mm256_loadu_pd(m->far_beta[b]),p);
        }
        q+=sum4(p);
    }
    *value=q;
    return trust_weight(outside);
}

static inline __attribute__((always_inline)) void read_view(const State *s,size_t j,size_t head,double *ref,
        double *ref_trust,double *maximum,double *weighted,double *shift) {
    const Packed *m=&packed[j];
    if (m->width==0) return;
    const size_t z=(head&(RECENT-1))+RECENT-m->near;
    double q;
    const double w=view_prediction(s,j,head,z,&q);
    *maximum=w>*maximum ? w : *maximum;
    if (w==0.0) return;
    if (j==REFERENCE) {
        *ref=q;
        *ref_trust=w;
    }
    if (C_ACTIVE) {
        *shift+=(q-c_mean[j])*w*c_scaled_coef[j];
        *weighted+=w*c_abs_coef[j];
    }
}

static double w_readout(double x) {
    double z=(x-w_mean)*w_inv_scale;
    z=z<w_edges[0] ? w_edges[0] : z;
    z=z>w_edges[W_SEGMENTS] ? w_edges[W_SEGMENTS] : z;
    size_t i=0;
    while (i+1<W_SEGMENTS && z>=w_edges[i+1]) ++i;
    const double t=(z-w_edges[i])*w_inv_span[i],*c=w_poly[i];
    return (((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4];
}

static double upstream(const State *s,size_t t) {
    const size_t head=t+1;
    double ref=0.0,ref_trust=0.0,maximum=0.0,weighted=0.0,shift=0.0;
    #define READ(j) read_view(s,j,head,&ref,&ref_trust,&maximum,&weighted,&shift);
    FOR_EACH_VIEW(READ)
    #undef READ
    if (NEEDS_MAX_TRUST && maximum<1.0) {
        for (size_t j=0;j<VIEWS;++j) {
            const Packed *m=&packed[j];
            if (!m->use || m->width!=0) continue;
            const size_t z=(head&(RECENT-1))+RECENT-m->near;
            const double w=view_trust(s,j,head,z);
            maximum=w>maximum ? w : maximum;
            if (maximum==1.0) break;
        }
    }
    double out=K_ACTIVE ? ref*ref_trust : 0.0;
    if (C_ACTIVE) {
        out+=shift+c_intercept*maximum;
        if (W_ACTIVE) out+=(c_sum_abs>0.0 ? weighted*c_inv_sum_abs : maximum)*w_readout(out);
    } else if (W_ACTIVE) {
        out+=ref_trust*w_readout(ref);
    }
    return out;
}

static double update_a(State *s,size_t t,size_t n,double mature,double delta) {
    if (!A_ACTIVE) return 0.0;
    const size_t slot=t&(DELAY-1);
    double out=0.0;
    if (n>=DELAY) {
        const double r=(mature-s->anchors[slot])-s->issued[slot];
        s->residual[t&(RESIDUAL_RING-1)]=r;
        out=a_bias;
        for (size_t j=0;j<A_RECENT;++j)
            out+=a_weight[j]*s->residual[(t-a_offsets[j])&(RESIDUAL_RING-1)];
        const double older=s->residual[(t-A_EMA_SHIFT)&(RESIDUAL_RING-1)];
        for (size_t j=0;j<A_EMA;++j) {
            s->ema[j]+=a_alpha[j]*(older-s->ema[j]);
            out+=a_weight[A_RECENT+j]*s->ema[j];
        }
    }
    s->anchors[slot]=mature;
    s->issued[slot]=delta;
    return out;
}

double replay(const double *data,size_t start,size_t rows,double *out) {
    State state;
    memset(&state,0,sizeof(state));
    for (size_t i=0;i<start;++i) ingest(&state,data+i*STRIDE,i);
    const uint64_t begin=ns();
    for (size_t i=0;i<rows;++i) {
        const size_t t=start+i;
        ingest(&state,data+t*STRIDE,t);
        const double mature=data[(t-DELAY)*STRIDE];
        const double q=upstream(&state,t);
        out[i]=mature+(q+update_a(&state,t,i,mature,q));
    }
    return (double)(ns()-begin)/(double)rows;
}
