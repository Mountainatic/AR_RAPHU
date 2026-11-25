#define _POSIX_C_SOURCE 200809L
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "sparse_model.h"

enum { L=LENGTH, D=ANCHOR_DELAY };
typedef struct { double ring[STRIDE*2*L]; } State;

static uint64_t ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW,&t);
    return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;
}

static inline void ingress(State *s,const double *x,double mature,size_t phase) {
    if (active[0]) s->ring[phase]=s->ring[phase+L]=mature;
    for (size_t k=1;k<STRIDE;++k) {
        if (active[k]) {
            size_t j=2*L*k+phase;
            s->ring[j]=s->ring[j+L]=x[k];
        }
    }
}

static inline double readout(const double *p) {
#if NONZERO <= 16
    double a=0.,b=0.;
    for (size_t i=0;i+1<NONZERO;i+=2) {
        a=__builtin_fma(p[index[i]],weight[i],a);
        b=__builtin_fma(p[index[i+1]],weight[i+1],b);
    }
    if (NONZERO%2) a=__builtin_fma(p[index[NONZERO-1]],weight[NONZERO-1],a);
    return bias+(a+b);
#else
    __m256d a=_mm256_setzero_pd(),b=a;
    for (size_t i=0;i<PADDED;i+=8) {
        __m256i address=_mm256_loadu_si256((const __m256i *)(index+i));
        __m256i next=_mm256_loadu_si256((const __m256i *)(index+i+4));
        a=_mm256_fmadd_pd(_mm256_i64gather_pd(p,address,8),_mm256_loadu_pd(weight+i),a);
        b=_mm256_fmadd_pd(_mm256_i64gather_pd(p,next,8),_mm256_loadu_pd(weight+i+4),b);
    }
    __m256d sum=_mm256_add_pd(a,b);
    __m128d pair=_mm_add_pd(_mm256_castpd256_pd128(sum),_mm256_extractf128_pd(sum,1));
    return bias+_mm_cvtsd_f64(_mm_add_sd(pair,_mm_unpackhi_pd(pair,pair)));
#endif
}

double replay(const double *data,size_t start,size_t count,double *out) {
    State s;
    for (size_t t=start-L;t<start;++t) ingress(&s,data+t*STRIDE,data[(t-D)*STRIDE],(-t)&(L-1));
    const uint64_t begin=ns();
    for (size_t i=0;i<count;++i) {
        const size_t t=start+i,phase=(-t)&(L-1);
        const double mature=data[(t-D)*STRIDE];
        ingress(&s,data+t*STRIDE,mature,phase);
        out[i]=readout(s.ring+phase)+(DELTA ? mature : 0.0);
    }
    return (double)(ns()-begin)/(double)count;
}
