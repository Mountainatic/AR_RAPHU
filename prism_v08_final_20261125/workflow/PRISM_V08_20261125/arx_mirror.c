#define _POSIX_C_SOURCE 200809L
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "arx_model.h"

enum { L=128, D=4 };
typedef struct { double samples[WIDTH][2*L]; } State;

static uint64_t ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW,&t);
    return (uint64_t)t.tv_sec*1000000000+t.tv_nsec;
}

static inline void ingress(State *s,const double *p,double y_mature,size_t j) {
    s->samples[0][j]=s->samples[0][j+L]=y_mature;
    for (size_t k=1;k<WIDTH;++k) s->samples[k][j]=s->samples[k][j+L]=p[k];
}

static inline double readout(const State *s,size_t j) {
#ifdef __AVX2__
    __m256d a=_mm256_setzero_pd(), b=a, c=a, d=a;
    for (size_t k=0;k<WIDTH;++k) {
        const double *x=s->samples[k]+j,*w=weights[k];
        for (size_t l=0;l<L;l+=16) {
            a=_mm256_fmadd_pd(_mm256_loadu_pd(x+l),_mm256_loadu_pd(w+l),a);
            b=_mm256_fmadd_pd(_mm256_loadu_pd(x+l+4),_mm256_loadu_pd(w+l+4),b);
            c=_mm256_fmadd_pd(_mm256_loadu_pd(x+l+8),_mm256_loadu_pd(w+l+8),c);
            d=_mm256_fmadd_pd(_mm256_loadu_pd(x+l+12),_mm256_loadu_pd(w+l+12),d);
        }
    }
    const __m256d sum=_mm256_add_pd(_mm256_add_pd(a,b),_mm256_add_pd(c,d));
    const __m128d pair=_mm_add_pd(_mm256_castpd256_pd128(sum),_mm256_extractf128_pd(sum,1));
    return bias+_mm_cvtsd_f64(_mm_add_sd(pair,_mm_unpackhi_pd(pair,pair)));
#else
    double a0=0.0,a1=0.0,a2=0.0,a3=0.0;
    for (size_t k=0;k<WIDTH;++k) {
        const double *x=s->samples[k]+j,*w=weights[k];
        for (size_t l=0;l<L;l+=4) {
            a0+=x[l]*w[l]; a1+=x[l+1]*w[l+1]; a2+=x[l+2]*w[l+2]; a3+=x[l+3]*w[l+3];
        }
    }
    return bias+((a0+a1)+(a2+a3));
#endif
}

double replay(const double *data,size_t start,size_t count,double *out) {
    State s;
    for (size_t t=start-L;t<start;++t) ingress(&s,data+t*WIDTH,data[(t-D)*WIDTH],(-t)&(L-1));
    const uint64_t begin=ns();
    for (size_t i=0;i<count;++i) {
        const size_t t=start+i,j=(-t)&(L-1);
        const double y_mature=data[(t-D)*WIDTH];
        ingress(&s,data+t*WIDTH,y_mature,j);
        out[i]=readout(&s,j)+(DELTA ? y_mature : 0.0);
    }
    return (double)(ns()-begin)/(double)count;
}
