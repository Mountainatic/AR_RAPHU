#include "stream.c"

/* Stress-only diagnostic. The production model and replay ABI are untouched. */
static double source_value(const double *data, const Channel *channel) {
    return (data[channel->column0]-channel->mean0)*channel->weight0+
        (data[channel->column1]-channel->mean1)*channel->weight1;
}

void probe(const double *data,size_t period,size_t length,double *maxima) {
    State state={0};
    for (size_t t=0;t<length;++t) {
        ingest_anchor(&state,data,t,period);
        const size_t near_reset=t%65536;
        if (t<HISTORY+DELAY || (t%1024!=1023 && near_reset!=0)) continue;
        for (size_t j=0;j<VIEWS;++j) {
            if (!packed[j].use && !drift_uses_view(j)) continue;
            const View *view=&views[j];
            if (!view->bins) continue;
            const Channel *channel=&channels[view->source];
            const size_t before=view->far[view->bins-1];
            long double reference=0.0L;
            for (size_t i=t+1-before;i<=t;++i) {
                double z=source_value(sample(data,i,period),channel);
                if (view->difference) z-=source_value(sample(data,i-DELAY,period),channel);
                reference+=z;
            }
            const double a=state.prefix[j][(t+1)&(PREFIX_RING-1)][0];
            const double b=state.prefix[j][(t+1-before)&(PREFIX_RING-1)][0];
            const double diff=(double)fabsl((long double)(a-b)-reference)/(double)before;
            if (diff>maxima[j]) maxima[j]=diff;
        }
    }
}
