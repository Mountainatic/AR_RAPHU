#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "stream.h"

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const size_t period = 4096, warmup = 4096, count = 1000000;
    const size_t interval = 10000, probe = 128, width = stream_channels() + 1;
    double *data = malloc(period * width * sizeof(*data));
    void *continuous = stream_create(), *recent = stream_create();
    FILE *f = fopen(argv[1], "rb");
    if (!f || !data || !continuous || !recent) return 3;
    if (fread(data, sizeof(*data), period*width, f) != period*width) return 4;
    fclose(f);
    double a[3], b[3], max[3] = {0., 0., 0.};
    puts("issue,points,prediction_max_abs,endo_max_abs,trust_max_abs");
    for (size_t t = 0; t < count; ++t) {
        const size_t pos = t % interval;
        if (t >= interval && pos == 0) {
            stream_reset(recent);
            for (size_t past = t-warmup; past < t; ++past)
                stream_tick(recent, data+(past%period)*width, b);
            for (int i = 0; i < 3; ++i) max[i] = 0.;
        }
        const double *row = data+(t%period)*width;
        stream_tick(continuous,row,a);
        if (t >= interval && pos < probe) {
            stream_tick(recent,row,b);
            for (int i = 0; i < 3; ++i)
                max[i] = fmax(max[i],fabs(a[i]-b[i]));
            if (pos+1 == probe)
                printf("%zu,%zu,%.17g,%.17g,%.17g\n",t-pos,probe,max[0],max[1],max[2]);
        }
    }
    stream_free(recent);
    stream_free(continuous);
    free(data);
    return 0;
}
