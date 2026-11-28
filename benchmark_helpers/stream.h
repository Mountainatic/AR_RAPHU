#ifndef PRISM_STREAM_H
#define PRISM_STREAM_H

#include <stddef.h>

/* The generated model fixes the number of columns. Each row is [anchor, x...].
   The caller supplies the latest legally mature anchor for the issue clock.
   The three output slots are prediction, endogenous prediction, and trust. */
size_t stream_channels(void);
size_t stream_state_bytes(void);
void *stream_create(void);
void stream_reset(void *state);
void stream_tick(void *state, const double *row, double out[3]);
void stream_free(void *state);
int replay(const double *data, size_t n, double *out);

#endif
