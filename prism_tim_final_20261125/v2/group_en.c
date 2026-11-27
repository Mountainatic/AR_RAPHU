#include <math.h>
#include <stddef.h>
#include <string.h>

/* Solve the two-coordinate group proximal problem on the FP64 Gram matrix. */
int solve_groups(size_t groups, const double *gram, const double *xy, double step,
                 double penalty, double tolerance, double *solution) {
    const size_t n = 2*groups;
    const double threshold = step*penalty;
    double forward[n], next[n], gradient[n];
    double momentum = 1.0;
    memset(solution, 0, n*sizeof(*solution));
    memset(forward, 0, sizeof forward);
    for (int iter = 0; iter < 100000; ++iter) {
        for (size_t j = 0; j < n; ++j) {
            double dot = -xy[j];
            for (size_t k = 0; k < n; ++k)
                dot += gram[j*n+k]*forward[k];
            next[j] = forward[j]-step*dot;
        }
        for (size_t j = 0; j < n; j += 2) {
            double length = sqrt(next[j]*next[j]+next[j+1]*next[j+1]);
            double shrink = length > threshold ? 1.0-threshold/length : 0.0;
            next[j] *= shrink;
            next[j+1] *= shrink;
        }
        if (iter % 16 == 0) {
            for (size_t j = 0; j < n; ++j) {
                gradient[j] = -xy[j];
                for (size_t k = 0; k < n; ++k)
                    gradient[j] += gram[j*n+k]*next[k];
            }
            double error = 0.0;
            for (size_t j = 0; j < n; j += 2) {
                double norm = sqrt(next[j]*next[j]+next[j+1]*next[j+1]);
                double x = gradient[j], y = gradient[j+1];
                if (norm > 0.0) {
                    x += (penalty/norm)*next[j];
                    y += (penalty/norm)*next[j+1];
                }
                double residual = sqrt(x*x+y*y) - (norm == 0.0 ? penalty : 0.0);
                error = fmax(error, residual);
            }
            if (error <= tolerance) {
                memcpy(solution, next, sizeof next);
                return 0;
            }
        }
        double restart = 0.0;
        for (size_t j = 0; j < n; ++j)
            restart += (forward[j]-next[j])*(next[j]-solution[j]);
        double accel = 0.0;
        if (restart > 0.0) {
            momentum = 1.0;
        } else {
            double next_momentum = (1.0+sqrt(1.0+4.0*momentum*momentum))/2.0;
            accel = (momentum-1.0)/next_momentum;
            momentum = next_momentum;
        }
        for (size_t j = 0; j < n; ++j)
            forward[j] = next[j]+accel*(next[j]-solution[j]);
        memcpy(solution, next, sizeof next);
    }
    return 1;
}

