#include "transient.h"

void build_rhs( const CSRMatrix *M, const double *uold, const double *f, double dt, double *rhs)
{
    csr_spmv(M, uold, rhs);

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < M->n; i++)
        rhs[i] += dt*f[i];
}
