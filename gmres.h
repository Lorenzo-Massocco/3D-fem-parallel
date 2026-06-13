#ifndef GMRES_H
#define GMRES_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "csr.h"

void gmres_jacobi(const CSRMatrix *A, const double *b, double *x, int restart,
                  int maxit, double tol, int *flag, int *iters, double *final_residual);

#endif
