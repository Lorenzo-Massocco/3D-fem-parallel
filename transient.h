#ifndef TRANSIENT_H
#define TRANSIENT_H

#include "csr.h"

void build_rhs(const CSRMatrix *M, const double *uold, const double *f, double dt, double *rhs);

#endif
