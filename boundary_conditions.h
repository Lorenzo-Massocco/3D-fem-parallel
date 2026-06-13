#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include "csr.h"

void apply_dirichlet_bc(CSRMatrix *A, double *rhs, int *bc_nodes, double *bc_values, int nbc);

void set_initial_condition(double *u, int n);

#endif
