#include <math.h>
#include "boundary_conditions.h"
#include "csr.h"

void apply_dirichlet_bc(CSRMatrix *A, double *rhs, int *bc_nodes, double *bc_values, int nbc)
{
    #pragma omp parallel for schedule(static)
    for (int ibc = 0; ibc < nbc; ibc++) {
        int row = bc_nodes[ibc];
        double value = bc_values[ibc];

        for (int k = A->rowptr[row]; k < A->rowptr[row + 1]; k++) {
            if (A->col[k] == row) {
                A->val[k] = 1.0;
            } else {
                A->val[k] = 0.0;
            }
        }
        rhs[row] = value;
    }
}

void set_initial_condition(double *u, int n)
{
    #pragma omp parallel for schedule(static)
    for(int i=0; i < n; i++)
    {
        u[i] = sin(0.01*i);
    }
}
