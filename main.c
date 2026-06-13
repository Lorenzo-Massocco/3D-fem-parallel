#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "csr.h"
#include "gmres.h"
#include "fem.h"
#include "boundary_conditions.h"
#include "transient.h"

static void print_matrix_info(const CSRMatrix *A)
{
    printf("\n===== CSR MATRIX INFO =====\n");
    printf("Matrix dimension : %d x %d\n", A->n, A->n);
    printf("Nonzeros         : %d\n", A->nnz);

    double sparsity =
        100.0 * (1.0 - ((double)A->nnz / ((double)A->n * A->n)));

    printf("Sparsity         : %.4f %%\n", sparsity);
}

static void check_matrix(const CSRMatrix *A)
{
    int invalid = 0;

    #pragma omp parallel for reduction(+:invalid)
    for (int i = 0; i < A->n; i++) {

        for (int k = A->rowptr[i]; k < A->rowptr[i + 1]; k++) {

            int j = A->col[k];

            if (j < 0 || j >= A->n)
                invalid++;

            if (!isfinite(A->val[k]))
                invalid++;
        }
    }

    if (invalid == 0)
        printf("Matrix consistency check : PASSED\n");
    else
        printf("Matrix consistency check : FAILED (%d issues)\n", invalid);
}

int main()
{
    printf("Number of threads available : %d\n", omp_get_max_threads());
    int active_threads = 1;
    omp_set_num_threads(active_threads);
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("Number of threads used      : %d\n\n", omp_get_num_threads());
        }
    }

    Node *nodes = NULL;
    Tetra *elem = NULL;

    int nnodes = 0;
    int nelem  = 0;

    printf("Reading nodes...\n");

    if (read_nodes("coord_1772481.coor", &nodes, &nnodes)) {
        fprintf(stderr, "Failed reading nodes.\n");
        return EXIT_FAILURE;
    }

    printf("Reading elements...\n");

    if (read_elements("elem_1772481.tetra", &elem, &nelem)) {
        fprintf(stderr, "Failed reading elements.\n");
        free(nodes);
        return EXIT_FAILURE;
    }

    printf("\n===== MESH INFO =====\n");
    printf("Nodes     : %d\n", nnodes);
    printf("Elements  : %d\n", nelem);

    double Dx = 1.0;
    double Dy = 1.0;
    double Dz = 1.0;

    double vx = 1.0;
    double vy = 0.0;
    double vz = 0.0;

    int flag;
    int iters;
    double final_res;

    int restart = 20;
    int maxit = 300;
    double tol = 1e-8;
    int nsteps = 100;
    double dt = 0.01;
    int nbc = 10, bc_nodes[10] = {10,11,12,13,14,15,16,17,18,19};
    double bc_values[10] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0};

    CSRMatrix A, M;

    printf("\nAssembling FEM matrix...\n");

    double runtime;

    assemble_system(nodes, elem, nelem, nnodes, Dx, Dy, Dz, vx, vy, vz, &A, &M, dt, &runtime);

    printf("Assembly completed.\n");
    print_matrix_info(&A);
    printf("Assembly time    : %.6f s\n", runtime);
    check_matrix(&A);
    int n = A.n;
    double *x = malloc(n * sizeof(double));
    double *f = malloc(n * sizeof(double));
    double *rhs = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        f[i] = 0.0;
    }
    set_initial_condition(x, n);
    int nfails = 0, max_iter = 0;
    double max_res = 0.0, gmres_runtime;
    CSRMatrix TEMP;
    csr_copy(&A, &TEMP);
    printf("\nSolving FEM formulation...\n");

    double start = omp_get_wtime();
    for(int t = 0; t < nsteps; t++){

        build_rhs(&M, x, f, dt, rhs);
        apply_dirichlet_bc(&TEMP, rhs, bc_nodes, bc_values, nbc);

        double t0 = omp_get_wtime();
        gmres_jacobi(&TEMP, rhs, x, restart, maxit, tol, &flag, &iters, &final_res);
        double t1 = omp_get_wtime();

        if (t == 50) gmres_runtime = t1 - t0; // for all meshes but the last one

        /* for the last mesh
        if (t == 1){
            gmres_runtime = t1 - t0;
            printf("Time elapsed for a GMRES iteration:                     %f s\n", gmres_runtime);
        }
        */
        if (flag == 1) nfails++;
        if (iters > max_iter) max_iter = iters;
        if (final_res > max_res) max_res = final_res;
        csr_copy(&A, &TEMP);

        if (t == 19 || t == 39 || t == 59 || t == 79)
            printf("%0.2f %% of the simulation completed\n", (double)100 * (t+1)/nsteps);

    }
    double end = omp_get_wtime();

    printf("FEM simulation completed.\n\n");
    printf("===== FEM SIMULATION INFO =====\n");

    if (nfails == 0){
        printf("GMRES converged at each time step\n");
        printf("Maximum number of GMRES iterations over the time steps  : %d\n", max_iter);
        printf("Maximum residual norm over the time steps               : %e\n", max_res);
    }
    else {
        printf("GMRES failed %d times\n", nfails);
        printf("Maximum residual norm over the non converged time steps : %e\n", max_res);
    }
    printf("Time elapsed for the FEM solution                       : %f s\n", end - start);
    printf("Time elapsed for a random GMRES iteration               : %f s\n", gmres_runtime);

    free(x);
    free(f);

    csr_free(&A);

    free(nodes);
    free(elem);

    return 0;
}
