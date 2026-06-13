#include "fem.h"

int read_nodes(const char *filename, Node **nodes, int *nnodes) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open nodes file '%s'\n", filename);
        return 1;
    }

    int count = 0;
    int id;
    double x, y, z;
    while (fscanf(fp, "%d %lf %lf %lf", &id, &x, &y, &z) == 4) {
        count++;
    }

    if (count == 0) {
        fprintf(stderr, "Error: Nodes file '%s' is empty or invalid.\n", filename);
        fclose(fp);
        return 1;
    }

    rewind(fp);
    *nodes = malloc(count * sizeof(Node));
    if (!(*nodes)) {
        fclose(fp);
        return 1;
    }

    for (int i = 0; i < count; i++) {
        if (fscanf(fp, "%d %lf %lf %lf", &id, &x, &y, &z) == 4) {
            (*nodes)[i].x = x;
            (*nodes)[i].y = y;
            (*nodes)[i].z = z;
        }
    }
    fclose(fp);
    *nnodes = count;
    return 0;
}

int read_elements(const char *filename, Tetra **elem, int *nelem) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open elements file '%s'\n", filename);
        return 1;
    }

    int count = 0;
    int id, n1, n2, n3, n4, reg;
    while (fscanf(fp, "%d %d %d %d %d %d", &id, &n1, &n2, &n3, &n4, &reg) == 6) {
        count++;
    }

    if (count == 0) {
        fprintf(stderr, "Error: Elements file '%s' is empty or invalid.\n", filename);
        fclose(fp);
        return 1;
    }

    rewind(fp);
    *elem = malloc(count * sizeof(Tetra));
    if (!(*elem)) {
        fclose(fp);
        return 1;
    }

    for (int i = 0; i < count; i++) {
        if (fscanf(fp, "%d %d %d %d %d %d", &id, &n1, &n2, &n3, &n4, &reg) == 6) {
            (*elem)[i].v[0] = n1 - 1; // Convert 1-based indexing to 0-based
            (*elem)[i].v[1] = n2 - 1;
            (*elem)[i].v[2] = n3 - 1;
            (*elem)[i].v[3] = n4 - 1;
            (*elem)[i].region = reg;
        }
    }
    fclose(fp);
    *nelem = count;
    return 0;
}

static double det3(double A[3][3]) {
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
           A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
}

static int exists_in_row(int *cols, int start, int end, int col)
{
    for (int i = start; i < end; i++) {
        if (cols[i] == col) return 1;
    }
    return 0;
}

void build_csr_pattern(Tetra *elem, int nelem, int nnodes, CSRMatrix *A)
{
    int *row_nnz = calloc(nnodes, sizeof(int));

    if (!row_nnz) {
        fprintf(stderr, "Allocation failed.\n");
        exit(EXIT_FAILURE);
    }
    int **temp_cols = malloc(nnodes * sizeof(int*));

    for (int i = 0; i < nnodes; i++) {
        temp_cols[i] = NULL;
    }

    for (int e = 0; e < nelem; e++) {

        for (int i = 0; i < 4; i++) {

            int I = elem[e].v[i];

            for (int j = 0; j < 4; j++) {

                int J = elem[e].v[j];

                if (!exists_in_row(temp_cols[I], 0, row_nnz[I], J))
                {
                    row_nnz[I]++;
                    temp_cols[I] = realloc(temp_cols[I], row_nnz[I] * sizeof(int));
                    temp_cols[I][row_nnz[I]-1] = J;
                }
            }
        }
    }

    int nnz = 0;

    for (int i = 0; i < nnodes; i++) {
        nnz += row_nnz[i];
    }

    if (csr_alloc(A, nnodes, nnz)) {
        fprintf(stderr, "CSR allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    A->rowptr[0] = 0;

    for (int i = 0; i < nnodes; i++) {
        A->rowptr[i+1] = A->rowptr[i] + row_nnz[i];
    }

    for (int i = 0; i < nnodes; i++) {

        int start = A->rowptr[i];

        for (int j = 0; j < row_nnz[i]; j++) {
            A->col[start + j] = temp_cols[i][j];
        }
    }

    for (int i = 0; i < nnz; i++) {
        A->val[i] = 0.0;
    }

    for (int i = 0; i < nnodes; i++) {
        free(temp_cols[i]);
    }

    free(temp_cols);
    free(row_nnz);
}

static inline int csr_find_position(const CSRMatrix *A, int row, int col)
{
    for (int k = A->rowptr[row]; k < A->rowptr[row+1]; k++)
    {
        if (A->col[k] == col)
            return k;
    }

    return -1;
}

static inline void local_matrix(const Node *nodes, const Tetra el, double Dx, double Dy,
                         double Dz, double vx, double vy, double vz, double Ke[4][4], double Me[4][4], double dt)
{
    double x0 = nodes[el.v[0]].x;
    double y0 = nodes[el.v[0]].y;
    double z0 = nodes[el.v[0]].z;

    double x1 = nodes[el.v[1]].x;
    double y1 = nodes[el.v[1]].y;
    double z1 = nodes[el.v[1]].z;

    double x2 = nodes[el.v[2]].x;
    double y2 = nodes[el.v[2]].y;
    double z2 = nodes[el.v[2]].z;

    double x3 = nodes[el.v[3]].x;
    double y3 = nodes[el.v[3]].y;
    double z3 = nodes[el.v[3]].z;

    double J[3][3] = {
        {x1 - x0, x2 - x0, x3 - x0},
        {y1 - y0, y2 - y0, y3 - y0},
        {z1 - z0, z2 - z0, z3 - z0}
    };

    double detJ =
          J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
        - J[0][1]*(J[1][0]*J[2][2] - J[1][2]*J[2][0])
        + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);

    double volume = fabs(detJ) / 6.0;

    if (volume < 1e-15) {
        fprintf(stderr, "Degenerate tetrahedron detected.\n");
        exit(EXIT_FAILURE);
    }

    double invdet = 1.0 / detJ;

    double invJ[3][3];

    invJ[0][0] =  (J[1][1]*J[2][2] - J[1][2]*J[2][1]) * invdet;
    invJ[0][1] = -(J[0][1]*J[2][2] - J[0][2]*J[2][1]) * invdet;
    invJ[0][2] =  (J[0][1]*J[1][2] - J[0][2]*J[1][1]) * invdet;

    invJ[1][0] = -(J[1][0]*J[2][2] - J[1][2]*J[2][0]) * invdet;
    invJ[1][1] =  (J[0][0]*J[2][2] - J[0][2]*J[2][0]) * invdet;
    invJ[1][2] = -(J[0][0]*J[1][2] - J[0][2]*J[1][0]) * invdet;

    invJ[2][0] =  (J[1][0]*J[2][1] - J[1][1]*J[2][0]) * invdet;
    invJ[2][1] = -(J[0][0]*J[2][1] - J[0][1]*J[2][0]) * invdet;
    invJ[2][2] =  (J[0][0]*J[1][1] - J[0][1]*J[1][0]) * invdet;

    static const double grad_ref[4][3] = {
        {-1.0, -1.0, -1.0},
        { 1.0,  0.0,  0.0},
        { 0.0,  1.0,  0.0},
        { 0.0,  0.0,  1.0}
    };
    double grad[4][3];

    for (int a = 0; a < 4; a++) {

        for (int i = 0; i < 3; i++) {

            grad[a][i] =
                  invJ[0][i] * grad_ref[a][0]
                + invJ[1][i] * grad_ref[a][1]
                + invJ[2][i] * grad_ref[a][2];
        }
    }
    for (int i = 0; i < 4; i++) {

        for (int j = 0; j < 4; j++) {

            double diffusion =
                  Dx * grad[i][0] * grad[j][0]
                + Dy * grad[i][1] * grad[j][1]
                + Dz * grad[i][2] * grad[j][2];

            diffusion *= volume;
            double adv =
                (vx * grad[j][0]
               + vy * grad[j][1]
               + vz * grad[j][2]) * volume / 4.0;

            Me[i][j] = (i == j) ? volume/10 : volume/20;
            Ke[i][j] = dt * (diffusion + adv) + Me[i][j];
        }
    }
}

void assemble_system(Node *nodes, Tetra *elem, int nelem, int nnodes, double Dx, double Dy, double Dz,
                     double vx, double vy, double vz, CSRMatrix *A, CSRMatrix *M, double dt, double *runtime)
{
    build_csr_pattern(elem, nelem, nnodes, A);
    build_csr_pattern(elem, nelem, nnodes, M);

    double t0 = omp_get_wtime();;

    #pragma omp parallel for schedule(static)
    for (int e = 0; e < nelem; e++) {

        double Ke[4][4];
        double Me[4][4];

        local_matrix(nodes, elem[e], Dx, Dy, Dz, vx, vy, vz, Ke, Me, dt);

        for (int i = 0; i < 4; i++) {

            int I = elem[e].v[i];

            for (int j = 0; j < 4; j++) {

                int J = elem[e].v[j];

                int posA = csr_find_position(A, I, J);
                int posM = csr_find_position(M, I, J);

                #pragma omp atomic
                A->val[posA] += Ke[i][j];

                #pragma omp atomic
                M->val[posM] += Me[i][j];
            }
        }
    }

    double t1 = omp_get_wtime();
    *runtime = t1 - t0;
}
