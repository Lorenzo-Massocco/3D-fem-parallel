#include "csr.h"

int csr_alloc(CSRMatrix *A, int n, int nnz) {
    A->n = n;
    A->nnz = nnz;
    A->rowptr = malloc((n + 1) * sizeof(int));
    A->col = malloc(nnz * sizeof(int));
    A->val = malloc(nnz * sizeof(double));
    if (!A->rowptr || !A->col || !A->val) return 1;
    return 0;
}

void csr_free(CSRMatrix *A) {
    if (A) {
        free(A->rowptr);
        free(A->col);
        free(A->val);
        A->rowptr = NULL;
        A->col = NULL;
        A->val = NULL;
        A->n = 0;
        A->nnz = 0;
    }
}

void csr_spmv(const CSRMatrix *A, const double *restrict x, double *restrict y)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A->n; i++) {

        double sum = 0.0;
        const int start = A->rowptr[i];
        const int end   = A->rowptr[i + 1];

        for (int k = start; k < end; k++) {
            sum += A->val[k] * x[A->col[k]];
        }

        y[i] = sum;
    }
}

void csr_copy(const CSRMatrix *A, CSRMatrix *B) {
    csr_alloc(B, A->n, A->nnz);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= A->n; i++) {
        B->rowptr[i] = A->rowptr[i];
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A->nnz; i++) {
        B->col[i] = A->col[i];
        B->val[i] = A->val[i];
    }
}

double dot_product(const double *a, const double *b, int n) {
    double s = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:s)
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

double norm2(const double *x, int n) {
    return sqrt(dot_product(x, x, n));
}

void axpy(double alpha, const double *x, double *y, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

void vec_copy(const double *x, double *y, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}

void vec_set(double *x, double value, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        x[i] = value;
    }
}

void vec_sub(const double *a, const double *b, double *c, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        c[i] = a[i] - b[i];
    }
}


