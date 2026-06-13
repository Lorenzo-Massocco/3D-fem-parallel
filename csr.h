#ifndef CSR_H
#define CSR_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

typedef struct
{
    int n;
    int nnz;
    int *rowptr;
    int *col;
    double *val;

} CSRMatrix;

int csr_alloc(CSRMatrix *A, int n, int nnz);
void csr_free(CSRMatrix *A);
void csr_spmv( const CSRMatrix *A, const double *x, double *y);
double dot_product(const double *a, const double *b, int n);
double norm2( const double *x, int n);
void axpy(double alpha, const double *x, double *y, int n);
void vec_copy( const double *x, double *y, int n);
void vec_set( double *x, double value, int n);
void vec_sub( const double *a, const double *b, double *c, int n);
void csr_copy( const CSRMatrix *A, CSRMatrix *B);

#endif
