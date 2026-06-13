#ifndef FEM_H
#define FEM_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "csr.h"

typedef struct
{
    double x,y,z;
} Node;

typedef struct
{
    int v[4];
    int region;
} Tetra;

int read_nodes(const char *filename, Node **nodes, int *nnodes);
int read_elements(const char *filename, Tetra **elem, int *nelem);
void build_csr_pattern(Tetra *elem, int nelem, int nnodes, CSRMatrix *A);
void assemble_system(Node *nodes, Tetra *elem, int nelem, int nnodes, double Dx,
                     double Dy, double Dz, double vx, double vy, double vz, CSRMatrix *A, CSRMatrix *M, double dt, double *runtime);

#endif
