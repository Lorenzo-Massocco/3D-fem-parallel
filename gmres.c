#include "gmres.h"
#include "csr.h"
#include <string.h>

void gmres_jacobi(const CSRMatrix *A, const double *b, double *x, int restart,
                  int maxit, double tol, int *flag, int *iters, double *final_residual)
{
    int n = A->n;
    *flag = 1;
    *iters = 0;

    double *diag = malloc(n * sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        diag[i] = 1.0;
        for (int k = A->rowptr[i]; k < A->rowptr[i + 1]; k++) {
            if (A->col[k] == i) {
                diag[i] = (fabs(A->val[k]) > 1e-14) ? A->val[k] : 1.0;
                break;
            }
        }
    }

    double *r = malloc(n * sizeof(double));
    double *w = malloc(n * sizeof(double));
    double *V = calloc((restart + 1) * n, sizeof(double));
    double *H = calloc((restart + 1) * restart, sizeof(double));
    double *cs = calloc(restart, sizeof(double));
    double *sn = calloc(restart, sizeof(double));
    double *g = calloc(restart + 1, sizeof(double));

    double normb = norm2(b, n);
    if (normb == 0.0) normb = 1.0;

    double relres = 1.0;

    for (int iter_outer = 0; iter_outer < maxit; iter_outer += restart) {
        csr_spmv(A, x, w);
        vec_sub(b, w, r, n);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            r[i] /= diag[i];
        }
        double beta = norm2(r, n);
        relres = beta / normb;

        if (relres < tol) {
            *flag = 0;
            break;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            V[i] = r[i] / beta;
        }
        vec_set(g, 0.0, restart + 1);
        g[0] = beta;

        int j;
        for (j = 0; j < restart && (*iters) < maxit; j++) {
            (*iters)++;

            csr_spmv(A, &V[j * n], w);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                w[i] /= diag[i];
            }

            for (int i = 0; i <= j; i++) {
                H[i * restart + j] = dot_product(w, &V[i * n], n);
                axpy(-H[i * restart + j], &V[i * n], w, n);
            }

            H[(j + 1) * restart + j] = norm2(w, n);

            if (H[(j + 1) * restart + j] != 0.0) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; i++) {
                    V[(j + 1) * n + i] = w[i] / H[(j + 1) * restart + j];
                }
            }

            for (int i = 0; i < j; i++) {
                double temp = cs[i] * H[i * restart + j] + sn[i] * H[(i + 1) * restart + j];
                H[(i + 1) * restart + j] = -sn[i] * H[i * restart + j] + cs[i] * H[(i + 1) * restart + j];
                H[i * restart + j] = temp;
            }

            double h1 = H[j * restart + j];
            double h2 = H[(j + 1) * restart + j];
            double den = sqrt(h1 * h1 + h2 * h2);
            cs[j] = h1 / den;
            sn[j] = h2 / den;

            H[j * restart + j] = cs[j] * h1 + sn[j] * h2;
            H[(j + 1) * restart + j] = 0.0;

            double temp = cs[j] * g[j];
            g[j + 1] = -sn[j] * g[j];
            g[j] = temp;

            relres = fabs(g[j + 1]) / normb;
            if (relres < tol) {
                j++;
                *flag = 0;
                break;
            }
        }

        double *y = calloc(j, sizeof(double));
        for (int k = j - 1; k >= 0; k--) {
            y[k] = g[k];
            for (int i = k + 1; i < j; i++) {
                y[k] -= H[k * restart + i] * y[i];
            }
            y[k] /= H[k * restart + k];
        }

        for (int k = 0; k < j; k++) {
            axpy(y[k], &V[k * n], x, n);
        }
        free(y);

        if (*flag == 0) break;
    }

    *final_residual = relres;
    free(diag); free(r); free(w); free(V); free(H); free(cs); free(sn); free(g);
}
