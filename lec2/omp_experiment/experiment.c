#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>  // usleep 模擬 I/O delay

#define SIZE 1000   // matrix size
#define LOG_FILE "output.log"

double **A, **B, **C;

void allocate_matrices() {
    A = (double **)malloc(SIZE * sizeof(double *));
    B = (double **)malloc(SIZE * sizeof(double *));
    C = (double **)malloc(SIZE * sizeof(double *));
    for (int i = 0; i < SIZE; i++) {
        A[i] = (double *)malloc(SIZE * sizeof(double));
        B[i] = (double *)malloc(SIZE * sizeof(double));
        C[i] = (double *)malloc(SIZE * sizeof(double));
    }
}

void init_matrices() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
            C[i][j] = 0.0;
        }
    }
}

void memory_bound_with_io() {
    FILE *fp = fopen(LOG_FILE, "w");
    if (!fp) {
        perror("Cannot open file");
        return;
    }

    #pragma omp parallel
    {
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                for (int k = 0; k < SIZE; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
                // 模擬 I/O op，寫入 log file
                if (j % 100 == 0) {
                    usleep(5000); // 模擬 Disk I/O delay（5ms）
                    fprintf(fp, "Thread %d processed row %d column %d\n",
                            omp_get_thread_num(), i, j);
                }
            }
        }
    }

    fclose(fp);
}

int main() {
    printf("Running with %d threads...\n", omp_get_max_threads());
    allocate_matrices();
    init_matrices();

    double start = omp_get_wtime();
    memory_bound_with_io();
    double end = omp_get_wtime();

    printf("Execution Time: %.3f seconds\n", end - start);

    return 0;
}
