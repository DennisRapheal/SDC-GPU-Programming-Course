#include <stdio.h>
#include <omp.h>
#include <time.h>

#define SIZE 10000000

// 宣告陣列和變數
double a[SIZE], b[SIZE], c[SIZE];

int main() {
    
    int i;
    double start_time, end_time;

    // 初始化陣列
    for (i = 0; i < SIZE; i++) {
        a[i] = i * 1.0;
        b[i] = i * 2.0;
    }

    // 記錄開始時間
    start_time = omp_get_wtime();

    // 平行化迴圈
    #pragma omp parallel for
    for (i = 0; i < SIZE; i++) {
        c[i] = a[i] + b[i];
    }

    // 記錄結束時間
    end_time = omp_get_wtime();

    // 輸出計算時間
    printf("Computation completed! Time: %f seconds\n", end_time - start_time);

    // 驗證計算結果
    printf("Verification: c[0] = %f, c[%d] = %f\n", c[0], SIZE-1, c[SIZE-1]);

    return 0;
}
