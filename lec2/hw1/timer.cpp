#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

int main() {
    srand(42);
    
    double t0 = omp_get_wtime(); // 整體起始時間

    const int numParticles = 10000000;
    const int gridRows = 50000;
    const int gridCols = 50000;

    // 1. 初始化三個 vector<double>
    double t1 = omp_get_wtime();
    vector<double> velocity(numParticles), pressure(numParticles), energy(numParticles);
    double t2 = omp_get_wtime();
    cout << "初始化 vector 時間: " << (t2 - t1) << " 秒" << endl;

    // 2. 初始化 velocity 和 pressure（兩次 O(n) 操作）
    t1 = omp_get_wtime();
    for (int i = 0; i < numParticles; i++) {
        velocity[i] = i * 1.0;
        pressure[i] = (numParticles - i) * 1.0;
        energy[i] = velocity[i] + pressure[i];
    }
    t2 = omp_get_wtime();
    cout << "初始化 velocity, pressure, energy 時間: " << (t2 - t1) << " 秒" << endl;

    // 3. 計算 energy（O(n) 操作）
    // t1 = omp_get_wtime();
    // for (int i = 0; i < numParticles; i++) {
    //     energy[i] = velocity[i] + pressure[i];
    // }
    // t2 = omp_get_wtime();
    // cout << "計算 energy 時間: " << (t2 - t1) << " 秒" << endl;

    // 4. 複雜運算：sin, log, fabs（O(n) 外加內層迴圈）
    t1 = omp_get_wtime();
    for (int i = 0; i < numParticles; i++) {
        double work = 0.0;
        int loops = (i % 10) * 10 + 10; // 10, 20, ..., 100
        for (int j = 0; j < loops; j++) {
            work += sin(j * 0.01);
        }
        velocity[i] = sin(energy[i]) + log(1 + fabs(work));
    }
    t2 = omp_get_wtime();
    cout << "複雜運算（sin, log, fabs）時間: " << (t2 - t1) << " 秒" << endl;

    // 5. 二維網格運算（O(gridRows * gridCols)）
    t1 = omp_get_wtime();
    double fieldSum = 0.0;
    for (int r = 0; r < gridRows; r++) {
        for (int c = 0; c < gridCols; c++) {
            fieldSum += sqrt(r * 2.0) + log1p(c * 2.0);
        }
    }
    t2 = omp_get_wtime();
    cout << "二維網格運算時間: " << (t2 - t1) << " 秒" << endl;

    // 6. atomicFlux 的計算
    t1 = omp_get_wtime();
    double atomicFlux = 0.0;
    for (int i = 0; i < numParticles; i++){
        atomicFlux += velocity[i] * 0.000001;
    }
    t2 = omp_get_wtime();
    cout << "atomicFlux 計算時間: " << (t2 - t1) << " 秒" << endl;

    // 7. criticalFlux 的計算（複雜邏輯，含條件判斷）
    t1 = omp_get_wtime();
    double criticalFlux = 0.0;
    for (int i = 0; i < numParticles; i++){
        double tempVal = sqrt(fabs(energy[i])) / 100.0;
        double extraVal = log(1 + fabs(velocity[i])) * 0.01;
        double oldValue = criticalFlux;
        if (oldValue < 500.0) {
            criticalFlux = oldValue + tempVal + extraVal;
        } else {
            criticalFlux = oldValue + sqrt(tempVal) - extraVal;
        }
    }
    t2 = omp_get_wtime();
    cout << "criticalFlux 計算時間: " << (t2 - t1) << " 秒" << endl;

    // 8. 累加檢查結果（三個 vector 的累加）
    t1 = omp_get_wtime();
    double sumVelocity = 0.0, sumPressure = 0.0, sumEnergy = 0.0;
    for (int i = 0; i < numParticles; i++) {
        sumVelocity += velocity[i];
        sumPressure += pressure[i];
        sumEnergy += energy[i];
    }
    t2 = omp_get_wtime();
    cout << "累加檢查結果時間: " << (t2 - t1) << " 秒" << endl;

    // 輸出結果
    cout << "=== result ===" << endl;
    cout << "fieldValue      = " << fieldSum << endl;
    cout << "energy[0]       = " << energy[0] << endl;
    cout << "Sum(velocity)   = " << sumVelocity << endl;
    cout << "Sum(pressure)   = " << sumPressure << endl;
    cout << "Sum(energy)     = " << sumEnergy << endl;
    cout << "atomicFlux      = " << atomicFlux << endl;
    cout << "criticalFlux    = " << criticalFlux << endl;

    double tEnd = omp_get_wtime();
    cout << "總執行時間: " << (tEnd - t0) << " 秒" << endl;

    return 0;
}
