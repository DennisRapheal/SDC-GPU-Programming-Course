#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#include <fstream>
#include <iomanip>  // for std::setprecision
#include <string>
#include <sstream>

using namespace std;

int main() {
    srand(42);
    omp_set_num_threads(12); // 不同 threads 數量可能會有精度差異
    cout << "OMP threads = " << omp_get_max_threads() << "\n";

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
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numParticles; i++) {
        velocity[i] = i * 1.0;
        pressure[i] = (numParticles - i) * 1.0;
        energy[i] = velocity[i] + pressure[i];
    }
    t2 = omp_get_wtime();
    cout << "初始化 velocity, pressure, energy 時間: " << (t2 - t1) << " 秒" << endl;

    // 4. 複雜運算：sin, log, fabs（O(n) 外加內層迴圈）
    t1 = omp_get_wtime();
    // 預計算 sin 查找表
    vector<double> sinTable(100); // Sufficient size
    for (int j = 0; j < 100; j++) { // Populate up to index 99
        sinTable[j] = sin(j * 0.01);
    }

    vector<double> precalculatedWorkSums(10);
    for (int k = 0; k < 10; ++k) {
        double currentWorkSum = 0.0;
        int currentLoops = k * 10 + 10;
        for (int j = 0; j < currentLoops; ++j) {
            currentWorkSum += sinTable[j]; // sinTable is already populated
        }
        precalculatedWorkSums[k] = currentWorkSum;
    }

    // 並行計算
    const double const_num_particles_val = static_cast<double>(numParticles);
    const double sin_of_num_particles = sin(const_num_particles_val); // Calculate this ONCE

    // (Make sure precalculatedWorkSums is already computed as you did)

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numParticles; i++) {
        double work = precalculatedWorkSums[i % 10];
        // energy[i] is const_num_particles_val, so sin(energy[i]) is sin_of_num_particles
        velocity[i] = sin_of_num_particles + log(1 + fabs(work));
    }
    t2 = omp_get_wtime();
    cout << "複雜運算（sin, log, fabs）時間: " << (t2 - t1) << " 秒" << endl;

    // 5. 二維網格運算（O(gridRows * gridCols)）
    t1 = omp_get_wtime();
    double fieldSum = 0.0;
    vector<double> sqrtTable(gridRows), logTable(gridCols);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < gridRows; r++) {
        sqrtTable[r] = sqrt(r * 2.0);
    }
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < gridCols; c++) {
        logTable[c] = log1p(c * 2.0);
    }

    double A = 0.0, B = 0.0;

    #pragma omp parallel for reduction(+:A)
    for (int r = 0; r < gridRows; ++r)
        A += sqrt(r * 2.0);

    #pragma omp parallel for reduction(+:B)
    for (int c = 0; c < gridCols; ++c)
        B += log1p(c * 2.0);

    fieldSum = gridCols * A + gridRows * B;

    t2 = omp_get_wtime();
    cout << "二維網格運算時間: " << (t2 - t1) << " 秒" << endl;

    // 6. atomicFlux 的計算 -> 加入 8. 

    // 7. criticalFlux 的計算（複雜邏輯，含條件判斷）
    t1 = omp_get_wtime();
    const double term_for_tempVal_calc = sqrt(fabs(const_num_particles_val)) / 100.0; // This is your 'tempVal', it's constant
    const double term_for_sqrt_tempVal_calc = sqrt(term_for_tempVal_calc);      // This is 'sqrt(tempVal)', also constant

    double criticalFlux = 0.0;
    for (int i = 0; i < numParticles; i++) {
        // energy[i] is const_num_particles_val.
        // Thus, tempVal is term_for_tempVal_calc.
        // And sqrt(tempVal) is term_for_sqrt_tempVal_calc.
        
        double extraVal = log(1 + fabs(velocity[i])) * 0.01; // velocity[i] comes from the previous "複雜運算" step
        double oldValue = criticalFlux;
        
        if (oldValue < 500.0) {
            criticalFlux = oldValue + term_for_tempVal_calc + extraVal;
        } else {
            criticalFlux = oldValue + term_for_sqrt_tempVal_calc - extraVal;
        }
    }
    t2 = omp_get_wtime();
    cout << "criticalFlux 計算時間: " << (t2 - t1) << " 秒" << endl;

    // 8. 累加檢查結果（三個 vector 的累加）
    t1 = omp_get_wtime();
    double atomicFlux = 0.0, sumVelocity = 0.0, sumPressure = 0.0, sumEnergy = 0.0;
    #pragma omp parallel for reduction(+:atomicFlux, sumVelocity, sumPressure, sumEnergy)
    for (int i = 0; i < numParticles; i++) {
        atomicFlux += velocity[i] * 0.000001;
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

    // verify the answer
    ostringstream oss;
    oss << fixed << setprecision(6);  // 固定小數點後 6 位
    oss << "fieldValue = "    << fieldSum    << "\n";
    oss << "energy[0] = "     << energy[0]   << "\n";
    oss << "sumVelocity = "   << sumVelocity << "\n";
    oss << "sumPressure = "   << sumPressure << "\n";
    oss << "sumEnergy = "     << sumEnergy   << "\n";
    oss << "atomicFlux = "    << atomicFlux  << "\n";
    oss << "criticalFlux = "  << criticalFlux << "\n";

    // 寫入或比對 ans.txt
    ifstream ansFileIn("ans.txt");

    if (!ansFileIn.good()) {
        // 如果 ans.txt 不存在，寫入作為標準答案
        ofstream ansFileOut("ans.txt");
        ansFileOut << oss.str();
        cout << "[INFO] ans.txt not found. Created new ans.txt as ground truth.\n";
        cout << "[INFO] Result saved. Verified: TRUE\n";
    } else {
        // 比對每行
        string refLine, curLine;
        istringstream curInput(oss.str());
        bool allMatch = true;
        int lineNum = 1;

        while (getline(ansFileIn, refLine) && getline(curInput, curLine)) {
            if (refLine != curLine) {
                allMatch = false;
                cout << "[Mismatch at line " << lineNum << "]\n";
                cout << "Expected: " << refLine << "\n";
                cout << "Actual  : " << curLine << "\n";
            }
            ++lineNum;
        }

        // 檢查是否行數不同
        if ((getline(ansFileIn, refLine) || getline(curInput, curLine)))
            allMatch = false;

        if (allMatch)
            cout << "[Verified] All results match ans.txt\n";
        else
            cout << "[Verification Failed]\n";
    }
    return 0;
}
