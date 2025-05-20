#include <iostream>
#include <random>
#include <cstdint>
#include <ctime>
#include <omp.h>
#include <iomanip>

#include <string.h>
#include "shishua-neon.h"
#include <algorithm>

// real euler number: 2.71828 18284 59045 23536 02874 71352 66249 77572 47093 69995..... 50 decimals

class ShishuaGenerator {
public:
    using result_type = uint64_t;

    explicit ShishuaGenerator(const uint64_t seed[4]) {
        memcpy(local_seed, seed, sizeof(uint64_t) * 4);
        prng_init(&state, local_seed);
        output_index = 16;  // Force a first gen
    }

    result_type operator()() {
        if (output_index >= 16) {
            // Fill output[] in prng_state
            prng_gen(&state, (uint8_t *)&buffer, 16 * 8); // 128 bytes = 16 * 8
            output_index = 0;
        }
        // return ((uint64_t *)state.output)[output_index++];
        return buffer[output_index++];
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }

private:
    prng_state state;
    uint64_t local_seed[4];
    uint64_t buffer[16] = { 0};
    size_t output_index = 16;
};

long double estimate_e_one_chunk(uint64_t total_trials) {
    const uint64_t chunk_size = 10000000;
    uint64_t chunk_len = (total_trials + chunk_size - 1) / chunk_size;

    long double global_sum = 0.0;
    #pragma omp parallel default(none) \
        shared(chunk_len, total_trials, global_sum)   \
        firstprivate(chunk_size)
    {
        thread_local uint64_t seed[4] = {
            static_cast<uint64_t>(time(NULL)) ^ static_cast<uint64_t>(omp_get_thread_num()),
            static_cast<uint64_t>(time(NULL)) ^ static_cast<uint64_t>(omp_get_thread_num()) ^ 1,
            static_cast<uint64_t>(2),
            static_cast<uint64_t>(time(NULL)) ^ static_cast<uint64_t>(omp_get_thread_num()) ^ 3
        };
        thread_local ShishuaGenerator gen(seed);

        const double inv_2pow53 = 1.0 / (1ull << 53); // 給 2️⃣ 預留

        long double local_sum = 0.0;                  // ❷ 每執行緒自有累加器

        #pragma omp for schedule(static,1) nowait
        for (uint64_t chunk = 0; chunk < chunk_len; ++chunk) {

            const uint64_t start = chunk * chunk_size;
            const uint64_t end   = std::min(start + chunk_size, total_trials);
            const uint64_t range = end - start;

            uint64_t trial_sum = 0;

            for (uint64_t i = 0; i < range; ++i) {
                double s = 0.0;
                uint32_t cnt = 0;
                while (s < 1.0) {
                    // 2️⃣ 先維持 dist(gen)，之後可替換
                    s += std::uniform_real_distribution<double>{0.0,1.0}(gen);
                    ++cnt;
                }
                trial_sum += cnt;
            }
            local_sum += static_cast<long double>(trial_sum) / range;
        }

        #pragma omp atomic
        global_sum += local_sum;                      // ❸ 單一原子操作代替重複 reduction
    }

    return global_sum / chunk_len;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <total_trials> [num_threads]\n";
        return 1;
    }

    uint64_t trials = strtoull(argv[1], nullptr, 10);
    int req_threads = (argc >= 3) ? std::atoi(argv[2]) : 0;

    omp_set_dynamic(0);                // ① 關閉 dynamic
    if (req_threads > 0) omp_set_num_threads(req_threads);

    double t0 = omp_get_wtime();
    long double est = estimate_e_one_chunk(trials);
    double t1 = omp_get_wtime();

    constexpr long double TRUE_E = 2.71828182845904523536L;
    long double abs_err = std::fabsl(est - TRUE_E);
    long double rel_err = abs_err / TRUE_E;

    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Estimated e    : " << est << '\n';
    std::cout << "True e         : " << TRUE_E << '\n';
    std::cout << "Absolute Error : " << abs_err << '\n';
    std::cout << "Relative Error : " << std::setprecision(6) << (rel_err * 100) << "%\n";
    std::cout << "Execution Time : " << std::setprecision(4) << (t1 - t0) << " s\n";
}
