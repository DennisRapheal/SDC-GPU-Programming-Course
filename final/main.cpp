#include <iostream>
#include <random>
#include <cstdint>
#include <ctime>
#include <omp.h>
#include <iomanip>

#include <string.h>
#include "shishua.h"

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
    long double chunk_total = 0.0;
    const uint64_t chunk_size = 1000;
    uint64_t chunk_len = (total_trials + chunk_size - 1) / chunk_size;

    #pragma omp parallel for schedule(static) reduction(+:chunk_total)
    for (uint64_t chunk = 0; chunk < chunk_len; ++chunk) {
        // thread_local std::mt19937 gen(time(NULL) ^ omp_get_thread_num());
        thread_local uint64_t seed[ 4] = { (uint64_t)time(NULL) ^ omp_get_thread_num(), (uint64_t)time(NULL) ^ omp_get_thread_num() ^ 1, 
                                    (uint64_t) 2, (uint64_t)time(NULL) ^ omp_get_thread_num() ^ 3};
        thread_local ShishuaGenerator gen(seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        uint64_t start = chunk * chunk_size;
        uint64_t end = std::min(start + chunk_size, total_trials);

        uint64_t chunk_sum = 0;

        for (uint64_t i = start; i < end; ++i) {
            double s = 0.0;
            uint32_t count = 0;
            while (s < 1.0) {
                s += dist(gen);
                ++count;
            }
            chunk_sum += count;
        }

        chunk_total += static_cast<long double>(chunk_sum) / (end - start);
    }

    return chunk_total / chunk_len;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <total_trials>\n";
        return 1;
    }

    uint64_t trial = strtoull(argv[1], nullptr, 10);
    long double result = estimate_e_one_chunk(trial);
    std::cout << "Estimated e: " << std::fixed << std::setprecision(6) << result << "\n";

    return 0;
}
