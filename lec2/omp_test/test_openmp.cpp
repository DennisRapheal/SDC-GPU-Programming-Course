#include <iostream>
#include <omp.h>

int main() {
    std::cout << "OpenMP Version: " << _OPENMP << std::endl;

    #pragma omp parallel
    {
        #pragma omp critical
        {
            std::cout << "Thread " << omp_get_thread_num() 
                      << " of " << omp_get_num_threads() 
                      << std::endl;
        }
    }
    
    return 0;
}
