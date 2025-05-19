make clean && make

# 測試不同 thread 數量的執行速度
for threads in 4 8 12 16 20; do
    export OMP_NUM_THREADS=$threads
    echo "Running with $OMP_NUM_THREADS threads..."
    time ./my_program
done