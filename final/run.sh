#!/bin/bash

make clean
make

# 真實常數 e
TRUE_E=2.718281828459045

for t in 4 8 12 16 20 24 28 32; do
    # 將結果寫入 <thread>_result.csv
    RESULT_FILE="${t}_result.csv"
    echo "Trials,Estimated_e,Execution_Time(s),Relative_Error" > "$RESULT_FILE"

    # 逐一測試 10^1 到 10^10
    for exp in {1..10}
    do
        TRIALS=$(echo "10^$exp" | bc)

        echo "Running trials = $TRIALS"

        OUTPUT=$(./my_program "$TRIALS" "$t")

        EST_E=$(echo "$OUTPUT" | grep "Estimated e" | grep -oE "[0-9]+\.[0-9]+")
        TIME=$(echo "$OUTPUT" | grep "Execution Time" | grep -oE "[0-9]+\.[0-9]+")
        REL_ERROR=$(echo "scale=15; ($EST_E - $TRUE_E)/$TRUE_E" | bc -l | sed 's/-//')

        # 記錄結果
        echo "$TRIALS,$EST_E,$TIME,$REL_ERROR" >> "$RESULT_FILE"
    done

    echo "✅ Results saved to $RESULT_FILE"
done
