import os
import pandas as pd
import matplotlib.pyplot as plt

# === 基本設定 ===
BASE_DIR = './chunks'
CSV_SUFFIX = '_result.csv'

# === 收集所有資料：data[(thread_size, chunk_size)] = df ===
data = {}

chunk_dirs = sorted([d for d in os.listdir(BASE_DIR)
                     if d.startswith('chunk_size') and os.path.isdir(os.path.join(BASE_DIR, d))])

for chunk_dir in chunk_dirs:
    chunk_size = int(chunk_dir.replace("chunk_size", ""))
    folder_path = os.path.join(BASE_DIR, chunk_dir)

    for file in os.listdir(folder_path):
        if not file.endswith(CSV_SUFFIX): continue
        thread_size = int(file.split('_')[0])
        csv_path = os.path.join(folder_path, file)

        df = pd.read_csv(csv_path, sep=r'\t|,', engine='python')
        df['Trials'] = df['Trials'].astype(int)
        df.sort_values('Trials', inplace=True)

        data.setdefault(thread_size, {})[chunk_size] = df

# === 為每個 thread size 畫圖：比較不同 chunk_size 對執行時間的影響 ===
for thread_size, chunk_df_map in sorted(data.items()):
    plt.figure(figsize=(10, 6))

    for trials_val in [10**i for i in range(4, 11)]:  # 只畫 1e4 ~ 1e10
        x_chunks = []
        y_times = []

        for chunk_size, df in sorted(chunk_df_map.items()):
            matched_row = df[df['Trials'] == trials_val]
            if not matched_row.empty:
                x_chunks.append(chunk_size)
                y_times.append(matched_row['Execution_Time(s)'].values[0])

        if x_chunks:
            plt.plot(x_chunks, y_times, marker='o', label=f'Trials={trials_val:.0e}')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Chunk Size (log scale)')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time vs Chunk Size @ {thread_size} Threads')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    fname = f'exec_time_vs_chunksize_thread{thread_size}.png'
    plt.savefig(fname)
    print(f'✅ Saved: {fname}')
    plt.close()
