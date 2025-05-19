#!/bin/bash

#SBATCH -A lintinwei20030906   # -A, --account : 指定帳戶 ID (通常是計算資源的計費帳戶)
#SBATCH -p aipc # -p, --partition : 指定 job 需要的 partition (類似 queue)
#SBATCH -J mpi_test   # -J, --job-name : 設定 job 名稱
#SBATCH -N 1            # -N, --nodes : 指定使用的節點數 (這裡為 1 個節點)
#SBATCH -n 4            # -n, --ntasks : 指定總共要執行的 task 數 (這裡是 4)，可以看成幾次
#SBATCH -c 1            # -c, --cpus-per-task : 指定每個 task 需要的 CPU 數 (這裡是 1)
#SBATCH -o batch_job_test1.out  # -o, --output : 指定標準輸出 (stdout) 的輸出檔案
#SBATCH -e batch_job_test1.err  # -e, --error : 指定錯誤輸出 (stderr) 的輸出檔案

mpicc mpi_test.c -o mpi_test && mpirun -np 4 ./mpi_test