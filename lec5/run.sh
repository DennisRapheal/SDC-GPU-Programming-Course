#!/bin/bash
#SBATCH --nodes=1           # Require 1 nodes to be used
#SBATCH --ntasks-per-node=1 # 1 tasks/processes to be executed in each node
#SBATCH --cpus-per-task=1   # 1 CPU cores to be used for each task/process
#SBATCH --mem=512M          # Specify the 512M real memory required per node
#SBATCH --gres=gpu:1        # Require 1 GPU
#SBATCH --time=0-0:00:10    # Set 10 seconds as waltime of the job
#SBATCH -p aipc             # Partition/Queue name

#==========================
# Load modules
#==========================

#==========================
# Execute Programs
#==========================

hipcc -o async ./asynchronous.cpp # Or your program name
./async # Or your binary file name