#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=2:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=filter_ngrams
#SBATCH --mail-type=END
#SBATCH --mail-user=edk202@nyu.edu
#SBATCH --output=slurm_%j.out

module load python/3.9.16

base_dir="/vast/edk202/NLP_corpora/Google_Books/20200217/eng"

total_files=19423
start_file=0
files_per_task=$(( (total_files + SLURM_NTASKS - 1) / SLURM_NTASKS ))

task_id=${SLURM_LOCALID}
task_start=$((start_file + task_id * files_per_task))
task_end=$((task_start + files_per_task - 1))

# Ensure we don't exceed the total file count
if [ $task_end -gt $((start_file + total_files - 1)) ]; then
  task_end=$((start_file + total_files - 1))
fi

python download_and_filter_ngrams.py \
    --ngram_size 5 \
    --processes 48 \
    --file_range $task_start $task_end \
    --vocab_file "${base_dir}/valid_vocab_membertest.txt" \
    --overwrite
