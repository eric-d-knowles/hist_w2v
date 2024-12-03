#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=filter_ngrams
#SBATCH --mail-type=END
#SBATCH --mail-user=edk202@nyu.edu
#SBATCH --output=slurm_%j_%a.out
#SBATCH --array=0-2

# Define base directory for input and output files
base_dir="/vast/edk202/NLP_corpora/Google_Books/20200217/eng"

# Total number of files to process
TOTAL_FILES=19423

# Calculate the start and end file indices based on the job array ID
FILES_PER_JOB=$((TOTAL_FILES / 3))
START_FILE=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB))
END_FILE=$(((SLURM_ARRAY_TASK_ID + 1) * FILES_PER_JOB - 1))

# Adjust for the last job to handle any remaining files (if necessary)
if [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    END_FILE=$TOTAL_FILES
fi

# Now you can use $START_FILE and $END_FILE for the range of files to process in each job
echo "Processing files from $START_FILE to $END_FILE"

# Run the Python script inside a singularity container
module purge
singularity exec \
    --overlay /scratch/edk202/hist_w2v_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh;
    python /scratch/edk202/hist_w2v/download_and_filter_ngrams.py \
        --ngram_size 5 \
        --processes 14 \
        --file_range $START_FILE $END_FILE \
        --vocab_file $base_dir/valid_vocab_membertest.txt \
        --overwrite \
	--save_empty"

