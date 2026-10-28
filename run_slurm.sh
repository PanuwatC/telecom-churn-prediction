#!/bin/bash
# Run the whole notebook headless on an HPC node and save it with its outputs.
# Submit from the repository folder:  sbatch run_slurm.sh
#SBATCH --job-name=churn
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

# Spark needs Java; the module name differs between clusters (check with `module avail java`).
module load java 2>/dev/null || true
java -version

source .venv/bin/activate          # or: conda activate churn

jupyter nbconvert --to notebook --execute churn_prediction.ipynb --inplace \
    --ExecutePreprocessor.timeout=-1
cat outputs/summary.txt
