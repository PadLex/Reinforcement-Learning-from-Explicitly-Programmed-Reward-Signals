#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=1
#SBATCH --job-name=fitness_eval_exp
#SBATCH --mail-type=END
#SBATCH --mail-user=gdrtodd@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load jdk/20.0.2

cd /scratch/gdt9380/ludii-lms
conda init
conda activate ludii-lms
python fitness_eval_experiment.py