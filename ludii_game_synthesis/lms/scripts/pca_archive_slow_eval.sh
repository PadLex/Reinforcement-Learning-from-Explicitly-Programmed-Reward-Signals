#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=96:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pca_archive_slow_eval
#SBATCH --mail-type=END
#SBATCH --mail-user=gdrtodd@nyu.edu
#SBATCH --output=./slurm_outputs/slurm_pca_archive_slow_%j.out

module purge
module load jdk/20.0.2

cd /scratch/gdt9380/ludii-lms
conda init
conda activate ludii-lms
python evolution.py --fitness_evaluation_strategy uct --games_per_eval 1 --num_fitness_evals 10 --thinking_time 0.25 --max_turns 100 --archive_type pca --save_dir ./exp_outputs/pca_archive_slow_eval --num_threads 8 --overwrite --add_current_date