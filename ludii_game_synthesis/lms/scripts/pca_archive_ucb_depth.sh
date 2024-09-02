#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pca_archive_ucb_depth
#SBATCH --mail-type=END
#SBATCH --mail-user=gdrtodd@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load jdk/20.0.2

cd /scratch/gdt9380/ludii-lms
conda init
conda activate ludii-lms
python evolution.py --fitness_evaluation_strategy uct --fitness_eval_timeout 90 --archive_type pca --mutation_selection_strategy ucb_depth --save_dir ./exp_outputs/pca_archive_ucb_depth_4_30_24 --num_threads 8 --overwrite