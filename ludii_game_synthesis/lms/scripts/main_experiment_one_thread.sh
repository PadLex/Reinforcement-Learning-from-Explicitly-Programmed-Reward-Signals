#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=47:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=main_experiment_one_thread
#SBATCH --mail-type=END
#SBATCH --mail-user=gdrtodd@nyu.edu
#SBATCH --output=./slurm_outputs/main_experiment_one_thread_%j.out

module purge
module load jdk/20.0.2

cd /scratch/gdt9380/ludii-lms
conda init
conda activate ludii-lms
python evolution.py --model LudiiLMs/code-llama-13b-fitm-mask-heldout-1-epoch --fitness_evaluation_strategy uct \
    --games_per_eval 10 --num_fitness_evals 1 --thinking_time 0.25 --max_turns 50 --archive_type pca \
    --num_selections 3 --num_mutations 3 \
    --save_dir ./exp_outputs/main_experiment_one_thread --num_threads 1 --overwrite --add_current_date