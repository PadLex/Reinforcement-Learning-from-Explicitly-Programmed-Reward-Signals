#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=140:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=slurm_train_lm_heldout_v3
#SBATCH --mail-type=END
#SBATCH --mail-user=gdrtodd@nyu.edu
#SBATCH --output=./slurm_outputs/slurm_train_lm_heldout_v3_%j.out

module purge

cd /scratch/gdt9380/ludii-lms
conda init
conda activate ludii-lms

python train.py --model code-llama-13b --bits 8 --lora --mask_names --use_val_set --dataset_type fitm --data_subcategories hunt race space war --block_size 1024 --save_dir ./logs/code-llama-13b-fitm-mask-heldout_v3 --save_freq 49968