# Exploring RL-based LLM Training for Formal Language Tasks with Programmed Rewards

This repository contains the code for the paper "Exploring RL-based LLM Training for Formal Language Tasks with Programmed Rewards" by Alexander G. Padula and Dennis J.N.J. Soemers. The paper was excepted to [BNAIC 2024](https://bnaic2024.sites.uu.nl).

The code in this repository is provided for reproducibility, as well as serving as a reference for using [TRL](https://github.com/huggingface/trl) with external reward functions and for interfacing with the Ludii game-playing environment. It is not intended to be a package for general use, but rather a collection of scripts and notebooks that can be used to reproduce the experiments in the paper.

The modifications we made to TRL are available in [our fork](https://github.com/PadLex/trl) of the repository.

## Installation
The following instructions have been tested on Lambda instances with A100s or A5000s. Some dependencies are quite fragile and may not work on other systems. For example, at the time of writing, TRL is not compatible with any system with an H100.

Clone the repository and navigate to the root directory of the project:
```bash
git clone https://github.com/PadLex/Reinforcement-Learning-from-Explicitly-Programmed-Reward-Signals.git
cd LudiiRL
```

(optional) Create a virtual environment and activate it:
```bash
sudo apt install python3-venv
python -m venv torch_cuda_11-7
. ./torch_cuda_11-7/bin/activate
```

Install the required dependencies:
```bash
pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
python -c 'import torch ; print("Is available: ", torch.cuda.is_available()) ; print("Pytorch CUDA Compiled version: ", torch._C._cuda_getCompiledVersion()) ; print("Pytorch version: ", torch.version) ; print("pytorch file: ", torch.__file__)'
pip install git+https://github.com/PadLex/trl.git bitsandbytes wandb huggingface_hub nltk
sudo apt install openjdk-19-jdk openjdk-19-jre -y
```

Login to wandb and huggingface:
```bash
wandb login
huggingface-cli login
```
