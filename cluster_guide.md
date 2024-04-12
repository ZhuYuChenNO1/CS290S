# Environment Guide

This guide is for CS290S class students to use AI Cluster and install conda and torch environment correctly.


## Login AI Cluster
[AI Cluster User Manual](http://10.15.89.177:8889)

User name: `{EMAIL_PREFIX}-cs290s`  
Password: `sist`  

Login AI Cluster with `ssh`.

    ssh -p 22112 xxx@10.15.89.191

Change your password with `yppasswd` after the first login.

## Copy files with SCP

After finishing code blanks, copy your homework file to AI Cluster using `scp`, add `-r` if `SRC_PATH` is a folder.

    scp -P 22112 SRC_PATH xxx@10.15.89.191:DST_PATH


## Install Anaconda3 and initialize

`anaconda3` is already installed on `/public/software/anaconda3` for AI Cluster users.   

Run conda initialization for shell

    /public/software/anaconda3/bin/conda init
    source ~/.bashrc

(Optional) 

    conda config --set auto_activate_base false

If you are goine to use your own machine, please refer [https://www.anaconda.com/download](https://www.anaconda.com/download) to download and install.

## Create environment

Create conda environment with name `cs290s` and install `python3.8`

    conda create -n cs290s python==3.8
    conda activate cs290s

Change `pip` source, refer [https://mirrors.tuna.tsinghua.edu.cn/help/pypi](https://mirrors.tuna.tsinghua.edu.cn/help/pypi)

    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

Install `torch` with `pip`, since AI Cluster has `CUDA Version: 12.1`

    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

if you are going to use your own machine, use `nvidia-smi` to check your `CUDA Version` and refer [https://pytorch.org/get-started/previous-versions](https://pytorch.org/get-started/previous-versions). For `torch` installation, `pip` is always recommended against `conda` to avoid strange problems.

Install other requirements with `pip`

    pip install -r requirements.txt

## Submit Slurm training job

Write a slurm script `run.slurm` like:

    #!/bin/bash
    #SBATCH -J project1_training
    #SBATCH -p CS290S
    #SBATCH --cpus-per-task=4
    #SBATCH -N 1
    #SBATCH --gres=gpu:1
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err

    cd ~/project1_sequence_classification
    /public/home/CS290S/qiuqw-cs290s/.conda/envs/cs290s/bin/python train.py

where `/public/home/CS290S/qiuqw-cs290s/.conda/envs/cs290s/bin/python` is the `python` in your `conda` environment.  

Then submit the script:

    sbatch run.slurm

You can view your job status with:

    squeue

The `stdout` and `stderr` of your job is saved in `%j.out` and `%j.err` where `%j` is job id.

## View training logs on tensorboard

`train.py` save `tensorboard` logs into `output/logs`.

Run `tensorboard` with:

    tensorboard --logdir output/logs --port xxxxx --host 0.0.0.0

Then you can view `tensorboard` logs on `10.15.89.191:xxxxx`

To avoid `ssh` disconnection, run `tensorboard` in tmux.

### tmux quickstart

Create new tmux session:

    tmux new -s tensorboard

Detach current session: press `ctrl-b` and `d` inside session

View current session:

    tmux ls

Attach exist session:
    
    tmux attach -t tensorboard

Kill session: press `ctrl-d` inside session or:

    tmux kill-session -t tensorboard

For more about tmux, refer [https://tmuxcheatsheet.com](https://tmuxcheatsheet.com)

Note that tmux is not allowed to use on compute node 




