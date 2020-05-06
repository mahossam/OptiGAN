#!/bin/bash
#SBATCH --job-name=RelGAN_Job
#SBATCH --account=pb90
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --error=RelGAN_job-%j.err
#SBATCH --output=RelGAN_job-%j.out
#SBATCH --time=3-00:00
#SBATCH --mem-per-cpu=10000

nvidia-smi
# module load cuda

# export PATH="/home/mahmoudm/anaconda3/bin:$PATH"
. /home/mahmoudm/anaconda3/etc/profile.d/conda.sh
conda activate tf_new_py3

pwd
date
# python coco_relgan_pg_rl_only_meth1.py 0 0
# python coco_relgan_pg_rl_only_meth2.py 0 0
# python coco_relgan_pg_meth2.py 0 0
# python emnlp_relgan_rl_only_meth2.py 0 0
python emnlp_small_relgan_meth2.py 0 0

date
