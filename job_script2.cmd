#!/bin/bash
#SBATCH --job-name=IG
#SBATCH -D .
#SBATCH --output=IG_%j.out
#SBATCH --error=IG_%j.err
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --qos=bsc_ls

module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1  python/3.7.4_ML torch/1.9.0a0

 
python IG.py --model_path=trained_models/model20_08_13_45_beta_10.0_channel_25.0.pth --data_path=SingleCellCLL.csv


