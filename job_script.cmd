#!/bin/bash
#SBATCH --job-name=VAE080621
#SBATCH -D .
#SBATCH --output=VAE080621_%j.out
#SBATCH --error=VAE080621_%j.err
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --qos=bsc_ls

module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1  python/3.7.4_ML torch/1.9.0a0

 
python main.py --hidden_layer=512 --features=32 --dropout=0.3 --epochs=120 --cycles=3 --initial_width=100 --reduction=35 --beta=20 --data_path=SingleCellCLL.csv --batch_size=128 --plots=1 --scaling=0


