#!/bin/bash
#SBATCH --job-name=defDETR       # Kurzname des Jobs
#SBATCH --output=U-%j.out
#SBATCH --partition=p1           #--partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

weights=r50_deformable_detr-checkpoint.pth
num_classes=0

while [ $# -gt 0 ]; do
  case "$1" in
    -w|-weights|--weights) weights="$2" ;;
    -n|-num_classes|--num_classes) num_classes="$2" ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
  shift
done

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate deformable_detr

srun python reinit.py --resume $weights --num_classes $num_classes
