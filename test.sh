#!/bin/bash
#SBATCH --job-name=defDETR       # Kurzname des Jobs
#SBATCH --output=T-%j.out
#SBATCH --partition=p2           #--partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=64G                # RAM pro CPU Kern #20G #32G #64G

batch=6
data=/mnt/md0/user/schmittth/datasets/semmel/cocoSetups640/Semmel38 # /mnt/md0/user/schmittth/coco # /mnt/md0/user/schmittth/yolov7/data/coco.yaml
weights=semmel_r50_deformable_detr-checkpoint.pth
name=None

while [ $# -gt 0 ]; do
  case "$1" in
    -b|-batch|--batch)       batch="$2"   ;;	  
    -d|-data|--data)         data="$2"    ;;
    -w|-weights|--weights)   weights="$2" ;;
    -n|-name|--name)         name="$2"    ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
  shift
done

: $data
: ${_%.*}
: $(basename $_)
: ${_,,}
runName=$_

if [ $name != "None" ]; then
   runName=$runName$name
fi

export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate deformable_detr

srun python main.py --batch_size $batch --coco_path $data --resume $weights --output_dir runs/test/$runName-$SLURM_JOB_ID --cache_mode --eval
