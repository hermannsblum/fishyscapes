export WANDB_MODE=offline
rclone copy -P fssubmissions:fishyscapes_pr_$1 /cluster/scratch/blumh/
sbatch -n 4 --mem-per-cpu=30000 --time=23:00:00 --gpus=1 --tmp=150000 --mail-type=END,FAIL --wrap="python3 euler.py $1 lostandfound_fishyscapes"
sbatch -n 4 --mem-per-cpu=30000 --time=80:00:00 --gpus=1 --tmp=150000 --mail-type=END,FAIL --wrap="python3 euler.py $1 fishyscapes_static_ood"
sbatch -n 4 --mem-per-cpu=30000 --time=23:00:00 --gpus=rtx_3090:1 --tmp=50000 --mail-type=END,FAIL --wrap="python3 euler.py $1 fishyscapes_5000timingsamples"
sbatch -n 4 --mem-per-cpu=30000 --time=23:00:00 --gpus=1 --tmp=50000 --mail-type=END,FAIL --wrap="python3 euler_cityscapes.py $1 cityscapes_validation"
