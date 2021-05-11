DATASET="fishyscapes_web_apr2021"

cd /cluster/home/blumh/fishyscapes/fishyscapes
source euler_env
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/deeplab.yaml testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/dirichlet.yaml testing_dataset.base_path=$DATASET
bsub -W 24:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/dropout_mi.yaml testing_dataset.base_path=$DATASET
bsub -W 24:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/learned_density_ensemble_min.yaml testing_dataset.base_path=$DATASET
bsub -W 24:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/learned_density_ensemble_regression.yaml testing_dataset.base_path=$DATASET
bsub -W 24:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/learned_density_single_layer.yaml testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/outlier_head_combined_instances.yaml testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/outlier_head_combined.yaml testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/outlier_head_fixed_patches.yaml testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/outlier_head_instances_1024.yaml testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/outlier_head_random_patches.yaml testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/softmax_max_prob.yaml testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/void_max_entropy.yaml testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py saved_model with experiments/configs/void_prob.yaml testing_dataset.base_path=$DATASET
module purge
source /cluster/home/blumh/fishyscapes/synboost/euler_env
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py resynthesis_model with experiments/configs/synboost.yaml  testing_dataset.base_path=$DATASET
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" python experiments/fishyscapes.py resynthesis_model with experiments/configs/epfl_resynthesis.yaml  testing_dataset.base_path=$DATASET
module purge
source /cluster/home/blumh/fishyscapes/robin/euler_env
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" -R "select[gpu_model0==TITANRTX]" python experiments/fishyscapes.py entropy_maximization  with experiments/configs/buwki_entropymax.yaml  testing_dataset.base_path=$DATASET
module purge
source /cluster/home/blumh/fishyscapes/awesomemango/euler_env
bsub -W 4:00 -n 2 -R "rusage[mem=60000,scratch=10000]" -R "rusage[ngpus_excl_p=1]" -R "select[gpu_model0==TITANRTX]" python experiments/fishyscapes.py ood_segmentation with experiments/configs/buwki_entropymax.yaml  testing_dataset.base_path=$DATASET
