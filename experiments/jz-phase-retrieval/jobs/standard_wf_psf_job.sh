#!/bin/bash
#SBATCH --job-name=std_wf_psf_wfv1_test_n1_    # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par n?~Sud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH -C v100-32g 
# /!\ Attention, "multithread" fait reference a l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --output=std_wf_psf_wfv1_test_n1_%j.out  # nom du fichier de sortie
#SBATCH --error=std_wf_psf_wfv1_test_n1_%j.err   # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --mail-use=tobiasliaudat@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -A ynx@gpu                   # specify the project
##SBATCH --qos=qos_gpu-dev            # using the dev queue, as this is only for profiling
##SBATCH --array=0-1

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load tensorflow-gpu/py3/2.9.1

# echo des commandes lancees
set -x


cd $WORK/wfv2/repos/v1-phase-retrieval/wf-psf/long-runs/

srun python -u ./train_eval_plot_script_click_multi_cycle.py \
    --random_seed 3877572 \
    --base_id_name _wf_std_wf_psf_cycles_ \
    --id_name _wf_std_wf_psf_cycles_2 \
    --total_cycles 2 \
    --saved_cycle cycle2 \
    --eval_only_param False \
    --reset_dd_features True \
    --eval_only_param False \
    --project_dd_features True \
    --d_max 2 \
    --n_zernikes 45 \
    --save_all_cycles True \
    --n_bins_lda 8 \
    --n_bins_gt 8 \
    --opt_stars_rel_pix_rmse True \
    --pupil_diameter 256 \
    --n_epochs_param_multi_cycle "15 15" \
    --n_epochs_non_param_multi_cycle "100 50" \
    --l_rate_non_param_multi_cycle "0.1 0.06" \
    --l_rate_param_multi_cycle "0.01 0.004" \
    --model poly \
    --model_eval poly \
    --cycle_def complete \
    --gt_n_zernikes 45 \
    --d_max_nonparam 5 \
    --saved_model_type checkpoint \
    --use_sample_weights True \
    --l2_param 0. \
    --interpolation_type none \
    --eval_batch_size 16 \
    --train_opt True \
    --eval_opt True \
    --plot_opt True \
    --dataset_folder /gpfswork/rech/ynx/ulx23va/wfv2/dataset_pr/data/ \
    --test_dataset_file test_Euclid_res_id_010_8_bins.npy \
    --train_dataset_file train_Euclid_res_2000_TrainStars_id_010_8_bins.npy \
    --base_path /gpfsscratch/rech/ynx/ulx23va/wfv2/phase-retrieval-PR/output_v1/ \
    --metric_base_path /gpfsscratch/rech/ynx/ulx23va/wfv2/phase-retrieval-PR/output_v1/metrics/ \
    --chkp_save_path /gpfsscratch/rech/ynx/ulx23va/wfv2/phase-retrieval-PR/output_v1/chkp/ \
    --plots_folder plots/std_wf_psf/ \
    --model_folder chkp/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \
    --suffix_id_name 2 \
    --star_numbers 100 \



