#!/bin/bash

##########################
# SMP Script for CANDIDE #
##########################
# Receive email when job finishes or aborts
#PBS -M tobiasliaudat@gmail.com
#PBS -m ea
# Set a name for the job
#PBS -N wavediff_test
# Join output and errors in one file
#PBS -j oe
# Set maximum computing time (e.g. 5min)
#PBS -l walltime=23:00:00
# Request number of cores
#PBS -l nodes=n03:ppn=8

module load tensorflow/2.11
source activate WF_PSF

cd /n05data/tliaudat/wf-projects/repos/wf-psf/long-runs/

python -u train_eval_plot_script_click_multi_cycle.py \
    --id_name _wf_PR_9_cycles_v1 \
    --suffix_id_name v1 \
    --base_id_name _wf_PR_9_cycles_ \
    --eval_only_param False \
    --total_cycles 9 \
    --saved_cycle cycle9 \
    --reset_dd_features True \
    --eval_only_param False \
    --project_dd_features True \
    --d_max 2 \
    --n_zernikes 45 \
    --save_all_cycles True \
    --n_bins_lda 8 \
    --opt_stars_rel_pix_rmse True \
    --pupil_diameter 256 \
    --n_epochs_param_multi_cycle "0" \
    --n_epochs_non_param_multi_cycle "75" \
    --l_rate_non_param_multi_cycle "0.1" \
    --l_rate_param_multi_cycle "0" \
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
    --dataset_folder /n05data/tliaudat/wf-projects/wf-phase-retrieval/data/ \
    --test_dataset_file test_Euclid_res_id_010_8_bins.npy \
    --train_dataset_file train_Euclid_res_2000_TrainStars_id_010_8_bins.npy \
    --base_path /n05data/tliaudat/wf-projects/wf-phase-retrieval/wf-outputs/ \
    --metric_base_path /n05data/tliaudat/wf-projects/wf-phase-retrieval/wf-outputs/metrics/ \
    --chkp_save_path /n05data/tliaudat/wf-projects/wf-phase-retrieval/wf-outputs/chkp/9_cycles/ \
    --plots_folder plots/9_cycles/ \
    --model_folder chkp/9_cycles/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \
    --star_numbers 1 \

