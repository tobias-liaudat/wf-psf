#/usr/bin/bash

cd $HOME
source .bash_profile

cd $WORK/repos/wf-psf/long-runs
pwd

# conda activate WF_PSF

export CUDA_VISIBLE_DEVICES=2


time /disk/xray0/tl3/miniconda3/envs/tf2_7_conda/bin/python -u train_eval_plot_script_click_multi_cycle.py \
    --id_name _wf_only-NP_02nm_v1_ \
    --suffix_id_name _ \
    --base_id_name _wf_only-NP_02nm_v1 \
    --eval_only_param False \
    --total_cycles 2 \
    --saved_cycle cycle2 \
    --reset_dd_features False \
    --eval_only_param False \
    --project_dd_features False \
    --d_max 2 \
    --n_zernikes 66 \
    --save_all_cycles True \
    --n_bins_lda 20 \
    --opt_stars_rel_pix_rmse True \
    --pupil_diameter 256 \
    --n_epochs_param_multi_cycle "0" \
    --n_epochs_non_param_multi_cycle "100 50" \
    --l_rate_non_param_multi_cycle "0.1 0.06" \
    --l_rate_param_multi_cycle "0.005 0.001" \
    --model poly_physical \
    --model_eval physical \
    --cycle_def only-non-parametric \
    --gt_n_zernikes 66 \
    --n_bins_gt 20 \
    --d_max_nonparam 5 \
    --saved_model_type checkpoint \
    --use_sample_weights True \
    --l2_param 0. \
    --interpolation_type none \
    --eval_batch_size 16 \
    --train_opt True \
    --eval_opt True \
    --plot_opt True \
    --dataset_folder /disk/xray0/tl3/datasets/wf-phase-retrieval/euclid_data_sims/data/ \
    --test_dataset_file test_SFE_10nm_id_11.npy \
    --train_dataset_file train_SFE_err_10nm_id_11.npy \
    --base_path /disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/ \
    --metric_base_path /disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/metrics/ \
    --chkp_save_path /disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/chkp/ \
    --plots_folder plots/ \
    --model_folder chkp/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \
    --star_numbers 1 \

time /disk/xray0/tl3/miniconda3/envs/tf2_7_conda/bin/python -u train_eval_plot_script_click_multi_cycle.py \
    --id_name _wf_only-NP_02nm_v2_ \
    --suffix_id_name _ \
    --base_id_name _wf_only-NP_02nm_v2 \
    --eval_only_param False \
    --total_cycles 2 \
    --saved_cycle cycle2 \
    --reset_dd_features False \
    --eval_only_param False \
    --project_dd_features False \
    --d_max 2 \
    --n_zernikes 66 \
    --save_all_cycles True \
    --n_bins_lda 20 \
    --opt_stars_rel_pix_rmse True \
    --pupil_diameter 256 \
    --n_epochs_param_multi_cycle "0" \
    --n_epochs_non_param_multi_cycle "100 50" \
    --l_rate_non_param_multi_cycle "0.1 0.06" \
    --l_rate_param_multi_cycle "0.005 0.001" \
    --model poly_physical \
    --model_eval physical \
    --cycle_def only-non-parametric \
    --gt_n_zernikes 66 \
    --n_bins_gt 20 \
    --d_max_nonparam 5 \
    --saved_model_type checkpoint \
    --use_sample_weights True \
    --l2_param 0. \
    --interpolation_type none \
    --eval_batch_size 16 \
    --train_opt True \
    --eval_opt True \
    --plot_opt True \
    --dataset_folder /disk/xray0/tl3/datasets/wf-phase-retrieval/euclid_data_sims/data/ \
    --test_dataset_file test_SFE_10nm_id_11.npy \
    --train_dataset_file train_SFE_err_10nm_id_11.npy \
    --base_path /disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/ \
    --metric_base_path /disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/metrics/ \
    --chkp_save_path /disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/chkp/ \
    --plots_folder plots/ \
    --model_folder chkp/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \
    --star_numbers 1 \

time /disk/xray0/tl3/miniconda3/envs/tf2_7_conda/bin/python -u train_eval_plot_script_click_multi_cycle.py \
    --id_name _wf_only-NP_02nm_v3_ \
    --suffix_id_name _ \
    --base_id_name _wf_only-NP_02nm_v3 \
    --eval_only_param False \
    --total_cycles 2 \
    --saved_cycle cycle2 \
    --reset_dd_features False \
    --eval_only_param False \
    --project_dd_features False \
    --d_max 2 \
    --n_zernikes 66 \
    --save_all_cycles True \
    --n_bins_lda 20 \
    --opt_stars_rel_pix_rmse True \
    --pupil_diameter 256 \
    --n_epochs_param_multi_cycle "0" \
    --n_epochs_non_param_multi_cycle "100 50" \
    --l_rate_non_param_multi_cycle "0.1 0.06" \
    --l_rate_param_multi_cycle "0.005 0.001" \
    --model poly_physical \
    --model_eval physical \
    --cycle_def only-non-parametric \
    --gt_n_zernikes 66 \
    --n_bins_gt 20 \
    --d_max_nonparam 5 \
    --saved_model_type checkpoint \
    --use_sample_weights True \
    --l2_param 0. \
    --interpolation_type none \
    --eval_batch_size 16 \
    --train_opt True \
    --eval_opt True \
    --plot_opt True \
    --dataset_folder /disk/xray0/tl3/datasets/wf-phase-retrieval/euclid_data_sims/data/ \
    --test_dataset_file test_SFE_10nm_id_11.npy \
    --train_dataset_file train_SFE_err_10nm_id_11.npy \
    --base_path /disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/ \
    --metric_base_path /disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/metrics/ \
    --chkp_save_path /disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/chkp/ \
    --plots_folder plots/ \
    --model_folder chkp/ \
    --log_folder log-files/ \
    --optim_hist_folder optim-hist/ \
    --star_numbers 1 \
