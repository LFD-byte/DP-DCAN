python DCAN_main.py --data_file 'Muraro.h5' --experiment_name '1' --sigma 0.0 --z_dim 32 --select_genes 2000 --n_clusters 9 --pretrain_epochs 50 \
      --fit_epochs 40 --batch_size 210 --c1_dropout 0.2 --c2_dropout 0.2 --alpha 0.8 --beta1 0.4 --beta2 0.4 --beta3 0.2 --device 'cpu'

python DP_DCAN_main.py --data_file 'Muraro.h5' --experiment_name 'dp-1' --sigma 0.0 --z_dim 32 --select_genes 2000 --n_clusters 9 --pretrain_epochs 50 \
      --fit_epochs 40  --batch_size 210 --c1_dropout 0.2 --c2_dropout 0.2 --alpha 0.8 --beta1 0.4 --beta2 0.4 --beta3 0.2 --l2_norm_clip 0.1 \
      --noise_multiplier 3.46 --steps_dp 842 --device 'cpu'