python -u -m pp_query.evaluate \
    --seed 123 \
    --dont_print_args \
    --train_data_percentage 1.0 \
    --num_channels 182 \
    --channel_embedding_size 32 \
    --dec_recurrent_hidden_size 64 \
    --dec_num_recurrent_layers 1 \
    --dec_intensity_hidden_size 32 \
    --dec_num_intensity_layers 2 \
    --dec_act_func "gelu" \
    --dropout 0.01 \
    --checkpoint_path "./data/mooc/nhp_models/" \
    --train_epochs 100 \
    --num_workers 2 \
    --batch_size 128 \
    --log_interval 500 \
    --save_epochs 5 \
    --optimizer "adam" \
    --grad_clip 10000.0 \
    --lr 0.001 \
    --weight_decay 0.0 \
    --warmup_pct 0.01 \
    --lr_decay_style "constant" \
    --train_data_path "./data/mooc/train/" \
    --valid_data_path "./data/mooc/valid/" \
    --test_data_path "./data/mooc/test/" \
    --neural_hawkes \
    --cuda \
    --device_num 1 \
    --calculate_is_bounds \
    --marginal_mark_queries \
    --marg_query_n 3 \
    --num_seqs 1000 250 50 20 10 4 2 \
#    --normalize_by_window \
#    --loss_monotonic \
    