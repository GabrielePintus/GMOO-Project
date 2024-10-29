python3 script-BO.py \
    --setup_file setups/setup-sinewave-mlp-BO.py \
    --dataset datasets/sinewave.parquet \
    --warmup_steps 10 \
    --optimization_steps 90 \
    --batch_size 128 \
    --n_epochs 100 \
    --tags sinewave BO plateau MLP

