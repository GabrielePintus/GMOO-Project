python3 script-BO.py \
    --setup_file setups/setup-mnist-mlp-BO.py \
    --dataset datasets/MNIST \
    --warmup_steps 10 \
    --optimization_steps 90 \
    --batch_size 1024 \
    --n_epochs 100 \
    --tags mnist GA plateau MLP

