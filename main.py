# Machine Learning
import lightning as L
from lightning.pytorch.loggers import NeptuneLogger, WandbLogger
import neptune
import wandb


# Custom modules
from libraries.models import LitMLP
from setup import *

# Miscellaneous
from math import ceil

NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMjM0NTVmMi0zNzg3LTQyMmQtOGI5NS0xYjNkMjk0MGUyYTAifQ=="
WANDB_API_TOKEN = "ad41ea382b5d95333648eff0dac021aab7e66863"

if __name__ == '__main__':
    # Load the data
    df = pd.read_parquet('datasets/creditcard_downsampled.parquet')
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=SEED)
    X_train, y_train = df_train.drop(columns=['Class']).values, df_train['Class'].values.reshape(-1, 1)
    X_val, y_val = df_val.drop(columns=['Class']).values, df_val['Class'].values.reshape(-1, 1)
    X_test, y_test = df_test.drop(columns=['Class']).values, df_test['Class'].values.reshape(-1, 1)

    # Build the datasets
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    # Create the data loaders
    BATCH_SIZE = 128
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Tensor cores
    torch.set_float32_matmul_precision('high')

    # Create the model
    model = MLP(
        input_size=X_train.shape[1],
        hidden_size=16,
        output_size=y_train.shape[1],
        activation=0,
        dropout=0.0
    )

    # Define useful callbacks
    # early_stopping = L.pytorch.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=5,
    #     mode='min'
    # )
    # lr_logger = L.pytorch.callbacks.LearningRateMonitor(
    #     logging_interval='epoch'
    # )
    # model_checkpoint = ModelCheckpoint(
    #     dirpath="my_model/checkpoints/",
    #     filename="{epoch:02d}",
    #     save_weights_only=True,
    #     save_top_k=2,
    #     save_last=True,
    #     monitor="val/loss",
    #     every_n_epochs=1,
    # )

    # Prepare logger
    wandb.login(key=WANDB_API_TOKEN)
    wandb_logger = WandbLogger(
        project="GMOOP",
    )

    steps_per_epoch = ceil(len(train_loader) // 128)

    # Train the model
    trainer = L.Trainer(
        max_epochs=150,
        enable_checkpointing=False,
        callbacks=[lr_logger, early_stopping],
        gradient_clip_val=1.0,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        # log_every_n_steps=steps_per_epoch
    )

    # neptune_logger.log_model_summary(model, max_depth=-1)
    # neptune_logger.log_hyperparams(model.hparams)
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model
    trainer.test(model, val_loader)
