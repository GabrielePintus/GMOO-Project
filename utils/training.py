import torch
from tqdm import tqdm



def train(
    model,
    device,
    train_loader,
    val_loader,
    n_epochs,
    optimizer,
    optimizer_params,
    criterion,
    lr_scheduler,
    lr_scheduler_params,
    clip_grad=1.0,
    progress_bar=False
):
    # Class to object
    optimizer = optimizer(model.parameters(), **optimizer_params)
    lr_scheduler = lr_scheduler(optimizer, **lr_scheduler_params)

    train_losses = []
    val_losses = []
    lr_evolution = [optimizer_params['lr']]

    progress_bar = tqdm(range(n_epochs)) if progress_bar else range(n_epochs)

    for epoch in progress_bar:
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        lr_scheduler.step()
        lr_evolution.append(optimizer.param_groups[0]['lr'])

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
            val_losses.append(val_loss / len(val_loader))

    return train_losses, val_losses, lr_evolution
        
