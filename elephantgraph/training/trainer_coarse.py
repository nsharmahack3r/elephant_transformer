import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


def train_coarse_model(model, train_dataset, val_dataset, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    node_embeddings = train_dataset.node_embeddings.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 128),
        shuffle=False,
        num_workers=0,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    epochs = config.get('epochs', 100)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    # Seasonal latent dictionary regularizer weights
    w_div  = config.get('latent_diversity_weight', 0.01)
    w_load = config.get('latent_load_balance_weight', 0.01)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        epoch_div  = 0.0
        epoch_load = 0.0
        for batch in train_loader:
            node_indices = batch['node_indices'].to(device)
            season       = batch['season'].to(device)
            behavior     = batch['behavior'].to(device)
            target       = batch['target_nodes'].to(device)

            logits, occupancy, aux = model(node_indices, node_embeddings,
                                           season, behavior, return_aux=True)
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            loss = loss + 0.1 * nn.functional.mse_loss(
                occupancy[:, :-1],
                batch['occupancy'][:, 1:].float().to(device)
            )
            # Seasonal latent dictionary regularizers
            loss = loss + w_div  * aux['diversity_loss']
            loss = loss + w_load * aux['load_balance_loss']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            epoch_div  += aux['diversity_loss'].item()
            epoch_load += aux['load_balance_loss'].item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                node_indices = batch['node_indices'].to(device)
                season       = batch['season'].to(device)
                behavior     = batch['behavior'].to(device)
                target       = batch['target_nodes'].to(device)
                logits, _    = model(node_indices, node_embeddings,
                                     season, behavior)
                loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        epoch_div  /= len(train_loader)
        epoch_load /= len(train_loader)
        scheduler.step()

        print(f"Coarse Epoch {epoch + 1:03d} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"Dict[div={epoch_div:.4f} load={epoch_load:.4f}]", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/coarse/')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss':    val_loss,
                'config':      config,
            }, os.path.join(checkpoint_dir, 'best_coarse_model.pt'))
            print(f"  Saved best coarse model (val_loss={val_loss:.5f})", flush=True)
