import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


def train_fine_model(model, diffusion, train_dataset,
                     val_dataset, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 200)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        weight_decay=config.get('wd', 1e-4)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            t = torch.randint(0, diffusion.T,
                              (batch['gps'].shape[0],), device=device)

            x_noisy, true_noise = diffusion.q_sample(batch['gps'], t)

            pred_noise = model(
                x_noisy, t,
                speed=batch['speed'], accel=batch['accel'],
                turning=batch['turning'], bearing=batch['bearing'],
                persist=batch['persist'], step=batch['step'],
                ndvi=batch['ndvi'], evi=batch['evi'],
                lst=batch['lst'], elev=batch['elev'],
                slope=batch['slope'], water=batch['water'],
                behavior=batch['behavior'], season=batch['season'],
                time_of_day=batch['time_of_day'], lulc=batch['lulc'],
                human_settle=batch['human_settle'],
                move_type=batch['move_type']
            )

            loss = nn.functional.mse_loss(pred_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                t = torch.randint(0, diffusion.T,
                                  (batch['gps'].shape[0],), device=device)
                x_noisy, true_noise = diffusion.q_sample(batch['gps'], t)
                pred_noise = model(
                    x_noisy, t,
                    speed=batch['speed'], accel=batch['accel'],
                    turning=batch['turning'], bearing=batch['bearing'],
                    persist=batch['persist'], step=batch['step'],
                    ndvi=batch['ndvi'], evi=batch['evi'],
                    lst=batch['lst'], elev=batch['elev'],
                    slope=batch['slope'], water=batch['water'],
                    behavior=batch['behavior'], season=batch['season'],
                    time_of_day=batch['time_of_day'], lulc=batch['lulc'],
                    human_settle=batch['human_settle'],
                    move_type=batch['move_type']
                )
                val_loss += nn.functional.mse_loss(pred_noise, true_noise).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch + 1:03d} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = config.get('checkpoint_dir', 'checkpoints/fine/')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"  Saved best model (val_loss={val_loss:.5f})")
