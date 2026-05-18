import os
import sys
import torch
import numpy as np
import random
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from torch.utils.data import DataLoader

from src.datasets.el_dataset import ELDataset
from src.utils.transforms import get_train_transforms
from src.utils.transforms import get_val_transforms
from src.models.unet_model import get_model
from src.losses.losses import loss_fn


# -----------------------
# Reproducibility
# -----------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Train function
# -----------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):

    model.train()

    total_loss = 0

    for imgs, masks in tqdm(loader):

        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            outputs = model(imgs)

            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        

    return total_loss / len(loader)


# -----------------------
# Validation function
# -----------------------

def validate(model, loader, loss_fn, device):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for imgs, masks in loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------
# Main Training Loop
# -----------------------
best_val = float("inf")
def main():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT = "data/raw"

    train_dataset = ELDataset(
        ROOT,
        split="train",
        transform=get_train_transforms()
    )

    val_dataset = ELDataset(
        ROOT,
        split="val",
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    model = get_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )



    scaler = torch.amp.GradScaler("cuda")

    epochs = 40

    best_val_loss = float("inf")

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler
        )

        val_loss = validate(
            model,
            val_loader,
            loss_fn,
            device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), "checkpoints/last_model.pth")

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth"
            )


if __name__ == "__main__":
    main()
    