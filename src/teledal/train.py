import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

from teledal.data_processing.predict_next.data import PredictNextData
from teledal.model import TopKLoss
from teledal.model.teledal import TeleDAL

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_set: Dataset,
    validation_set: Dataset,
    optimizer: Optimizer,
    criteria: TopKLoss = TopKLoss(),
    batch_size: int = 64,
    num_epochs: int = 25,
    scheduler: LRScheduler = None,
    device: torch.device = None,
    validate: bool = False,
) -> None:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model.to(device)
    logger.info("Starting training loop")
    logger.info(f"Using device: {device}")
    history: list[float] = []

    # TODO: track best model during training process
    # best_params = None
    # best_losses = torch.inf

    # Setting up dataloaders
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        logger.info(f"Currently at epoch: {epoch}/{num_epochs}")
        model.train()
        for X, y, _ in train_loader:
            X = X.to(device)
            y = y.to(device)
            model.zero_grad()
            outputs = model(X)
            loss = criteria(y, outputs)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        if not validate:
            continue
        validation_losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for X, y, _ in validation_loader:
                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                loss = criteria(y, outputs)
                validation_losses.append(loss.mean().item())

        history.append(np.mean(validation_losses))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from torch.utils.data import random_split

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--data-path", required=True, type=Path, help="Path to the preprocessed data"
    )
    parser.add_argument(
        "--k", required=True, type=int, help="Value of k for top K prediction"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size to use")
    parser.add_argument(
        "--num-epochs", type=int, default=25, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.5,
        help="Size of fraction to use for training",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=45, help="Cut or pad to this length"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use")

    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)

    model = TeleDAL(k=args.k, generator=generator)
    criteria = TopKLoss()
    optimizer = Adam(
        params=model.parameters(), lr=args.learning_rate, generator=generator
    )

    full_ds = PredictNextData(args.dataset_path)
    train_ds, val_ds = random_split(
        full_ds, lengths=[args.train_split, 1 - args.train_split], generator=generator
    )

    train_model(
        model,
        train_ds,
        val_ds,
        optimizer,
        criteria,
        args.batch_size,
        args.num_epochs,
    )
