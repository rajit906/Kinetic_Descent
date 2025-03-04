# TODO: Minimum/Final validation loss for model checkpoints and results for HPO. 
# Save initial untrained model to get same starting points for all optimizers.
# Check if optimal curves for Adam and mSGD are same. Then save. Run HPO on KD once finalized. Compare to ADAM and mSGD.

import argparse
import os
import json
import random
import ssl
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from loguru import logger

from kd import KineticDescentUB
from train import train, evaluate
from data import load_data
import torchvision.models as models

# Allow unverified HTTPS context
ssl._create_default_https_context = ssl._create_unverified_context


def run_FMNIST(
    optimizer_type: str,
    args: argparse.Namespace,
    seed: int,
    device: str
):
    """
    Train a model on the FashionMNIST dataset using a specified optimizer.

    Parameters:
        optimizer_type (str): Optimizer to use ('KD', 'SGD', or 'ADAM').
        args (argparse.Namespace): Arguments containing hyperparameters and settings.
        seed (int): Random seed for reproducibility.
        device (str): Device to use for training ('cpu' or 'cuda').
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Unpack arguments
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    
    # Initialize model, criterion, and optimizer
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(dtype=torch.float64)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Using optimizer: {optimizer_type}")
    if optimizer_type == 'KD':
        optimizer = KineticDescentUB(
            model.parameters(), lr=lr, gamma=args.gamma, c_init=args.c_init
        )
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=args.momentum
        )
    elif optimizer_type == 'ADAM':
        optimizer = optim.Adam(
            model.parameters(), lr=lr, betas=args.betas
        )
    else:
        raise ValueError("Invalid optimizer type")

    # Load data
    train_loader, val_loader, test_loader = load_data(
        batch_size, dataset_name="FashionMNIST"
    )

    # Train the model
    scheduler = None
    model_trained, train_loss_values, val_loss_values, train_acc_values, test_acc_values = train(
        model, optimizer, criterion, train_loader, val_loader, test_loader,
        num_epochs=num_epochs, scheduler=scheduler, flatten=False
    )

    # Evaluate the model
    train_accuracy, test_accuracy = evaluate(
        model_trained, test_loader, train_loader, flatten=False
    )
    logger.info(f"Final Train Acc: {train_accuracy}, Final Test Acc: {test_accuracy}")

    # Save results and model checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model_trained.state_dict(), os.path.join(output_dir, f"model_{optimizer_type}.pth"))

    results = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_loss_values": train_loss_values,
        "val_loss_values": val_loss_values,
        "train_acc_values": train_acc_values,
        "test_acc_values": test_acc_values
    }

    if optimizer_type == 'KD':
        ke = [optimizer.momentum_magnitude_history[i] for i in range(num_epochs * int(50000 // batch_size))]
        t = [optimizer.t[i] for i in range(num_epochs * int(50000 // batch_size))]
        results["kinetic_energy"] = ke
        results["time"] = t

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results


def hyperparameter_optimization(objective_fn: callable, n_trials: int):
    """
    Perform hyperparameter optimization using Optuna.

    Parameters:
        objective_fn (callable): Objective function for Optuna to optimize.
        n_trials (int): Number of trials to run.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=n_trials)

    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_trial.value}")
    logger.info(f"  Params: {study.best_trial.params}")

    return study


def objective(trial):
    """
    Optuna objective function to optimize hyperparameters for KD optimizer.

    Parameters:
        trial (optuna.Trial): Optuna trial object.

    Returns:
        float: Validation accuracy.
    """
    args = argparse.Namespace(
        num_epochs=40,
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        lr=trial.suggest_loguniform("lr", 1e-4, 1e-1),
        gamma=trial.suggest_loguniform("gamma", 1e-2, 1.0),
        c_init=trial.suggest_uniform("c_init", 0.1, 1.0),
        seed=42
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_FMNIST("KD", args, args.seed, device)

    return results["val_loss_values"][-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with different optimizers.")
    parser.add_argument("--optimizer", type=str, choices=["KD", "SGD", "ADAM"], required=True, help="Optimizer to use.")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for KD optimizer.")
    parser.add_argument("--c_init", type=float, default=0.5, help="C_init for KD optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="Betas for Adam optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--HPO", action="store_true", help="Run hyperparameter optimization.")
    parser.add_argument("--n_trials", type=int, default=200, help="Number of trials for Optuna.")

    args = parser.parse_args()

    if args.HPO:
        hyperparameter_optimization(objective, args.n_trials)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_FMNIST(args.optimizer, args, args.seed, device)
