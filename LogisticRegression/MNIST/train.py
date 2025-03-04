import numpy as np
import torch
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def predict(model, loader, flatten):
    targets, predictions = [], []
    for batch_inputs, batch_targets in loader:
        batch_inputs = batch_inputs.to(dtype=torch.float64)
        if flatten:
            batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)
        else:
            batch_inputs = batch_inputs.view(batch_inputs.size(0), 1, 28, 28)
        outputs = model(batch_inputs)
        _, predicted = torch.max(outputs, 1)
        targets.extend(batch_targets.cpu().numpy())
        predictions.extend(predicted.cpu().numpy())
    return targets, predictions

def evaluate(model, test_loader, train_loader, flatten):
    model.eval()
    with torch.no_grad():
        test_targets, test_predictions = predict(model, test_loader, flatten)
        train_targets, train_predictions = predict(model, train_loader, flatten)
    train_accuracy = 100. * (sum(1 for x, y in zip(train_predictions, train_targets) if x == y) / len(train_targets))
    test_accuracy = 100. * (sum(1 for x, y in zip(test_predictions, test_targets) if x == y) / len(test_targets))
    return train_accuracy, test_accuracy

def train(model, optimizer, criterion, train_loader, val_loader, test_loader, num_epochs, scheduler, flatten):
    train_loss_values = []
    val_loss_values = []
    train_acc_values = []  # To store train accuracies
    test_acc_values = []  # To store test accuracies

    # Zeroth epoch evaluation (before training begins)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Calculate zeroth epoch training loss
        train_epoch_loss = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(dtype=torch.float64)
            batch_targets = batch_targets.to(dtype=torch.long)

            if flatten:
                batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)
            else:
                batch_inputs = batch_inputs.view(batch_inputs.size(0), 1, 28, 28)

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            train_epoch_loss += loss.item()

        train_loss_values.append(train_epoch_loss / len(train_loader))

        # Calculate zeroth epoch validation loss
        val_epoch_loss = 0
        for val_inputs, val_targets in val_loader:
            val_inputs = val_inputs.to(dtype=torch.float64)
            val_targets = val_targets.to(dtype=torch.long)

            if flatten:
                val_inputs = val_inputs.view(val_inputs.size(0), -1)
            else:
                val_inputs = val_inputs.view(val_inputs.size(0), 1, 28, 28)

            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            val_epoch_loss += val_loss.item()

        val_loss_values.append(val_epoch_loss / len(val_loader))

        # Calculate zeroth epoch accuracy
        train_accuracy, test_accuracy = evaluate(model, test_loader, train_loader, flatten)
        train_acc_values.append(train_accuracy)
        test_acc_values.append(test_accuracy)

    # Training phase
    for _ in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(dtype=torch.float64)
            batch_targets = batch_targets.to(dtype=torch.long)

            if flatten:
                batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)
            else:
                batch_inputs = batch_inputs.view(batch_inputs.size(0), 1, 28, 28)

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss_values.append(epoch_loss / len(train_loader))

        # Calculate train and validation accuracies
        train_accuracy, test_accuracy = evaluate(model, test_loader, train_loader, flatten)
        train_acc_values.append(train_accuracy)
        test_acc_values.append(test_accuracy)

        # Validation loss
        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(dtype=torch.float64)
                val_targets = val_targets.to(dtype=torch.long)

                if flatten:
                    val_inputs = val_inputs.view(val_inputs.size(0), -1)
                else:
                    val_inputs = val_inputs.view(val_inputs.size(0), 1, 28, 28)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_epoch_loss += val_loss.item()
        val_loss_values.append(val_epoch_loss / len(val_loader))
        if scheduler:
            scheduler.step(val_loss)

    return model, train_loss_values, val_loss_values, train_acc_values, test_acc_values
