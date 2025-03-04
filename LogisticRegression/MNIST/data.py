import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def load_data(batch_size, dataset_name="FashionMNIST", val_size=7000, root='./data'):
    # Define transformations based on dataset
    if dataset_name == "MNIST" or dataset_name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR typically uses 3 channels
        ])
    elif dataset_name == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Standard ImageNet normalization
        ])
    else:
        raise ValueError("Dataset not supported. Choose from: MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageNet.")

    # Load the specified dataset
    if dataset_name == "MNIST":
        dataset = datasets.MNIST
    elif dataset_name == "FashionMNIST":
        dataset = datasets.FashionMNIST
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10
    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100
    elif dataset_name == "ImageNet":
        dataset = datasets.ImageNet
    else:
        raise ValueError("Dataset not supported. Choose from: MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageNet.")
    
    # Load the train and test datasets
    train_dataset = dataset(root=root, train=True, download=True, transform=transform)
    test_dataset = dataset(root=root, train=False, download=True, transform=transform)

    # Split training dataset into train and validation sets if specified
    if val_size > 0:
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        train_subset = train_dataset
        val_loader = None

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
