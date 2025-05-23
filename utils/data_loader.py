import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(train_path='data/raw/C-NMC_Leukemia/training_data', test_path='data/raw/C-NMC_Leukemia/testing_data', batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(train_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader
