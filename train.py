import os
import torch
import torch.nn as nn
from torchvision import models
from utils.data_loader import get_data_loaders

# Garante que a pasta models exista
os.makedirs("models", exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Dataset (agora considerando o novo caminho adaptado)
train_loader, test_loader = get_data_loaders(
    train_path='data/raw/C-NMC_Leukemia',
    test_path='data/raw/test'
)

# Treinamento
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'models/leukemia_model.pth')
print("✅ Modelo salvo em models/leukemia_model.pth")
