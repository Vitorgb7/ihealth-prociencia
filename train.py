import os
import torch
import torch.nn as nn
from torchvision import models
from utils.data_loader import get_data_loaders

os.makedirs("models", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

train_loader, test_loader = get_data_loaders(
    train_path='data/raw/C-NMC_Leukemia/training_data',
    test_path='data/raw/C-NMC_Leukemia/testing_data'
)


num_classes = len(train_loader.dataset.classes)
print(f"Número de classes no dataset: {num_classes}")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print("Iniciando o treinamento...")

for epoch in range(10):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{10} começando...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # A cada 100 batches, imprima o progresso
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / (batch_idx + 1):.4f}")

    print(f"Epoch {epoch + 1} concluída, Loss média: {running_loss / len(train_loader):.4f}")

# Salvando o modelo treinado
torch.save(model.state_dict(), 'models/leukemia_model.pth')
print("✅ Modelo salvo em models/leukemia_model.pth")