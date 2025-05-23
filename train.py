import os
import torch
import torch.nn as nn
from torchvision import models
from utils.data_loader import get_data_loaders

# Seleciona o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

_, test_loader = get_data_loaders(
    train_path='data/raw/C-NMC_Leukemia/training_data',
    test_path='data/raw/C-NMC_Leukemia/testing_data'
)

# Descobrir número de classes a partir dos dados
num_classes = len(test_loader.dataset.classes)
print(f"Número de classes no dataset: {num_classes}")

# Criar a estrutura do modelo e ajustar camada final
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Carregar os pesos salvos
model.load_state_dict(torch.load('models/leukemia_model.pth', map_location=device))
model.eval()

# Avaliação
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Mostrar a acurácia
print(f"🔍 Acurácia no conjunto de teste: {100 * correct / total:.2f}%")