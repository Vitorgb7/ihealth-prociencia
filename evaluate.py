from sklearn.metrics import classification_report
import torch
from torchvision import models
from utils.data_loader import get_data_loaders

_, test_loader = get_data_loaders(
    train_path='data/raw/C-NMC_Leukemia/training_data',
    test_path='data/raw/C-NMC_Leukemia/testing_data'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load('models/leukemia_model.pth', map_location=device))
model = model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))