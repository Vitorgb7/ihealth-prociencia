from sklearn.metrics import classification_report
import torch
from torchvision import models
from utils.data_loader import get_data_loaders

test_loader = get_data_loaders()[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('models/leukemia_model.pth'))
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

print(classification_report(y_true, y_pred, target_names=['Normal', 'Leukemia']))
