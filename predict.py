from PIL import Image
from torchvision import transforms, models
import torch

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('models/leukemia_model.pth'))
    model.eval()

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return 'Leukemia' if predicted.item() == 1 else 'Normal'

if __name__ == "__main__":
    print(predict("data/raw/test/Leukemia/image1.jpg"))