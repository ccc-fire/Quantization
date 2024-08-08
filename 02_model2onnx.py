import torch
import torch.nn as nn
import torchvision.models as models

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        # Replace the last fully connected layer to match your task
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def model2onnx(model, save_path):
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # 确保输入张量在相同的设备上
    torch.onnx.export(model, input_tensor, save_path, do_constant_folding=False)
    print(f"Model has been successfully converted to ONNX and saved at {save_path}")

if __name__ == '__main__':
    num_classes = 1000  # Change num_classes according to your task

    # Define your ResNet-50 model with pretrained weights
    model = ResNet50(num_classes=num_classes, pretrained=True).to(device)

    # Optionally, save the model weights
    weights_path = 'model/resnet50-0676ba61.pth'
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")

    # Convert the model to ONNX format
    onnx_path = 'resnet50.onnx'
    try:
        model2onnx(model, onnx_path)
        print("Model converted to ONNX format successfully.")
    except Exception as e:
        print(f"Error converting model to ONNX: {e}")
