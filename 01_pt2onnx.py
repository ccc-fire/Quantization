import torch
import torch.nn as nn
# from model.resnet50 import base_resnet as base_model
import torchvision.models as models

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights = pretrained)
        # Replace the last fully connected layer to match your task
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def model2onnx(checkpoint_path, save_path, num_classes=1000):

    model = ResNet50(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224).to(device)  # 这里就需要训练模型的真实输入
    torch.onnx.export(model, input_tensor, save_path, do_constant_folding=False)
    print(f"Model has been successfully converted to ONNX and saved at {save_path}")

def load_resnet50_weights(model, weights_path):
    pretrained_dict = torch.load(weights_path, map_location='cpu')
    # 将 map_location 设置为 'cpu'，以确保加载的权重首先存储在 CPU 上。
    # 这样可以避免设备不匹配的问题。
    model_dict = model.state_dict()

    # Remove 'module.' prefix if it was added due to DataParallel
    if list(pretrained_dict.keys())[0].startswith('module.'):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}

    # Load pretrained weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # # 仅加载匹配的参数, 以应对形状不匹配的情况
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

if __name__ == '__main__':
    # Define your ResNet-50 model
    model = ResNet50(num_classes=1000)  # Change num_classes according to your task

    # Load downloaded weights into the model
    weights_path = 'model/resnet50-0676ba61.pth'
    load_resnet50_weights(model, weights_path)

    # Optionally, save the model
    torch.save(model.state_dict(), weights_path)

    model2onnx(weights_path, 'resnet50.onnx', num_classes=1000)

