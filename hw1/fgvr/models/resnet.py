import torch.nn as nn
import torchvision.models as models

def resnet(model_name, num_classes, pretrained):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained, progress=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained, progress=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained, progress=True)
            
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model