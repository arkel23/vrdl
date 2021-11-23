import torch.nn as nn
import torchvision.models as models

from .util import freeze_layers


def resnet(model_name, num_classes, freeze, pretrained):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained, progress=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained, progress=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained, progress=True)
    
    if freeze:
        freeze_layers(model)
            
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    print('Loading pt={} {} model with {} classes output head'.format(
        pretrained, model_name, num_classes
    ))

    return model