import torch.nn as nn


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
        

class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(LinearClassifier, self).__init__()
        
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.classifier(x)