import torch.nn as nn
import torchvision.models as models


def resnet(args):

    if args.model == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained, progress=True)
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=args.pretrained, progress=True)
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained, progress=True)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, args.n_cls)

    return model
