
import torch.nn as nn
import torchvision.models.feature_extraction as feature_extraction
from einops.layers.torch import Rearrange

from .resnet import resnet
from .vit import vit


def model_extractor(
    model_name, num_classes, image_size, 
    freeze=False, pretrained=False, layers='default'):
    if 'resnet' in model_name:
        m = resnet(model_name, num_classes, freeze, pretrained)
        model = Extractor(m, model_name, layers)
    else:
        model = vit(model_name, num_classes, image_size, pretrained)
    return model


class Extractor(nn.Module):
    def __init__(self, model, model_name, layers='default'):
        super(Extractor, self).__init__()
        self.model_name = model_name
        return_nodes = self.get_return_nodes(model, model_name, layers)
        self.model = feature_extraction.create_feature_extractor(
            model, return_nodes=return_nodes)
        
        if layers not in ['default', 'last_only']:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), Rearrange('b c 1 1 -> b c'))
  
    def forward(self, x, classify_only=True):
        x = list(self.model(x).values())
        if classify_only:
            return x[-1]
        else:
            if hasattr(self, 'pool'):
                return [self.pool(feats) for feats in x[:-1]] + [x[-1]]  
            else:
                return x
            
    def get_return_nodes(self, model, model_name, layers):
        # train_nodes, eval_nodes = feature_extraction.get_graph_node_names(model)
        if layers == 'last':
            if model_name == 'resnet18':
                return_nodes =  {
                    'layer4.0.relu': 'layerminus4',
                    'layer4.0.relu_1': 'layerminus3',
                    'layer4.1.relu': 'layerminus2',
                    'layer4.1.relu_1': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet34':
                return_nodes =  {
                    'layer4.0.relu': 'layerminus6',
                    'layer4.0.relu_1': 'layerminus5',
                    'layer4.1.relu': 'layerminus4',
                    'layer4.1.relu_1': 'layerminus3',
                    'layer4.2.relu': 'layerminus2',
                    'layer4.2.relu_1': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet50':
                return_nodes =  {
                    'layer4.0.relu': 'layerminus9',
                    'layer4.0.relu_1': 'layerminus8',
                    'layer4.0.relu_2': 'layerminus7',
                    'layer4.1.relu': 'layerminus6',
                    'layer4.1.relu_1': 'layerminus5',
                    'layer4.1.relu_2': 'layerminus4',
                    'layer4.2.relu': 'layerminus3',
                    'layer4.2.relu_1': 'layerminus2',
                    'layer4.2.relu_2': 'layerminus1',
                    'fc': 'layerminus0'
                }
            else:
                raise NotImplementedError
        
        elif layers == 'blocks':
            if model_name == 'resnet18':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.1.relu_1': 'layerminus5',
                    'layer2.1.relu_1': 'layerminus4',
                    'layer3.1.relu_1': 'layerminus3',
                    'layer4.1.relu_1': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet34':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.relu_1': 'layerminus5',
                    'layer2.3.relu_1': 'layerminus4',
                    'layer3.5.relu_1': 'layerminus3',
                    'layer4.2.relu_1': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet50':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.relu_2': 'layerminus5',
                    'layer2.3.relu_2': 'layerminus4',
                    'layer3.5.relu_2': 'layerminus3',
                    'layer4.2.relu_2': 'layerminus2',
                    'fc': 'layerminus0'
                }
            else:
                raise NotImplementedError
                            
        elif layers == 'all':
            train_nodes, eval_nodes = feature_extraction.get_graph_node_names(model)
            if model_name in ['resnet18', 'resnet34', 'resnet50']:
                return_nodes = {node:i for i, node in enumerate(train_nodes) if 'relu' in train_nodes}
                return_nodes['fc'] = 'layerminus0' 
            else:
                raise NotImplementedError
        
        elif layers == 'default':
            if model_name == 'resnet18':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.1.relu_4': 'layerminus5',
                    'layer2.1.relu_8': 'layerminus4',
                    'layer3.1.relu_12': 'layerminus3',
                    'layer4.1.relu_16': 'layerminus2',
                    'flatten': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet34':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.relu_6': 'layerminus5',
                    'layer2.3.relu_14': 'layerminus4',
                    'layer3.5.relu_26': 'layerminus3',
                    'layer4.2.relu_34': 'layerminus2',
                    'flatten': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet50':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.relu_2': 'layerminus5',
                    'layer2.3.relu_2': 'layerminus4',
                    'layer3.5.relu_2': 'layerminus3',
                    'layer4.2.relu_2': 'layerminus2',
                    'flatten': 'layerminus1',
                    'fc': 'layerminus0'
                }

            else:
                raise NotImplementedError
            
        elif layers == 'last_only':
            if model_name in ['resnet18', 'resnet34', 'resnet50']:
                return_nodes = {'fc': 'layerminus0'}            
            else:
                raise NotImplementedError
        
        else:
            raise NotImplementedError      
        
        return return_nodes
