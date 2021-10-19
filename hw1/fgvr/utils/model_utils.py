import os
import torch

from fgvr.models import model_extractor


def get_model_name(path_model):
    """parse model name"""
    segments = path_model.split('/')[-2].split('_')
    if segments[0] == 'linear' or segments[0] == 'student':
        return segments [1]
    else:
        return segments[0]


def load_model_head(path_model, n_cls, pretrained, layers):
    print('==> loading model with classification head')
    
    model_name = get_model_name(path_model)
    model = model_extractor(model_name, num_classes=n_cls, 
                            pretrained=pretrained, layers=layers)
    
    model.load_state_dict(torch.load(path_model)['model'], strict=True)
    print('==> done')
    
    return model


def load_model_nohead(path_model, model_name, n_cls, pretrained, layers):
    if path_model:
        model_name = get_model_name(path_model)
    else:
        model_name = model_name

    model = model_extractor(model_name, num_classes=n_cls, 
                            pretrained=pretrained, layers=layers)

    if path_model:
        print('==> loading model without classification head')
        state_dict = torch.load(path_model)['model']
        for key in list(state_dict.keys())[-2:]:
            state_dict.pop(key)

        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys))
        print('==> done')
    
    return model


def load_model_inference(
    path_backbone, model_name, n_cls, pretrained, layers):
    if path_backbone:
        model_name = get_model_name(path_backbone)
    else:
        model_name = model_name

    model = model_extractor(model_name, num_classes=n_cls, 
                            pretrained=pretrained, layers=layers)

    if path_backbone:
        print('==> loading model backbone')
        state_dict = torch.load(path_backbone)['model']

        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys))
        print('==> done')
    
    return model


def save_model(args, model, epoch, acc, mode, optimizer=False, vanilla=True):
    if optimizer:
        state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': acc,
                'optimizer': optimizer.state_dict(),
            }
    else:
        state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': acc,
            }

    if mode == 'best':
        if vanilla:
            save_file = os.path.join(args.save_folder, '{}_best.pth'.format(args.model))
        else:
            save_file = os.path.join(args.save_folder, '{}_best.pth'.format(args.model_s))
        print('Saving the best model!')
        torch.save(state, save_file)
    elif mode == 'epoch':
        save_file = os.path.join(args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        print('==> Saving each {} epochs...'.format(args.save_freq))
        torch.save(state, save_file)
    elif mode == 'last':
        if vanilla:
            save_file = os.path.join(args.save_folder, '{}_last.pth'.format(args.model))
        else:
            save_file = os.path.join(args.save_folder, '{}_last.pth'.format(args.model_s))
        print('Saving last epoch')
        torch.save(state, save_file)
