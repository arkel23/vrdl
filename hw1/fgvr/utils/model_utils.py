import os
import torch

from fgvr.models import model_extractor


def get_model_name(path_model):
    """parse model name"""
    segments = path_model.split('/')[-2].split('_')
    return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_model(path_model, n_cls, layers):
    print('==> loading model')
    model_name = get_model_name(path_model)
    model = model_extractor(model_name, num_classes=n_cls, layers=layers)

    state_dict = torch.load(path_model)['model']
    for key in list(state_dict.keys())[-2:]:
        state_dict.pop(key)

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
