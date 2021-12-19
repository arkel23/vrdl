import os
import torch

from fgvr.models import resnet, VisionTransformer


def build_model(args):
    if 'resnet' in args.model:
        m = resnet(args)
    else:
        m = VisionTransformer(args)

    print('Loading pt={} {} model with {} classes output head'.format(
        args.pretrained, args.model, args.n_cls))

    m = m.to(args.device)
    return m


def get_model_name(path_model):
    """parse model name"""
    segments = path_model.split('/')[-2].split('_')
    if 'H' in segments or 'B' in segments or 'L' in segments:
        return segments[0] + '_' + segments[1]
    else:
        return segments[0]


def load_model_inference(args):

    model = build_model(args)

    if args.path_checkpoint:
        print('==> loading model backbone')
        state_dict = torch.load(args.path_checkpoint)['model']

        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys when loading pretrained weights: {}'.format(
            ret.missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(
            ret.unexpected_keys))
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
            save_file = os.path.join(
                args.save_folder, '{}_best.pth'.format(args.model))
        else:
            save_file = os.path.join(
                args.save_folder, '{}_best.pth'.format(args.model_s))
        print('Saving the best model!')
        torch.save(state, save_file)
    elif mode == 'epoch':
        save_file = os.path.join(
            args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        print('==> Saving each {} epochs...'.format(args.save_freq))
        torch.save(state, save_file)
    elif mode == 'last':
        if vanilla:
            save_file = os.path.join(
                args.save_folder, '{}_last.pth'.format(args.model))
        else:
            save_file = os.path.join(
                args.save_folder, '{}_last.pth'.format(args.model_s))
        print('Saving last epoch')
        torch.save(state, save_file)
