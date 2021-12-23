import os
import argparse
import torch

from .model_utils import get_model_name, get_ifa_tkgather_freeze


def add_adjust_common_dependent(args):
    args.lr = args.base_lr * (args.batch_size / 256)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not args.ifa:
        args.token_gather = 'cls'

    return args


def parse_common():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--print_freq', type=int,
                        default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int,
                        default=20, help='save frequency')
    parser.add_argument('--dataset_path', type=str, default='./data/',
                        help='path to download/read datasets')
    parser.add_argument('--image_size', type=int,
                        default=448, help='image_size')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch_size')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--opt', default='sgd', type=str,
                        help='Optimizer (def "sgd"')
    parser.add_argument('--base_lr', type=float, default=0.004,
                        help='base learning rate to scale based on batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm (def None, no clipping)')
    # scheduler
    parser.add_argument('--sched', default='cosine', type=str,
                        choices=['cosine'], help='LR scheduler (def "cosine"')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    parser.add_argument('--decay_epochs', type=float,
                        default=30, help='epoch interval to decay LR')

    # others
    parser.add_argument(
        '--deit_recipe', action='store_true', help='use deit augs')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained model on imagenet')
    parser.add_argument('--skip_eval', action='store_true', help='skip eval')
    parser.add_argument('--freeze', action='store_true', help='freeze back')
    parser.add_argument('--ifa', action='store_true', help='IFA cls head')
    parser.add_argument('--token_gather', type=str, default='cls',
                        choices=['cls', 'gappre', 'gappost'],
                        help='Use cls token, pool before or pool after')

    return parser


def parse_option_train():

    parser = parse_common()
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'B_16', 'B_32', 'L_16', 'H_14'])
    args = parser.parse_args()

    args = add_adjust_common_dependent(args)

    args.run_name = '{}_{}_{}_{}_is{}_bs{}_blr{}decay{}_pt{}_val{}'.format(
        args.model, args.ifa, args.token_gather, args.freeze,
        args.image_size, args.batch_size, args.base_lr, args.weight_decay,
        args.pretrained, not(args.skip_eval))

    args.save_folder = os.path.join('save', 'models', args.run_name)
    os.makedirs(args.save_folder, exist_ok=True)

    print(args)
    return args


def parse_option_inference():

    parser = parse_common()
    parser.add_argument('--path_checkpoint', type=str,
                        default=None, help='path ckpt')
    parser.set_defaults(print_freq=1000)
    args = parser.parse_args()

    assert args.path_checkpoint, 'Requires checkpoint to load model.'
    args.model = get_model_name(args.path_checkpoint)
    args.ifa, args.token_gather, args.freeze = get_ifa_tkgather_freeze(
        args.path_checkpoint)
    args = add_adjust_common_dependent(args)

    print(args)
    return args
