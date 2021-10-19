import os
import argparse
import torch

from .model_utils import get_model_name


def parse_common():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--dataset_path', type=str, default='./data/', help='path to download/read datasets')
    parser.add_argument('--image_size', type=int, default=448, help='image_size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')

    # optimization
    parser.add_argument('--opt', default='sgd', type=str, help='Optimizer (default: "sgd"')
    parser.add_argument('--base_lr', type=float, default=0.004, help='base learning rate to scale based on batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, help='Clip gradient norm (default: None, no clipping)')
    # scheduler
    parser.add_argument('--sched', default='cosine', type=str, choices=['cosine', 'step'],
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--decay_epochs', type=float, default=30, help='epoch interval to decay LR')

    # distributed
    parser.add_argument('--dist_eval', action='store_true', 
                        help='validate using dist sampler (else do it on one gpu)')

    # others 
    parser.add_argument('--deit_recipe', action='store_true', help='use deit augs')
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model on imagenet')
    
    return parser


def add_adjust_common_dependent(args):

    args.lr = args.base_lr * (args.batch_size / 256)

    # distributed
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

        args.lr = args.base_lr * ((args.world_size * args.batch_size) / 256)

    return args


def parse_option_vanilla():

    parser = parse_common()
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34' ,'resnet50'])
    args = parser.parse_args()

    args = add_adjust_common_dependent(args)

    args.model_name = '{}_is{}_bs{}_blr{}decay{}_pt{}_trial{}'.format(
        args.model, args.image_size, args.batch_size, args.base_lr, 
        args.weight_decay, args.pretrained, args.trial)

    args.save_folder = os.path.join('save', 'models', args.model_name)
    os.makedirs(args.save_folder, exist_ok=True)

    print(args)
    return args


def parse_option_inference():

    parser = parse_common()
    parser.add_argument('--model', type=str, default=None,
                        choices=[None, 'resnet18', 'resnet34' ,'resnet50'])
    parser.add_argument('--path_backbone', type=str, default=None, help='backbone ckpt')
    parser.add_argument('--path_classifier', type=str, default=None, help='classifier ckpt')
    args = parser.parse_args()

    if not args.model:
        args.model = get_model_name(args.path_backbone)
    args = add_adjust_common_dependent(args)

    print(args)
    return args


def parse_option_linear():
    
    parser = parse_common()
    parser.add_argument('--model', type=str, default=None,
                        choices=[None, 'resnet18', 'resnet34' ,'resnet50'])
    parser.add_argument('--path_model', type=str, default=None, help='model snapshot')
    parser.set_defaults(epochs=100, base_lr=0.4, sched='cosine')
    args = parser.parse_args()

    if not args.model:
        args.model = get_model_name(args.path_model)
    args = add_adjust_common_dependent(args)

    args.model_name = 'linear_{}_is{}_bs{}_blr{}decay{}_pt{}_trial_{}'.format(
        args.model, args.image_size, args.batch_size, args.base_lr, 
        args.weight_decay, args.pretrained, args.trial)

    args.save_folder = os.path.join('save', 'linear', args.model_name)
    os.makedirs(args.save_folder, exist_ok=True)

    print(args)
    return args


def parse_option_student():
    
    parser = parse_common()
    # model
    parser.add_argument('--model_s', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34' ,'resnet50'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['ifacrd', 'kd', 'crd'])    
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    
    # IFACRD distillation
    parser.add_argument('--layers', type=str, default='last', choices=['all', 'blocks', 'last'], 
                        help='features from last layers or blocks ends')
    parser.add_argument('--cont_no_l', default=2, type=int, 
                        help='no of layers from teacher to use to build contrastive batch')
    
    parser.add_argument('--rs_no_l', default=1, choices=[1, 2, 3], type=int, 
                        help='no of layers for rescaler mlp')
    parser.add_argument('--rs_hid_dim', default=128, type=int, 
                        help='dimension of rescaler mlp hidden layer space')
    parser.add_argument('--rs_ln', action='store_true', help='Use rescaler mlp with LN instead of BN')
    
    parser.add_argument('--proj_no_l', default=1, choices=[1, 2, 3], type=int, 
                        help='no of layers for projector mlp')
    parser.add_argument('--proj_hid_dim', default=128, type=int, 
                        help='dimension of projector mlp hidden layer space')
    parser.add_argument('--proj_ln', action='store_true', help='Use projector mlp with LN instead of BN')
    
    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])
    
    args = parser.parse_args()

    args.model_t = get_model_name(args.path_t)
    args = add_adjust_common_dependent(args)
    # set layers argument to blocks when using any method that is not ifacrd
    if args.distill != 'ifacrd':
        if args.distill == 'abound':
            args.layers = 'preact'
        else:
            args.layers = 'default'
        args.cont_no_l = 0

    if args.distill == 'ifacrd':
        args.model_name = 'S{}_T{}_{}_{}_r{}_a{}_b{}_bs{}_blr{}wd{}_temp{}_contl{}{}_rsl{}hd{}ln{}_pjl{}out{}hd{}ln{}_{}'.format(
            args.model_s, args.model_t, args.dataset, args.distill, args.gamma, args.alpha, args.beta, args.batch_size, 
            args.base_lr, args.weight_decay, args.nce_t, args.cont_no_l, args.layers, args.rs_no_l, args.rs_hid_dim, args.rs_ln, 
            args.proj_no_l, args.feat_dim, args.proj_hid_dim, args.proj_ln, args.trial)
    else:
        args.model_name = 'S{}_T{}_{}_{}_r{}_a{}_b{}_bs{}_blr{}wd{}_temp{}_{}'.format(
            args.model_s, args.model_t, args.dataset, args.distill, args.gamma, args.alpha, args.beta, args.batch_size,
            args.base_lr, args.weight_decay, args.nce_t, args.trial)

    args.save_folder = os.path.join('save', 'student', args.model_name)
    os.makedirs(args.save_folder, exist_ok=True)

    print(args)
    return args
