import torch.utils.data as data

from .build_transform import build_transform
from .fish import Fish


def build_dataloaders(args, vanilla=True):

    train_transform = build_transform(split='train', args=args)
    val_transform = build_transform(split='val', args=args)

    train_set = get_train_set(
        args.dataset_path, train_transform)
    val_set = get_val_set(args.dataset_path, val_transform)
    n_data = len(train_set)
    args.n_cls = train_set.num_classes

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True)
    if not args.skip_eval:
        val_loader = data.DataLoader(val_set, batch_size=args.batch_size//2,
                                     shuffle=False,
                                     num_workers=int(args.num_workers/2),
                                     pin_memory=True)
    else:
        val_loader = None

    if vanilla:
        return train_loader, val_loader
    return train_loader, val_loader, n_data


def get_val_set(dataset_path, transform):
    val_set = Fish(root=dataset_path, train=False, transform=transform)
    return val_set


def get_train_set(dataset_path, transform):
    train_set = Fish(root=dataset_path, train=True, transform=transform)
    return train_set
