import torch.utils.data as data

from .build_transform import build_transform
from .birds import Birds, BirdsInstance, BirdsInstanceSample


def build_dataloaders(args, vanilla=True):
    
    train_transform = build_transform(split='train', args=args)
    val_transform = build_transform(split='val', args=args)
    
    train_set = get_train_set(args.dataset_path, train_transform, args, vanilla)
    val_set = get_val_set(args.dataset_path, val_transform)
    n_data = len(train_set)
    n_cls = train_set.num_classes
    
    train_sampler = None
    val_sampler = None            
    if args.distributed:
        train_sampler = data.distributed.DistributedSampler(train_set)
        if args.dist_eval:
            val_sampler = data.distributed.DistributedSampler(val_set)
        
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, 
                                   shuffle=(train_sampler is None), 
                                   num_workers=args.num_workers, 
                                   pin_memory=True, drop_last=True, 
                                   sampler=train_sampler)    
    if not args.skip_eval:
        val_loader = data.DataLoader(val_set, batch_size=args.batch_size//2, 
                                 shuffle=False, 
                                 num_workers=int(args.num_workers/2), 
                                 pin_memory=True, sampler=val_sampler)
    else:
        val_loader = None

    if vanilla:
        return train_loader, val_loader, n_cls
    return train_loader, val_loader, n_cls, n_data


def get_val_set(dataset_path, transform):
    
    val_set = Birds(root=dataset_path, train=False, transform=transform)

    return val_set


def get_train_set(dataset_path, transform, args, vanilla):
    
    if vanilla:
        train_set = Birds(root=dataset_path, train=True, transform=transform)
    else:
        if args.distill in ['crd']:
            train_set = BirdsInstanceSample(
                root=dataset_path, train=True, transform=transform,
                k=args.nce_k, mode=args.mode, is_sample=True, percent=1.0)        
        else:
            train_set = BirdsInstance(root=dataset_path, train=True,
                                      transform=transform)
        
    return train_set