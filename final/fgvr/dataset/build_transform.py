from torchvision import transforms
from timm.data import create_transform


def build_deit_transform(is_train, args):
    ''' taken from DeiT paper
    https://arxiv.org/abs/2012.12877
    https://github.com/facebookresearch/deit/blob/main/main.py'''
    # augmentation and random erase params
    args.color_jitter = 0.4
    args.aa = 'rand-m9-mstd0.5-inc1'
    args.smoothing = 0.1
    args.train_interpolation = 'bicubic'
    args.repeated_aug = True
    args.reprob = 0.25
    args.remode = 'pixel'
    args.recount = 1
    args.resplit = False

    resize_im = args.image_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.image_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.image_size, padding=4)
        return transform

    t = []
    if resize_im:
        # to maintain same ratio w.r.t. 224 images
        size = int((256 / 224) * args.image_size)
        t.append(transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(args.image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5]))
    return transforms.Compose(t)


def build_transform(args, split):
    if split == 'train':
        if args.deit_recipe:
            transform = build_deit_transform(is_train=True, args=args)
        else:
            transform = transforms.Compose([
                transforms.Resize((args.image_size+32, args.image_size+32)),
                transforms.RandomCrop((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1,
                                       contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
    else:
        if args.deit_recipe:
            transform = build_deit_transform(is_train=False, args=args)
        else:
            transform = transforms.Compose([
                transforms.Resize(args.image_size+32),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

    return transform
