from timm.scheduler import create_scheduler
from timm.optim import create_optimizer


def return_optimizer_scheduler(opt, model):

    opt.warmup_lr = 1e-6
    opt.opt_eps = 1e-8
    opt.opt_betas = None
    opt.lr_noise = None
    opt.lr_noise_pct = 0.67
    opt.lr_noise_std = 1.0
    opt.min_lr = 1e-5
    opt.cooldown_epochs = 10
    opt.patience_epochs = 10

    optimizer = create_optimizer(opt, model)

    lr_scheduler, _ = create_scheduler(opt, optimizer)

    return optimizer, lr_scheduler
