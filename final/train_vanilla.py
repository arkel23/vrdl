import time
import wandb
import torch

from fgvr.models import model_extractor
from fgvr.dataset.build_dataloaders import build_dataloaders
from fgvr.utils.model_utils import save_model
from fgvr.utils.parser import parse_option_vanilla
from fgvr.utils.optim_utils import return_optimizer_scheduler
from fgvr.utils.loops import train_vanilla as train, validate
from fgvr.utils.misc_utils import count_params_single, set_seed, summary_stats


def main():
    time_start = time.time()
    best_acc = 0
    best_epoch = 0
    max_memory = 0

    args = parse_option_vanilla()
    set_seed(args.seed, args.rank)

    # dataloader
    train_loader, val_loader, n_cls = build_dataloaders(args)

    # model and criterion
    model = model_extractor(
        args.model, n_cls, args.image_size,
        args.freeze, args.pretrained, 'last_only')
    model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # optimizer and scheduler
    optimizer, lr_scheduler = return_optimizer_scheduler(args, model)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.local_rank == 0:
        wandb.init(config=args)
        wandb.run.name = '{}'.format(args.model_name)

    # routine
    for epoch in range(1, args.epochs+1):
        lr_scheduler.step(epoch)
        train_acc, train_loss = train(epoch, train_loader, model, criterion,
                                      optimizer, args)
        if not args.skip_eval:
            val_acc, val_loss = validate(val_loader, model, criterion, args)
        else:
            val_acc = 0
            val_loss = 0

        print("Training...Epoch: {} | LR: {}".format(
            epoch, optimizer.param_groups[0]['lr']))
        wandb.log({'epoch': epoch, 'train_acc': train_acc,
                   'train_loss': train_loss,
                   'val_acc': val_acc, 'val_loss': val_loss})

        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_model(args, model, epoch, val_acc,
                       mode='best', optimizer=optimizer)
        # regular saving
        if epoch % args.save_freq == 0:
            save_model(args, model, epoch, val_acc,
                       mode='epoch', optimizer=optimizer)
        # VRAM memory consumption
        curr_max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
        if curr_max_memory > max_memory:
            max_memory = curr_max_memory

    # save last model
    save_model(args, model, epoch, val_acc, mode='last', optimizer=optimizer)

    # summary stats
    time_end = time.time()
    time_total = time_end - time_start
    no_params = count_params_single(model)
    summary_stats(args.epochs, time_total, best_acc,
                  best_epoch, max_memory, no_params)


if __name__ == '__main__':
    main()
