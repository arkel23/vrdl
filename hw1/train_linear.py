# https://github.com/Spijkervet/SimCLR/blob/master/linear_evaluation.py
# https://github.com/winycg/HSAKD/blob/main/eval_rep.py
import time
import wandb
import torch
import torch.nn as nn

from fgvr.models import LinearClassifier
from fgvr.dataset.build_dataloaders import build_dataloaders
from fgvr.utils.parser import parse_option_linear
from fgvr.utils.model_utils import load_model_nohead, save_model
from fgvr.utils.optim_utils import return_optimizer_scheduler
from fgvr.utils.misc_utils import count_params_single, set_seed, summary_stats
from fgvr.utils.loops import train_vanilla as train, validate, feature_extraction


def get_features(backbone, train_loader, val_loader, args):
    train_X, train_y = feature_extraction(train_loader, backbone, args)
    val_X, val_y = feature_extraction(val_loader, backbone, args)
    return train_X, train_y, val_X, val_y


def get_features_size(backbone, image_size, device):
    data = torch.randn(2, 3, image_size, image_size).to(device)
    with torch.no_grad():
        out = backbone(data, classify_only=False)
    logits_dim = out[-2].shape[-1]
    return logits_dim


def create_data_loaders_from_arrays(X_train, y_train, X_val, y_val, batch_size):
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), 
                                           torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, 
                                               shuffle=False, pin_memory=True)

    val = torch.utils.data.TensorDataset(torch.from_numpy(X_val), 
                                          torch.from_numpy(y_val))
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, 
                                             shuffle=False, pin_memory=True
    )
    return train_loader, val_loader


def main():
    time_start = time.time()
    best_acc = 0
    max_memory = 0

    args = parse_option_linear()
    set_seed(args.seed, args.rank)
    
    # dataloader
    train_loader, val_loader, n_cls = build_dataloaders(args)
    
    # backbone
    backbone = load_model_nohead(
        args.path_model, args.model, n_cls, args.image_size,
        args.pretrained, 'default')
    backbone.to(args.device)
    backbone.eval()
    
    # linear classifier head and criterion
    logits_dim = get_features_size(backbone, args.image_size, args.device)
    classifier = LinearClassifier(in_features=logits_dim, num_classes=n_cls)
    classifier.to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    
    # optimizer and scheduler
    optimizer, lr_scheduler = return_optimizer_scheduler(args, classifier)
   
    if args.distributed:
        backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
        backbone = torch.nn.parallel.DistributedDataParallel(
            backbone, device_ids=[args.local_rank])
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[args.local_rank])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.local_rank == 0:
        wandb.init(config=args)
        wandb.run.name = '{}'.format(args.model_name)
    
    (train_X, train_y, val_X, val_y) = get_features(
        backbone, train_loader, val_loader, args)

    arr_train_loader, arr_val_loader = create_data_loaders_from_arrays(
        train_X, train_y, val_X, val_y, args.batch_size
    )

    for epoch in range(1 , args.epochs+1):
        
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        lr_scheduler.step(epoch)       
        
        train_acc, train_loss = train(epoch, arr_train_loader, classifier, criterion, optimizer, args)
        val_acc, val_loss = validate(arr_val_loader, classifier, criterion, args)

        if args.local_rank == 0:
            print("==> Training...Epoch: {} | LR: {}".format(epoch, optimizer.param_groups[0]['lr']))
            wandb.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss,
                       'val_acc': val_acc, 'val_loss': val_loss})  
        
            # save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                save_model(args, classifier, epoch, val_acc, mode='best', optimizer=optimizer)
            # regular saving
            if epoch % args.save_freq == 0:
                save_model(args, classifier, epoch, val_acc, mode='epoch', optimizer=optimizer)
            # VRAM memory consumption
            curr_max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
            if curr_max_memory > max_memory:
                max_memory = curr_max_memory

    if args.local_rank == 0:
        # save last model
        save_model(args, classifier, epoch, val_acc, mode='last', optimizer=optimizer)
        
        # summary stats
        time_end = time.time()
        time_total = time_end - time_start
        no_params = count_params_single(classifier)
        summary_stats(args.epochs, time_total, best_acc, best_epoch, max_memory, no_params)
        

if __name__ == '__main__':
    main()
