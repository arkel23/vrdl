# https://github.com/Spijkervet/SimCLR/blob/master/linear_evaluation.py
# https://github.com/winycg/HSAKD/blob/main/eval_rep.py
import time
import wandb
import torch
import torch.nn as nn

from distiller.models import LinearClassifier
from distiller.dataset.loaders import build_dataloaders
from distiller.helper.parser import parse_option_linear
from distiller.helper.model_utils import load_model, save_model
from distiller.helper.optim_utils import return_optimizer_scheduler
from distiller.helper.misc_utils import count_params_single, random_seed, summary_stats
from distiller.helper.loops import train_vanilla as train, validate, feature_extraction

def get_features(backbone, train_loader, val_loader, opt):
    train_X, train_y = feature_extraction(train_loader, backbone, opt)
    test_X, test_y = feature_extraction(val_loader, backbone, opt)
    return train_X, train_y, test_X, test_y


def get_features_size(backbone, opt):
    data = torch.randn(2, 3, opt.image_size, opt.image_size).to(opt.device)
    with torch.no_grad():
        out = backbone(data, classify_only=False)
    logits_dim = out[-2].shape[-1]
    return logits_dim


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, pin_memory=True)

    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    val_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return train_loader, val_loader


def main():
    time_start = time.time()
    best_acc = 0
    max_memory = 0

    opt = parse_option_linear()
    random_seed(opt.seed, opt.rank)
    
    # dataloader
    train_loader, val_loader, n_cls = build_dataloaders(opt)
    
    # backbone
    backbone = load_model(opt.path_model, n_cls, 'default')
    backbone.to(opt.device)
    backbone.eval()
    
    # linear classifier head and criterion
    logits_dim = get_features_size(backbone, opt)
    classifier = LinearClassifier(in_features=logits_dim, num_classes=n_cls)
    classifier.to(opt.device)

    criterion = nn.CrossEntropyLoss().to(opt.device)
    
    # optimizer and scheduler
    optimizer, lr_scheduler = return_optimizer_scheduler(opt, classifier)
   
    if opt.distributed:
        backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
        backbone = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[opt.local_rank])
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[opt.local_rank])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if opt.local_rank == 0:
        wandb.init(config=opt)
        wandb.run.name = '{}'.format(opt.model_name)
    
    (train_X, train_y, test_X, test_y) = get_features(backbone, train_loader, val_loader, opt)

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, opt.batch_size
    )

    for epoch in range(1 , opt.epochs+1):
        
        if opt.distributed:
            train_loader.sampler.set_epoch(epoch)
        lr_scheduler.step(epoch)       
        
        train_acc, train_loss = train(epoch, arr_train_loader, classifier, criterion, optimizer, opt)
        test_acc, test_loss = validate(arr_test_loader, classifier, criterion, opt)

        if opt.local_rank == 0:
            print("==> Training...Epoch: {} | LR: {}".format(epoch, optimizer.param_groups[0]['lr']))
            wandb.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss,
                       'test_acc': test_acc, 'test_loss': test_loss})  
        
            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                save_model(opt, classifier, epoch, test_acc, mode='best', optimizer=optimizer)
            # regular saving
            if epoch % opt.save_freq == 0:
                save_model(opt, classifier, epoch, test_acc, mode='epoch', optimizer=optimizer)
            # VRAM memory consumption
            curr_max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
            if curr_max_memory > max_memory:
                max_memory = curr_max_memory

    if opt.local_rank == 0:
        # save last model
        save_model(opt, classifier, epoch, test_acc, mode='last', optimizer=optimizer)
        
        # summary stats
        time_end = time.time()
        time_total = time_end - time_start
        no_params = count_params_single(classifier)
        summary_stats(opt.epochs, time_total, best_acc, best_epoch, max_memory, no_params)
        

if __name__ == '__main__':
    main()
