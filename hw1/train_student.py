import time
import wandb
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


from distiller.models import model_extractor, Embed, ConvReg, LinearEmbed
from distiller.models import Connector, Translator, Paraphraser

from distiller.dataset.loaders import build_dataloaders

from distiller.helper.parser import parse_option_student
from distiller.helper.model_utils import load_teacher, save_model
from distiller.helper.optim_utils import return_optimizer_scheduler
from distiller.helper.misc_utils import count_params_module_list, random_seed, summary_stats
from distiller.helper.loops import train_distill as train, validate
from distiller.helper.pretrain import init

from distiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation
from distiller.distiller_zoo import VIDLoss, RKDLoss, PKT, ABLoss, FactorTransfer, KDSVD
from distiller.distiller_zoo import FSP, NSTLoss, CRDLoss, IFACRDLoss

def main():
    time_start = time.time()
    best_acc = 0
    max_memory = 0

    opt = parse_option_student()
    random_seed(opt.seed, opt.rank)
    
    # dataloader
    train_loader, val_loader, n_cls, n_data = build_dataloaders(opt, vanilla=False)
    
    # model
    model_t = load_teacher(opt.path_t, n_cls, opt.layers)
    model_s = model_extractor(opt.model_s, num_classes=n_cls, layers=opt.layers)
    
    # init wandb logger
    if opt.local_rank == 0:
        wandb.init(config=opt)
        wandb.run.name = '{}'.format(opt.model_name)

    data = torch.randn(2, 3, opt.image_size, opt.image_size)
    model_t.eval()
    model_s.eval()
    out_t = model_t(data, classify_only=False)
    out_s = model_s(data, classify_only=False)
    feat_t = out_t[:-1]
    feat_s = out_s[:-1]
    
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'ifacrd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        criterion_kd = IFACRDLoss(opt, model_t)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
        if opt.cont_no_l != 1:
            module_list.append(criterion_kd.rescaler)
            trainable_list.append(criterion_kd.rescaler)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        #raise NotImplementedError
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        #raise NotImplementedError
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)
    
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer and lr
    optimizer, lr_scheduler = return_optimizer_scheduler(opt, trainable_list)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    module_list.to(opt.device)
    criterion_list.to(opt.device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if opt.distributed:
        module_list = nn.SyncBatchNorm.convert_sync_batchnorm(module_list)
        module_list = DDP(module_list, device_ids=[opt.local_rank])
        criterion_list = nn.SyncBatchNorm.convert_sync_batchnorm(criterion_list)
        criterion_list = DDP(criterion_list, device_ids=[opt.local_rank])
    
    # validate teacher accuracy
    teacher_acc, _ = validate(val_loader, model_t, criterion_cls, opt)
    if opt.local_rank == 0:
        print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs+1):
        if opt.distributed:
            train_loader.sampler.set_epoch(epoch)
        lr_scheduler.step(epoch)

        train_acc, train_loss, loss_cls, loss_div, loss_kd = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        test_acc, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        if opt.local_rank == 0:
            print("==> Training...Epoch: {} | LR: {}".format(epoch, optimizer.param_groups[0]['lr']))
            wandb.log({'epoch': epoch, 'train_acc': train_acc, 'train_loss': train_loss, 
                       'train_loss_cls': loss_cls, 'train_loss_div': loss_div, 'train_loss_kd': loss_kd,
                       'test_acc': test_acc, 'test_loss': test_loss})

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                save_model(opt, model_s, epoch, test_acc, mode='best', vanilla=False)
            # regular saving
            if epoch % opt.save_freq == 0:
                save_model(opt, model_s, epoch, test_acc, mode='epoch', vanilla=False)
            # VRAM memory consumption
            curr_max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
            if curr_max_memory > max_memory:
                max_memory = curr_max_memory
                
    if opt.local_rank == 0:  
        # save last model     
        save_model(opt, model_s, epoch, test_acc, mode='last', vanilla=False)

        # summary stats
        time_end = time.time()
        time_total = time_end - time_start
        
        no_params_modules = count_params_module_list(module_list)
        no_params_criterion = count_params_module_list(criterion_list)
        no_params = no_params_modules + no_params_criterion
        
        summary_stats(opt.epochs, time_total, best_acc, best_epoch, max_memory, no_params)


if __name__ == '__main__':
    main()
