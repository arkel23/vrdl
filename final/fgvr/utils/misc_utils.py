import random
import torch
import wandb


def count_params_module_list(module_list):
    return sum([count_params_single(model) for model in module_list])


def count_params_single(model):
    return sum([p.numel() for p in model.parameters()])


def set_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)


def summary_stats(epochs, time_total, best_acc,
                  best_epoch, max_memory, no_params):
    time_avg = time_total / epochs
    best_time = time_avg * best_epoch
    no_params = no_params / (1e6)

    print('''Total run time (s): {}
          Average time per epoch (s): {}
          Best accuracy (%): {} at epoch {}. Time to reach this (s): {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          '''.format(time_total, time_avg, best_acc, best_epoch,
                     best_time, max_memory, no_params))

    wandb.run.summary['time_total'] = time_total
    wandb.run.summary['time_avg'] = time_avg
    wandb.run.summary['best_acc'] = best_acc
    wandb.run.summary['best_epoch'] = best_epoch
    wandb.run.summary['best_time'] = best_time
    wandb.run.summary['max_memory'] = max_memory
    wandb.run.summary['no_params'] = no_params

    wandb.finish()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
    for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
