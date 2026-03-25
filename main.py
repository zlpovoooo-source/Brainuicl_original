import torch
import argparse
import os
import numpy as np
import logging
from utils.util import fix_randomness, analysis
from utils.logging_utils import setup_logging
from torch.utils.data import DataLoader
from dataloader.data_loader import Builder
from trainer.pretrainer import pretraining
from trainer.trainer import trainer
# from trainer.trainer_bci2000 import trainer_bci2000


logger = logging.getLogger(__name__)

def get_path_loader(params):
    path = [i for i in range(1, 101) if i not in [8, 40]]
    path_name = {int(j): [[], []] for j in path}
    for t_idx in path:
        num = 0
        file_path = params.file_path + f"/{t_idx}/data"
        label_path = params.file_path + f"/{t_idx}/label"
        while os.path.exists(file_path + f"/{num}.npy"):
            path_name[t_idx][0].append(file_path + f"/{num}.npy")
            path_name[t_idx][1].append(label_path + f"/{num}.npy")
            num += 1

    return path, path_name


def get_idx(params, path, performance):
    fix_randomness(params.seed)
    idx = path
    path_len = len(idx)
    old_task_idx = list(np.random.choice(idx, int(path_len*0.2), replace=False))
    new_task_idx = list(np.random.choice(list(set(idx)-set(old_task_idx)), int(path_len*0.5), replace=False))

    train_val_idx = list(set(idx)-set(old_task_idx)-set(new_task_idx))

    train_idx = list(np.random.choice(train_val_idx, int(len(train_val_idx)*0.8), replace=False))
    params.train_num = len(train_idx)
    val_idx = [i for i in train_val_idx if i not in train_idx]
    performance["stability"] = {"ACC": [], "MF1": [], "AAA": [], "FR": []}
    performance["plasticity"] = {i: {"ACC": [], "MF1": []} for i in new_task_idx}

    logger.info(
        "Train Idx %s %s | Val Idx %s %s | Old Task Idx %s %s | New Task Idx %s %s",
        len(train_idx), sorted(train_idx), len(val_idx), sorted(val_idx),
        len(old_task_idx), sorted(old_task_idx), len(new_task_idx), sorted(new_task_idx),
    )
    return train_idx, val_idx, old_task_idx, new_task_idx


def get_loader(params, path, path_name, performance):
    train_path = [[], []]
    val_path = [[], []]
    old_path = [[], []]
    train_idx, val_idx, old_task_idx, new_task_idx = get_idx(params, path, performance)

    for t_idx in train_idx:
        train_path[0].extend(path_name[t_idx][0])
        train_path[1].extend(path_name[t_idx][1])

    for v_idx in val_idx:
        val_path[0].extend(path_name[v_idx][0])
        val_path[1].extend(path_name[v_idx][1])

    for o_idx in old_task_idx:
        old_path[0].extend(path_name[int(o_idx)][0])
        old_path[1].extend(path_name[int(o_idx)][1])

    params.train_path = train_path
    params.train_len = len(params.train_path[0])
    train_builder = Builder(train_path, params).Dataset
    val_builder = Builder(val_path, params).Dataset
    old_task_builder = Builder(old_path, params).Dataset
    logger.info("Buffer_Length %s %s", len(params.train_path[0]), len(params.train_path[1]))

    return train_builder, val_builder, old_task_builder, new_task_idx


def main():
    parser = argparse.ArgumentParser(description='Unsupervised Individual Continual Learning Framework')
    parser.add_argument('--pretrain_epoch', type=int, default=100, help='pretrain epoch')
    parser.add_argument('--incremental_epoch', type=int, default=10, help='incremental epoch')
    parser.add_argument('--ssl_epoch', type=int, default=10, help='self-supervised learning epoch')
    parser.add_argument('--algorithm', type=str, default='cpc', help='ssl algorithm')
    parser.add_argument('--dataset', type=str, default='ISRUC', help='ISRUC dataset')
    parser.add_argument('--gpu', type=int, default=0, help='cuda number(default:0)')
    parser.add_argument('--cross_epoch', type=int, default=2, help='CEA cross epoch alignment')
    parser.add_argument('--seed', type=int, default=4321, help='random seed')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ssl_lr', type=float, default=1e-6, help='ssl learning rate')
    parser.add_argument('--cl_lr', type=float, default=1e-6, help='cl learning rate')
    parser.add_argument('--alpha', type=int, default=0.01, help='loss weight')
    parser.add_argument('--file_path', type=str, default="/data3/lzy/ZZ/SPICED/ISRUC/Group1_processed/", help='data file path')  # input your file path
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--num_worker', type=int, default=4, help='num worker')
    parser.add_argument('--is_pretrain', type=bool, default=False, help='pretraining')
    parser.add_argument('--train_path', type=list, default=None, help='train path')
    parser.add_argument('--train_len', type=int, default=0, help='train path len')
    parser.add_argument('--train_num', type=int, default=0, help='train individual number')

    # 权重冻结
    # parser.add_argument('--wbm_enable', type=bool, default=True, help='enable weight behavior modeling freeze')
    # parser.add_argument('--wbm_threshold', type=float, default=1e-5, help='variance threshold for stable weights')
    # parser.add_argument('--wbm_exclude', type=str, default='clf,bn', help='comma separated exclude keys')

    params = parser.parse_args()
    # params.wbm_exclude = [s.strip() for s in params.wbm_exclude.split(',') if s.strip()]

    log_path = setup_logging(params)
    logger.info("Logging to %s", log_path)
    logger.info("Run params: %s", params)

    performance = dict()

    fix_randomness(params.seed)
    torch.multiprocessing.set_start_method('spawn')
    path, path_name = get_path_loader(params)

    train_dataset, val_dataset, old_dataset, new_task_idx = get_loader(params, path, path_name, performance)

    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch,
                              shuffle=False, num_workers=params.num_worker)
    val_loader = DataLoader(dataset=val_dataset, batch_size=params.batch,
                            shuffle=True, num_workers=params.num_worker)
    old_task_loader = DataLoader(dataset=old_dataset, batch_size=params.batch,
                                 shuffle=True, num_workers=params.num_worker)
    if params.is_pretrain:
        pretraining(train_loader, val_loader, params)
    else:
        # if params.dataset == 'BCI2000':
        #     trainer_bci2000(old_task_loader, new_task_idx, params, performance)
        # else:
        trainer(old_task_loader, new_task_idx, params, performance)
    has_full = False
    if performance.get("plasticity"):
        has_full = any(
            len(v.get("ACC", [])) >= 3 and len(v.get("MF1", [])) >= 3
            for v in performance["plasticity"].values()
        )
    if has_full:
        analysis(performance)
    else:
        logger.warning("Skip analysis: plasticity metrics are incomplete.")
    # analysis(performance)


if __name__ == '__main__':
    main()



