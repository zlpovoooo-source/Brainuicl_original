import copy
import numpy as np
import torch


def _collect_named_params(blocks):
    """blocks = (feature_extractor, feature_encoder, classifier)"""
    prefix = ["fe", "enc", "clf"] # 提取器/编码器/分类器
    named = {}
    for pfx, module in zip(prefix, blocks):
        for name, param in module.named_parameters():
            named[f"{pfx}.{name}"] = param
    return named


def snapshot_blocks(blocks):
    """保存当前时刻可训练参数快照（cpu numpy）"""
    named = _collect_named_params(blocks)
    snap = {}
    for name, param in named.items():
        snap[name] = copy.deepcopy(param.data).detach().cpu().numpy()
    return snap


def calculate_variance(weight_snapshots):
    """
    weight_snapshots: list[np.ndarray], shape = [n_epoch, ...]
    """
    arr = np.stack(weight_snapshots, axis=0)
    return np.var(arr, axis=0)


def freeze_stable_params(blocks, weight_histories, threshold=1e-5, exclude_keys=None):
    """
    weight_histories: dict[name] = [snap_epoch1, snap_epoch2, ...]
    exclude_keys: list[str], 参数名中包含任一关键字则跳过冻结
    """
    if exclude_keys is None:
        exclude_keys = []

    named = _collect_named_params(blocks)
    frozen_names = []

    for name, param in named.items():
        if not param.requires_grad:
            continue
        if any(k in name for k in exclude_keys):
            continue
        if name not in weight_histories or len(weight_histories[name]) < 2:
            continue

        variances = calculate_variance(weight_histories[name])

        # 与 CMuST 一致：整张权重矩阵都足够稳定才冻结
        if np.all(variances < threshold):
            param.requires_grad = False
            frozen_names.append(name)

    return frozen_names


def count_trainable_params(blocks):
    total = 0
    trainable = 0
    for module in blocks:
        for p in module.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
    return total, trainable, total - trainable