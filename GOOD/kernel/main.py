r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import os
import time
from typing import Tuple, Union
import numpy as np
import torch.nn
from torch.utils.data import DataLoader

from GOOD import config_summoner
from GOOD.data import load_dataset, create_dataloader
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.networks.model_manager import load_model
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import CommonArgs, Munch, process_configs
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.logger import load_logger
from GOOD.definitions import OOM_CODE


def initialize_model_dataset(config: Union[CommonArgs, Munch], fold: int = 0) -> Tuple[torch.nn.Module, Union[dict, DataLoader]]:
    r"""
    Fix random seeds and initialize a GNN and a dataset. (For project use only)

    Returns:
        A GNN and a data loader.
    """
    # Initial
    reset_random_seed(config)

    print(f'#IN#\n-----------------------------------\n    Task: {config.task}\n'
          f'{time.asctime(time.localtime(time.time()))}')
    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config, fold)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, loader


def compute_10fold_metrics(ckpts):
    train_scores = []
    id_val_scores = []
    id_test_scores = []
    ood_val_scores = []
    ood_test_scores = []
    val_scores = []
    test_scores = []
    for ckpt in ckpts:
        train_scores.append(ckpt['train_score'])
        id_val_scores.append(ckpt['id_val_score'])
        id_test_scores.append(ckpt['id_test_score'])
        ood_val_scores.append(ckpt['ood_val_score'])
        ood_test_scores.append(ckpt['ood_test_score'])
        val_scores.append(ckpt['val_score'])
        test_scores.append(ckpt['test_score'])

    # compute the mean and std of these metrics, keep four decimal places
    train_mean = torch.mean(torch.tensor(train_scores))
    train_std = torch.std(torch.tensor(train_scores))
    id_val_mean = torch.mean(torch.tensor(id_val_scores))
    id_val_std = torch.std(torch.tensor(id_val_scores))
    id_test_mean = torch.mean(torch.tensor(id_test_scores))
    id_test_std = torch.std(torch.tensor(id_test_scores))
    ood_val_mean = torch.mean(torch.tensor(ood_val_scores))
    ood_val_std = torch.std(torch.tensor(ood_val_scores))
    ood_test_mean = torch.mean(torch.tensor(ood_test_scores))
    ood_test_std = torch.std(torch.tensor(ood_test_scores))
    val_score_mean = torch.mean(torch.tensor(val_scores))
    val_score_std = torch.std(torch.tensor(val_scores))
    test_score_mean = torch.mean(torch.tensor(test_scores))
    test_score_std = torch.std(torch.tensor(test_scores))
    results = {
        'train_mean': round(train_mean.item(), 4),
        'train_std': round(train_std.item(), 4),
        'id_val_mean': round(id_val_mean.item(), 4),
        'id_val_std': round(id_val_std.item(), 4),
        'id_test_mean': round(id_test_mean.item(), 4),
        'id_test_std': round(id_test_std.item(), 4),
        'ood_val_mean': round(ood_val_mean.item(), 4),
        'ood_val_std': round(ood_val_std.item(), 4),
        'ood_test_mean': round(ood_test_mean.item(), 4),
        'ood_test_std': round(ood_test_std.item(), 4),
        'val_score_mean': round(val_score_mean.item(), 4),
        'val_score_std': round(val_score_std.item(), 4),
        'test_score_mean': round(test_score_mean.item(), 4),
        'test_score_std': round(test_score_std.item(), 4)
    }
    return results


def compute_mixed_10fold_metrics(ckpts, id_ckpts):
    val_scores = []
    test_scores = []
    test_precision = []
    test_recall = []
    test_f1 = []
    test_roc_auc = []
    for ckpt, id_ckpt in zip(ckpts, id_ckpts):
        val_scores.append((id_ckpt['id_val_score'] * id_ckpt['id_val_subject_num'] + ckpt['ood_val_score'] * ckpt['ood_val_subject_num']) / (id_ckpt['id_val_subject_num'] + ckpt['ood_val_subject_num']))
        test_scores.append((id_ckpt['id_test_score'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_score'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))
        test_precision.append((id_ckpt['id_test_precision'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_precision'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))
        test_recall.append((id_ckpt['id_test_recall'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_recall'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))
        test_f1.append((id_ckpt['id_test_f1'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_f1'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))
        test_roc_auc.append((id_ckpt['id_test_roc_auc'] * id_ckpt['id_test_subject_num'] + ckpt['ood_test_roc_auc'] * ckpt['ood_test_subject_num']) / (id_ckpt['id_test_subject_num'] + ckpt['ood_test_subject_num']))

    # compute the mean and std of these metrics, keep four decimal places
    val_score_mean = torch.mean(torch.tensor(val_scores))
    val_score_std = torch.std(torch.tensor(val_scores))
    test_score_mean = torch.mean(torch.tensor(test_scores))
    test_score_std = torch.std(torch.tensor(test_scores))
    test_precision_mean = torch.mean(torch.tensor(test_precision))
    test_precision_std = torch.std(torch.tensor(test_precision))
    test_recall_mean = torch.mean(torch.tensor(test_recall))
    test_recall_std = torch.std(torch.tensor(test_recall))
    test_f1_mean = torch.mean(torch.tensor(test_f1))
    test_f1_std = torch.std(torch.tensor(test_f1))
    test_roc_auc_mean = torch.mean(torch.tensor(test_roc_auc))
    test_roc_auc_std = torch.std(torch.tensor(test_roc_auc))
    results = {
        'val_score_mean': round(val_score_mean.item(), 4),
        'val_score_std': round(val_score_std.item(), 4),
        'test_score_mean': round(test_score_mean.item(), 4),
        'test_score_std': round(test_score_std.item(), 4),
        'test_precision_mean': round(test_precision_mean.item(), 4),
        'test_precision_std': round(test_precision_std.item(), 4),
        'test_recall_mean': round(test_recall_mean.item(), 4),
        'test_recall_std': round(test_recall_std.item(), 4),
        'test_f1_mean': round(test_f1_mean.item(), 4),
        'test_f1_std': round(test_f1_std.item(), 4),
        'test_roc_auc_mean': round(test_roc_auc_mean.item(), 4),
        'test_roc_auc_std': round(test_roc_auc_std.item(), 4)
    }
    return results


def main():
    args = args_parser()
    config = config_summoner(args)
    id_ckpts = []
    ckpts = []

    for i in range(10):
        print(f'\nFold {i + 1}')
        process_configs(config, i)
        if i == 0:
            load_logger(config)

        config.task = 'train'
        model, loader = initialize_model_dataset(config, i)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

        pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
        if i == 0:
            view_model_param(pipeline.model)
        pipeline.load_task(fold=i)

        if config.task == 'train':
            pipeline.task = 'test'
            id_ckpt, ckpt = pipeline.load_task(fold=i)
            id_ckpts.append(id_ckpt)
            ckpts.append(ckpt)

    id_ckpt_results = compute_10fold_metrics(id_ckpts)
    ckpt_results = compute_10fold_metrics(ckpts)
    mixed_results = compute_mixed_10fold_metrics(ckpts, id_ckpts)

    print('#IN#\n\nID-ckpt results:')
    print('#IN#Train: {} ± {}'.format(id_ckpt_results['train_mean'], id_ckpt_results['train_std']))
    print('#IN#ID-val: {} ± {}'.format(id_ckpt_results['id_val_mean'], id_ckpt_results['id_val_std']))
    print('#IN#ID-test: {} ± {}'.format(id_ckpt_results['id_test_mean'], id_ckpt_results['id_test_std']))
    print('#IN#OOD-val: {} ± {}'.format(id_ckpt_results['ood_val_mean'], id_ckpt_results['ood_val_std']))
    print('#IN#OOD-test: {} ± {}'.format(id_ckpt_results['ood_test_mean'], id_ckpt_results['ood_test_std']))
    print('#IN#Val: {} ± {}'.format(id_ckpt_results['val_score_mean'], id_ckpt_results['val_score_std']))
    print('#IN#Test: {} ± {}'.format(id_ckpt_results['test_score_mean'], id_ckpt_results['test_score_std']))
    print('#IN#\nOOD-ckpt results:')
    print('#IN#Train: {} ± {}'.format(ckpt_results['train_mean'], ckpt_results['train_std']))
    print('#IN#ID-val: {} ± {}'.format(ckpt_results['id_val_mean'], ckpt_results['id_val_std']))
    print('#IN#ID-test: {} ± {}'.format(ckpt_results['id_test_mean'], ckpt_results['id_test_std']))
    print('#IN#OOD-val: {} ± {}'.format(ckpt_results['ood_val_mean'], ckpt_results['ood_val_std']))
    print('#IN#OOD-test: {} ± {}'.format(ckpt_results['ood_test_mean'], ckpt_results['ood_test_std']))
    print('#IN#Val: {} ± {}'.format(ckpt_results['val_score_mean'], ckpt_results['val_score_std']))
    print('#IN#Test: {} ± {}'.format(ckpt_results['test_score_mean'], ckpt_results['test_score_std']))
    print('#IN#\nMixed results:')
    print('#IN#Val: {} ± {}'.format(mixed_results['val_score_mean'], mixed_results['val_score_std']))
    print('#IN#Test: {} ± {}'.format(mixed_results['test_score_mean'], mixed_results['test_score_std']))
    print('#IN#Test precision: {} ± {}'.format(mixed_results['test_precision_mean'], mixed_results['test_precision_std']))
    print('#IN#Test recall: {} ± {}'.format(mixed_results['test_recall_mean'], mixed_results['test_recall_std']))
    print('#IN#Test F1: {} ± {}'.format(mixed_results['test_f1_mean'], mixed_results['test_f1_std']))
    print('#IN#Test ROC AUC: {} ± {}'.format(mixed_results['test_roc_auc_mean'], mixed_results['test_roc_auc_std']))


def goodtg():
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'#E#{e}')
            exit(OOM_CODE)
        else:
            raise e


def view_model_param(model):
    # model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('Total parameters:', total_param)


if __name__ == '__main__':
    main()
