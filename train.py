"""
Created on Sept 11, 2020

@author: Shiying Ni
"""

import os
from datetime import datetime
import time
import argparse
import json
from collections import defaultdict

# Supress INFO and WARNING, should be placed before import tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='Train model')

# basic arguments
parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--dataset', default='taobao')
parser.add_argument('--mode', default='train_500K')
parser.add_argument('--model_type', default='IMRec')
parser.add_argument('--run_times', default=1, type=int)
parser.add_argument('--K-int-type', nargs='+', default=[1, 5, 10, 20, 50], type=int)
parser.add_argument('--total_epochs', default=30, type=int)
parser.add_argument('--eval_interval', default=1, type=int)
parser.add_argument('--verbose', default=1, type=int)
parser.add_argument('--patience', default=3, type=int)

# some key arguments in config.py, can be overwritten by passing command line arguments
parser.add_argument('--embed_dim', default=-1, type=int)
parser.add_argument('--att_len', default=-1, type=int)
parser.add_argument('--alpha', default=-1, type=float)

args, unknown = parser.parse_known_args()

if not args.use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # disable gpu
    print('Ban GPU device.')

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as backend
from openpyxl import load_workbook

from module import RecordLoss, RecordMetrics, HR
from utils import get_data
from configs import _model_config, _ds_config, _exp_config, _models


# =============================== GPU ==============================
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print("Num GPUs Available:", len(gpus))
if gpus:
    tf.config.experimental.set_visible_devices(gpus[:], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Use GPU when training.')
else:
    print('Do not use GPU when training.')

# =========================Shared Hyper Parameters =======================
# key args
dataset = args.dataset
mode = args.mode
model_type = args.model_type

# load configs
ds_config = _ds_config.get(model=model_type, dataset=dataset, mode=mode)
model_config = _model_config.get(model=model_type, dataset=dataset, mode=mode)
exp_config = _exp_config.get(model=model_type, dataset=dataset, mode=mode)

# basic args
run_times = args.run_times
total_epochs = args.total_epochs
eval_interval = args.eval_interval
verbose = args.verbose
K = args.K_int_type
patience = args.patience

# exp args
learning_rate = exp_config['learning_rate']
batch_size = exp_config['batch_size']

# dataset args
embed_dim = args.embed_dim if args.embed_dim != -1 else ds_config['embed_dim']
recurrent = False if model_type == 'SASRec' else True
ds_config.update(
    embed_dim=embed_dim,
    recurrent=recurrent,
    )

# IMRec model args
if model_type == 'IMRec':
    att_len = args.att_len if args.att_len != -1 else model_config['att_len']
    alpha = args.alpha if args.alpha != -1 else model_config['alpha']
    model_config.update(att_len=att_len, alpha=alpha)

exp_args = {}
print('Experimental Args'.center(50, '='))
for arg in vars(args):
    if getattr(args, arg) != -1:
        print(f'{arg.capitalize()}: {getattr(args, arg)}')
        exp_args[arg] = getattr(args, arg)

# ========================== Create dataset =======================
feature_columns, train, val, test = get_data(
    dataset=dataset,
    mode=mode,
    redump=False,
    **ds_config,
    )

# dataset args
print('Dataset Args'.center(50, '='))
for key, value in ds_config.items():
    print(f'{key.capitalize()}: {value}')

def generate_dataset(data):
    label = data['target_item_pos']  # dummy ouputs
    return data, label

trainsets, trainlabel = generate_dataset(train)
validationsets, validationlabel = generate_dataset(val)
testsets, testlabel = generate_dataset(test)

now = datetime.now().strftime('%m%d%H%M')

# ===========================Run Experiments============================
if __name__ == '__main__':

    summary_list = []
    exp_begin = time.time()
    for run in range(1, run_times + 1):
        model = None

    # =============================Build Model==============================

        model_config.update(maxlen=ds_config['maxlen'])
        model = _models[model_type](feature_columns, model_config)

        val_K = 20
        val_metrics = f'HR@{val_K}'
        model.compile(optimizer=Adam(learning_rate=exp_config['learning_rate']),
                      metrics=[HR(K=val_K, name=val_metrics)],
                      )

        log_dir = './log'
        os.makedirs(log_dir, exist_ok=True)

        model_name = model.model_name
        para_info = f'{now}_{model_name}_{dataset}-{mode[6:]}_bs_{batch_size}_dim_{embed_dim}'

        # creat log folder
        if args.model_type == 'IMRec':
            para_info += f'_len_{att_len}'
            para_info += f'_a_{alpha}'

        exp_dir = f'{log_dir}/{para_info}'
        os.makedirs(exp_dir, exist_ok=True)

        # model args
        print('Model Args'.center(50, '='))
        for key, value in model_config.items():
            print(f'{key.capitalize()}: {value}')

        # ========================Fit============================
        print(f'{model.model_name}: {run}/{run_times} run'.center(50, '*'))

        results_list = []
        best_score = 0.0
        best_model = None
        loss_log = defaultdict(list)  # record losses
        metrics_log = defaultdict(list)  # record metrics
        ckpt_dir = f'{exp_dir}/run_{run}'
        early_stopping = EarlyStopping(
            monitor=f'val_{val_metrics}',
            min_delta=0.0001,
            patience=patience,
            verbose=1,
            mode='max',
            baseline=None,
            restore_best_weights=True)

        model.fit(
            x=trainsets,
            y=trainlabel,
            validation_data=(validationsets, validationlabel),
            batch_size=batch_size,
            epochs=total_epochs,
            verbose=verbose,
            workers=8,
            use_multiprocessing=True,
            callbacks=[
                RecordLoss(loss_log, mode='batch'),
                RecordMetrics(testsets, metrics_log, interval=eval_interval, K=K),
                early_stopping,
                 ],
                  )
        # save the restored best model
        try:
            model.save(f'{ckpt_dir}/best')
        except Exception:
            print('Saving model failed...')

        metrics_this_run = pd.DataFrame(metrics_log)
        loss_this_run = pd.DataFrame(loss_log)

        loss_cols = [col for col in loss_this_run.columns if 'loss' in col]
        metrics_loss_this_run = loss_this_run[['Epoch', 'Fitting Time'] + loss_cols]
        metrics_loss_this_run = metrics_loss_this_run.drop_duplicates(subset=['Epoch'], keep='last')  # 选择每个epoch最后一个batch的loss值
        metrics_this_run = pd.merge(metrics_this_run, metrics_loss_this_run, on='Epoch', how='left')

        if early_stopping.stopped_epoch == 0:
            best_epoch = total_epochs - 1  # if not early stopping, save the last epoch
        else:
            best_epoch = early_stopping.stopped_epoch - patience
        best_metrics = metrics_this_run.loc[best_epoch:best_epoch, :].copy()
        best_metrics.loc[best_epoch, 'Run'] = run
        summary_list.append(best_metrics)

        # =========================Save Model and Write Log Files=======================
        # save loss
        loss_file = exp_dir + f'/loss_{para_info}.xlsx'
        # create new .xlsx file
        if not os.path.exists(loss_file):
            pd.DataFrame([]).to_excel(loss_file,
                                      sheet_name='summary',
                                      header=True, index=False)
        # add loss sheet for each run
        book = load_workbook(loss_file)
        with pd.ExcelWriter(loss_file, engine='openpyxl') as excel_writer:
            excel_writer.book = book
            excel_writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            loss_this_run.to_excel(excel_writer,
                                   sheet_name='{}th_run'.format(run),
                                   header=True, index=False)
            print(f'Save loss of {run}th run'.center(50, '*'))

        # save metrics
        metrics_file = exp_dir + f'/metrics_{para_info}.xlsx'
        # create new .xlsx file
        if not os.path.exists(metrics_file):
            pd.DataFrame([]).to_excel(metrics_file,
                                      sheet_name='summary',
                                      header=True, index=False)
            # print(f'New metrics file: {os.path.basename(metrics_file)}')
        # add metrics sheet for each run
        book = load_workbook(metrics_file)
        with pd.ExcelWriter(metrics_file, engine='openpyxl') as excel_writer:
            excel_writer.book = book
            excel_writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            metrics_this_run.to_excel(excel_writer,
                                      sheet_name='{}th_run'.format(run),
                                      header=True, index=False)
            print(f'Save metrics of {run}th run'.center(50, '*'))

        # save summary of this exp
        summary = pd.concat(summary_list)
        summary.loc['average'] = summary.mean()
        summary.loc['average', 'Run'] = 'average'
        with pd.ExcelWriter(metrics_file, engine='openpyxl') as excel_writer:
            excel_writer.book = book
            excel_writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            summary.to_excel(excel_writer,
                             sheet_name='summary',
                             header=True, index=True)

    # save all args
    all_config = {}
    all_config.update(exp_args)
    all_config.update(exp_config)
    all_config.update(ds_config)
    all_config.update(model_config)

    trainable_count = int(np.sum([backend.count_params(w) for w in model.trainable_weights]))
    non_trainable_count = int(np.sum([backend.count_params(w) for w in model.non_trainable_weights]))
    all_config['trainable_params'] = trainable_count
    all_config['non_trainable_params'] = non_trainable_count

    with open(f'{exp_dir}/all_config.json', 'a') as file:
        json.dump(all_config, file)

    # rename the log folder with the average score
    best_score = summary.loc['average', val_metrics]
    new_exp_dir = f'{exp_dir}_{best_score:.4f}'
    os.rename(src=exp_dir, dst=new_exp_dir)

    print(f'Rename exp dir as {new_exp_dir}')
    exp_end = time.time()
    total_time = (exp_end - exp_begin) / 60
    time_per_run =  total_time / run_times

    print('Experiment done'.center(50, '='))

    print('Trainable params: {:,}'.format(int(trainable_count)))
    print('Non-trainable params: {:,}'.format(int(non_trainable_count)))
    print(f'Average best score of {run_times} runs: {val_metrics} = {best_score:.4f}')

    print(f'{run_times}-run time: {total_time:.2f} min.')
    print(f'Average time: {time_per_run:.2f} min.')
