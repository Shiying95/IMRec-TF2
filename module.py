# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:08:26 2021

@author: Shiying Ni
"""
import time
import IPython
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer


class MyConfig(object):
    def __init__(self, models, datasets, modes):
        config = {}
        for model in models:
            config[model] = {}
            for dataset in datasets:
                config[model][dataset] = {}
                for mode in modes:
                    config[model][dataset][mode] = {}

        self.config = config
        self.datasets = datasets
        self.modes = modes
        self.models = models

    def update(self, model=None, dataset=None, mode=None, update_dict=None):
        models = []
        datasets = []
        modes = []

        def trans(key, default):
            if not key:
                return default
            elif isinstance(key, list):
                return key
            else:
                return [key]

        models = trans(model, self.models)
        datasets = trans(dataset, self.datasets)
        modes = trans(mode, self.modes)

        for model in models:
            for dataset in datasets:
                for mode in modes:
                    try:
                        self.config[model][dataset][mode].update(update_dict)
                    except Exception as e:
                        print(f'Error when update model [{model}] dataset [{dataset}] mode [{mode}]: {e}')

    def get_all_configs(self):
        return self.config

    def get(self, model=None, dataset=None, mode=None):
        try:
            return self.config[model][dataset][mode]
        except Exception:
            model = model if model in self.models else self.models[0]
            dataset = dataset if dataset in self.datasets else self.datasets[0]
            mode = mode if mode in self.modes else self.modes[0]
            print(f'Cannot get configs. Switch to default configs: {model} {dataset} {mode}.')
            return self.config[model][dataset][mode]


class TimestepWeight(Layer):
    def __init__(self, att_len=1, embed_reg=1e-6, **kwargs):
        super(TimestepWeight, self).__init__(**kwargs)
        self.w = self.add_weight(
            name='timestep_scale_weight',
            shape=(att_len, ), initializer="random_normal",
            trainable=True,
            )

        self.att_len = att_len
        self.embed_reg = embed_reg

    def call(self, inputs, mask, **kwargs):
        # inputs: (None, att_len, 1(000), embed_dim)
        # w: (att_len, 1, 1)
        # mask: # (None, att_len, 1)

        self.add_loss(self.embed_reg * tf.nn.l2_loss(self.w))
        act_w = tf.nn.softmax(self.w)  # (1, att_len)
        w = tf.reshape(act_w, [self.att_len, 1])  # (att_len, 1)

        # normalize the weights
        mask_w = tf.multiply(w, mask)  # (None, att_len, 1)
        mask_sum = 1 / tf.reduce_sum(mask_w, axis=1, keepdims=True)  # (None, 1, 1)
        n_mask_w = tf.multiply(mask_w, mask_sum)  # (None, att_len, 1)
        n_mask_w = tf.expand_dims(n_mask_w, axis=-1)  # (None, att_len, 1, 1)
        output = tf.multiply(inputs, n_mask_w)  # (None, att_len, 1, embed_dim)

        return output

    def get_config(self):
        # return config
        base_config = super(TimestepWeight, self).get_config()
        return {**base_config,
                "att_len": self.att_len,
                'embed_reg': self.embed_reg,}


class ItemLoss(Layer):
    def __init__(self, from_logits=True, **kwargs):
        super(ItemLoss, self).__init__(**kwargs)
        self.from_logits = from_logits

    def call(self, inputs, **kwargs):
        # from logits
        pos_pred, neg_pred = inputs
        pos_true = K.ones_like(pos_pred)
        pos_loss = K.binary_crossentropy(pos_true, pos_pred, from_logits=self.from_logits)
        pos_loss = K.mean(pos_loss)

        neg_true = K.zeros_like(neg_pred)
        neg_loss = K.binary_crossentropy(neg_true, neg_pred, from_logits=self.from_logits)
        neg_loss = K.mean(neg_loss)

        loss = (pos_loss + neg_loss) / 2
        self.add_loss(loss)
        self.add_metric(loss, aggregation="mean", name="itemloss")
        return(loss)


class IntentionLoss(Layer):
    def __init__(self, from_logits=True, **kwargs):
        super(IntentionLoss, self).__init__(**kwargs)
        self.from_logits = from_logits

    def call(self, inputs, **kwargs):
        # from logits
        pos_pred, neg_pred, pos_inputs, neg_inputs = inputs
        pos_true = K.ones_like(pos_pred)
        pos_loss = K.binary_crossentropy(pos_true, pos_pred, from_logits=self.from_logits)
        pos_loss = K.mean(pos_loss)

        neg_true = tf.cast(tf.where(tf.equal(pos_inputs, neg_inputs), 1, 0), tf.float32)
        neg_loss = K.binary_crossentropy(neg_true, neg_pred, from_logits=self.from_logits)
        neg_loss = K.mean(neg_loss)

        loss = (pos_loss + neg_loss) / 2
        self.add_loss(loss)
        self.add_metric(loss, aggregation="mean", name="intentionloss")
        return(loss)


class RecordLoss(tf.keras.callbacks.Callback):
    """ Report loss when epoch ends and record loss when batch ends.

    Args:
        my_loss: dict, to store loss information
        mode: str, 'Batch' for record training loss in each batch;
                   'Epoch' for record training and validation loss in each epoch

    """

    def __init__(self, my_loss, mode='epoch'):
        super(RecordLoss, self).__init__()
        self.my_loss = my_loss
        self.my_loss['Epoch'] = []
        self.my_loss['Batch'] = []
        self.my_loss['Fitting Time'] = []
        self.my_loss['Batch Fitting Time'] = []
        self.my_epoch = 0
        self.my_batch = 0
        self.mode = mode.lower()
        self.epoch_begin = None

    def on_epoch_begin(self, epoch, logs={}):
        self.my_epoch = epoch
        self.epoch_begin = time.time()  # get epoch train begin time

    def on_epoch_end(self, epoch, logs={}):
        elapsed_time = time.time() - self.epoch_begin
        loss = logs['loss']
        val_loss = logs['val_loss']
        print(f'Callback @ epoch {self.my_epoch + 1}: loss = {loss:.4f}, val_loss = {val_loss:.4f}, fitting_time = {elapsed_time/60:.2f}min')

        if self.mode == 'epoch':
            self.my_loss['Epoch'].append(epoch)
            self.my_loss['Batch'].append(0)
            self.my_loss['Fitting Time'].append(elapsed_time)
            self.my_loss['Batch Fitting Time'].append(0)
            # record loss
            for key in logs.keys():
                self.my_loss[key].append(logs[key])
        elif self.mode == 'batch':
            for key in logs.keys():
                if 'val' in key:
                    for i in range(self.my_batch + 1):
                        self.my_loss[key].append(logs[key])

            for i in range(self.my_batch + 1):
                self.my_loss['Fitting Time'].append(elapsed_time)

    def on_train_batch_begin(self, batch, logs={}):
        self.batch_begin = time.time()

    def on_train_batch_end(self, batch, logs={}):
        elapsed_time = time.time() - self.batch_begin
        self.my_batch = batch
        if self.mode == 'batch':
            self.my_loss['Epoch'].append(self.my_epoch)
            self.my_loss['Batch'].append(batch)
            self.my_loss['Batch Fitting Time'].append(elapsed_time)
            # record loss
            for key in logs.keys():
                if key in self.my_loss.keys():
                    self.my_loss[key].append(logs[key])
                else:
                    self.my_loss[key] = [logs[key]]


class RecordMetrics(tf.keras.callbacks.Callback):
    def __init__(self, data, my_log, interval=1, K=[10]):
        super(RecordMetrics, self).__init__()
        self.data = data
        self.my_log = my_log
        self.my_log['Epoch'] = []
        self.my_log['Evaluation Time'] = []
        self.interval = interval
        self.K = K

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.interval == 0:
            eval_begin = time.time()
            y_pred = self.model.predict(self.data, verbose=0)
            rank = get_rank(y_pred)
            metrics = {'HR': hr, 'NDCG': ndcg, 'MAP': ap}
            for m in metrics:
                for k in self.K:
                    name = f'{m}@{k}'
                    score = metrics[m](rank, k)
                    if name in self.my_log:
                        self.my_log[name].append(score)
                    else:
                        self.my_log[name] = [score]
                    print(f'Callback @ epoch {epoch+1}: {name} = {score:.4f}')

            name = 'MRR'
            score = mrr(rank)
            if name in self.my_log:
                self.my_log[name].append(score)
            else:
                self.my_log[name] = [score]
            print(f'Callback @ epoch {epoch+1}: {name} = {score:.4f}')

            elapsed_time = (time.time() - eval_begin)  # trans to minute
            self.my_log['Epoch'].append(epoch)
            self.my_log['Evaluation Time'].append(elapsed_time)
            print(f'Callback @ epoch {epoch+1}: Evaluation time = {elapsed_time/60:.2f} min')


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


class MRR(tf.keras.metrics.Metric):
    def __init__(self, name='MRR', **kwargs):
        super(MRR, self).__init__(name=name, **kwargs)
        self.rr = self.add_weight(name='rr', dtype=tf.float32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        rank = tf.argsort(tf.argsort(y_pred))[:, 0]
        rank = y_pred.shape[1] - 1 - rank
        rank = tf.cast(rank, tf.float32)

        self.count.assign_add(tf.shape(y_pred)[0])
        for r in rank:
            self.rr.assign_add(1 / (r + 1))

    def result(self):
        count = tf.cast(self.count, tf.float32)
        return self.rr / count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'name': self.name}


class HR(tf.keras.metrics.Metric):
    def __init__(self, K=10, name='HR', **kwargs):
        super(HR, self).__init__(name=name, **kwargs)
        self.hits = self.add_weight(name='hits', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.K = K

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)  # argsort默认从小到大排序，故取负
        rank = tf.argsort(tf.argsort(y_pred))[:, 0]
        rank = y_pred.shape[1] - 1 - rank
        # tf.print(rank)

        self.count.assign_add(tf.shape(y_pred)[0])
        for r in rank:
            if r < self.K:
                self.hits.assign_add(1)

    def result(self):
        return self.hits / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "K": self.K,
                'name': self.name}


class NDCG(tf.keras.metrics.Metric):
    def __init__(self, K=10, name='NDCG', **kwargs):
        super(NDCG, self).__init__(name=name, **kwargs)
        self.ndcg = self.add_weight(name='ndcg', dtype=tf.float32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.float32, initializer=tf.zeros_initializer())
        self.K = K

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)  # argsort默认从小到大排序，故取负
        rank = tf.argsort(tf.argsort(y_pred))[:, 0]
        rank = y_pred.shape[1] - 1 - rank
        self.count.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))
        for r in rank:
            if r < self.K:
                gain = 1 / tf.experimental.numpy.log2(r + 2)
                self.ndcg.assign_add(gain)

    def result(self):
        return self.ndcg / self.count

    def reset_states(self):
        self.ndcg.assign(0)
        self.count.assign(0)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "K": self.K,
                'name': self.name}

def get_rank(y_pred):
    rank = y_pred.argsort(kind='stable').argsort()[:, 0]
    rank = y_pred.shape[1] - 1 - rank
    return rank

def hr(rank, K):
    """ Calculate Hit Rate @ K.

    Args:
        rank: An array
        K: int

    Returns:
        hr: float

    """
    counts = rank.shape[0]
    hits = 0
    for r in rank:
        if r < K:
            hits += 1
    hits / counts

    return hits / counts

def ndcg(rank, K):
    """ Calculate NDCG @ K.

    Args:
        y_pred: A 2D tensor
        K: int

    Returns:
        hr: float

    """
    counts = rank.shape[0]
    ndcg = 0
    gain = 0
    for r in rank:
        if r < K:
            gain = 1 / np.log2(r + 2)
            ndcg += gain
    return ndcg / counts

def ap(rank, K):
    counts = rank.shape[0]  # 用户数量
    ap = 0
    for r in rank:
        if r < K:
            ap += 1 / (r + 1)
    return ap / counts

def mrr(rank):
    counts = rank.shape[0]
    rr = 0
    for r in rank:
        rr += 1 / (r + 1)
    return rr / counts


def summary(model):

    inputs = {
        'user_id': Input(shape=(1, ), dtype=tf.int32, name='user_id'),
        'item_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='item_seq'),
        'cate_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='cate_seq'),
        'type_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='type_seq'),
        'intention_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='intention_seq'),
        'next_item_pos_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='next_item_pos_seq'),
        'next_item_neg_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='next_item_neg_seq'),
        'next_cate_pos_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='next_cate_pos_seq'),
        'next_cate_neg_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='next_cate_neg_seq'),
        'next_type_pos_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='next_type_pos_seq'),
        'next_type_neg_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='next_type_neg_seq'),
        'next_intention_pos_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='next_intention_pos_seq'),
        'next_intention_neg_seq': Input(shape=(model.maxlen,), dtype=tf.int32, name='next_intention_neg_seq'),
        'target_item_pos': Input(shape=(1, ), dtype=tf.int32, name='target_item_pos'),
        'target_item_neg': Input(shape=(1, ), dtype=tf.int32, name='target_item_neg'),
        'target_cate_pos': Input(shape=(1, ), dtype=tf.int32, name='target_cate_pos'),
        'target_cate_neg': Input(shape=(1, ), dtype=tf.int32, name='target_cate_neg'),
        'target_type_pos': Input(shape=(1, ), dtype=tf.int32, name='target_type_pos'),
        'target_type_neg': Input(shape=(1, ), dtype=tf.int32, name='target_type_neg'),
        'target_intention_pos': Input(shape=(1, ), dtype=tf.int32, name='target_intention_pos'),
        'target_intention_neg': Input(shape=(1, ), dtype=tf.int32, name='target_intention_neg'),
              }

    Model(inputs=inputs,
          outputs=model.call(inputs)).summary()


features = {}
for col in ['user_id', 'item', 'type', 'cate', 'intention']:
    features[col] = {'feat_num': 10, 'embed_dim': 8}
features['maxlen'] = 30
