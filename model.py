# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:13:30 2021

@author: Shiying
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding
from module import ItemLoss
from module import IntentionLoss
from module import TimestepWeight
from module import summary, features


class IMRec(Model):

    def __init__(self, data_info, config={}):

        super(IMRec, self).__init__()
        self.model_name = 'IMRec'

        for param in config:
            setattr(self, param, config[param])

        if self.BPR:
            self.model_name += '-BPR'

        if self.without_il:
            self.model_name += '-WOIL'

        if self.time_threshold:
            self.model_name += f'-TS{self.time_threshold}'

        # feature columns
        self.user_fea_col = data_info['user_id']
        self.item_fea_col = data_info['item']
        self.type_fea_col = data_info['type']
        self.intention_fea_col = data_info['intention']
        self.maxlen = data_info['maxlen']
        self.embed_dim = self.item_fea_col['embed_dim']

        if self.alpha != 0:
            # user embedding
            self.user_embedding = Embedding(
                input_dim=self.user_fea_col['feat_num'],
                input_length=1,
                output_dim=self.embed_dim,
                mask_zero=False,  # no need for padding users
                embeddings_initializer='random_normal',
                embeddings_regularizer=l2(self.embed_reg),
                name='user_embedding')

            # item embedding
            self.item_embedding = Embedding(
                input_dim=self.item_fea_col['feat_num'],
                input_length=1,
                output_dim=self.embed_dim,
                mask_zero=True,  # item padding
                embeddings_initializer='random_normal',
                embeddings_regularizer=l2(self.embed_reg),
                name='item_embedding')
        if not self.BPR:
            self.itemloss = ItemLoss(input_dim=None, name='item_loss_layer', from_logits=True)

        if self.alpha != 1:
            # intention embedding (from)
            self.from_intention_embedding = Embedding(
                input_dim=self.intention_fea_col['feat_num'],
                input_length=1,
                output_dim=self.embed_dim,
                mask_zero=True,
                embeddings_initializer='random_normal',
                embeddings_regularizer=l2(self.embed_reg),
                name='from_intention_seq_embedding')

            # intention embedding (to)
            self.to_intention_embedding = Embedding(
                input_dim=self.intention_fea_col['feat_num'],
                input_length=1,
                output_dim=self.embed_dim,
                mask_zero=True,
                embeddings_initializer='random_normal',
                embeddings_regularizer=l2(self.embed_reg),
                name='to_intention_seq_embedding')

            self.timestep_scale = TimestepWeight(
                att_len=self.att_len,
                embed_reg=self.embed_reg,
                name='timestep_weight')

            if not self.BPR:
                self.intentionloss = IntentionLoss(input_dim=None, name='intention_loss_layer', from_logits=True)

    def call(self, inputs):
        user_inputs, item_seq_inputs = inputs['user_id'], inputs['item_seq']
        target_item_pos_inputs, target_item_neg_inputs = inputs['target_item_pos'], inputs['target_item_neg']
        intention_seq_inputs = inputs['intention_seq']
        target_intention_pos_inputs, target_intention_neg_inputs = inputs['target_intention_pos'], inputs['target_intention_neg']

        if self.alpha != 0:
            user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
            user_embed = tf.expand_dims(user_embed, axis=1)  # (None, 1, dim)

        if self.time_threshold:
            timestamp_inputs = inputs['timestamp_seq']
            time_interval = timestamp_inputs[:, -1:] - timestamp_inputs
            mask = tf.where(time_interval < self.time_threshold, 1, 0)  # (None, att_len)
        else:
            mask = tf.where(tf.equal(item_seq_inputs, 0), 0, 1)  # (None, att_len)

        mask = mask[:, -self.att_len:][:, ::-1]  # (None, att_len)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)  # (None, att_len, 1)

        def cal_intention_score(target_intention_inputs):

            to_intention_embed = self.to_intention_embedding(target_intention_inputs)  # (None, 1(000), dim)
            seq_len = intention_seq_inputs.shape[1]
            from_intention_list = list()

            for i in range(self.att_len):
                # descending order: (last 1, last 2, last 3, ...)
                from_intention_embed = self.from_intention_embedding(intention_seq_inputs[:, seq_len-i-1:seq_len-i]) # (None, 1, dim)
                from_intention_list.append(from_intention_embed)

            # merge from-intention metrices
            from_intention = tf.stack(from_intention_list, axis=1)  # (None, att_len, 1, dim)
            # scale on timestep
            from_intention = self.timestep_scale(inputs=from_intention, mask=mask)

            from_intention = tf.reduce_sum(from_intention, axis=1)  # (None, 1, dim)
            short_term = tf.reduce_sum(tf.multiply(from_intention, to_intention_embed), axis=-1, keepdims=True)  # (None, 1(000))

            return short_term

        def cal_score(target_item_inputs, short_term):

            if self.alpha == 0:
                score = short_term
            else:
                item_embed = self.item_embedding(target_item_inputs)  # (None, 1(000), dim)
                long_term = tf.reduce_sum(tf.multiply(user_embed, item_embed), axis=-1, keepdims=True)

                if self.alpha == 1:
                    score = long_term
                else:
                    score = self.alpha * long_term + (1 - self.alpha) * short_term

            return score

        if self.alpha == 1:
            pos_intention_score, neg_intention_score = None, None
        else:
            pos_intention_score = cal_intention_score(target_intention_pos_inputs)
            neg_intention_score = cal_intention_score(target_intention_neg_inputs)

            # intention loss
            if not self.without_il:
                if self.BPR:
                    loss = -tf.math.log(tf.nn.sigmoid(pos_intention_score - neg_intention_score))
                    mask_loss = tf.cast(tf.where(tf.equal(target_intention_pos_inputs, target_intention_neg_inputs), 0, 1), tf.float32)
                    mask_loss = tf.expand_dims(mask_loss, -1)
                    intention_loss = tf.reduce_mean(loss * mask_loss)
                    self.add_loss(intention_loss)
                    self.add_metric(intention_loss, aggregation="mean", name="intentionloss")
                else:
                    intention_loss_inputs = (
                        pos_intention_score,
                        neg_intention_score,
                        tf.expand_dims(target_intention_pos_inputs, axis=-1),
                        tf.expand_dims(target_intention_neg_inputs, axis=-1),
                        )
                    self.intentionloss(intention_loss_inputs)

        pos_score = cal_score(target_item_pos_inputs, pos_intention_score)
        neg_score = cal_score(target_item_neg_inputs, neg_intention_score)

        # item loss
        if self.BPR:
            item_loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_score - neg_score)))
            self.add_loss(item_loss)
            self.add_metric(item_loss, aggregation="mean", name="itemloss")
        else:
            item_loss_inputs = (pos_score, neg_score)
            self.itemloss(item_loss_inputs) # calculate loss from logit

        outputs = tf.concat([tf.squeeze(pos_score, axis=-1), tf.squeeze(neg_score, axis=-1)], axis=-1)

        return outputs

    def summary(self):
        summary(self)


def test_model():

    config = dict(
        activation='relu',
        embed_reg=1e-6,
        alpha=0.8,
        att_len=5,
        BPR=True,
        without_il=True,
        time_threshold=0,
        )

    model = IMRec(features, config)
    model.summary()


if __name__ == '__main__':
    test_model()
