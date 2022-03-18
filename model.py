# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:13:30 2021

@author: Shiying Ni
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Input
from module import ItemLoss
from module import IntentionLoss
from module import TimestepWeight


class IMRec(Model):

    def __init__(self, feature_columns={}, config={}):

        super(IMRec, self).__init__()
        self.model_name = 'IMRec'

        for param in config:
            setattr(self, param, config[param])

        # feature columns
        self.user_fea_col = feature_columns['user_id']
        self.item_fea_col = feature_columns['item_id']
        self.type_fea_col = feature_columns['type']
        self.intention_fea_col = feature_columns['intention']
        self.column_names = feature_columns['column_names']
        self.embed_dim = self.item_fea_col['embed_dim']

        self.user_embedding = Embedding(
            input_dim=self.user_fea_col['feat_num'],
            input_length=1,
            output_dim=self.embed_dim,
            mask_zero=False,
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(self.embed_reg),
            name='user_embedding')

        self.item_embedding = Embedding(
            input_dim=self.item_fea_col['feat_num'],
            input_length=1,
            output_dim=self.embed_dim,
            mask_zero=True,
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(self.embed_reg),
            name='item_embedding')

        self.from_intention_embedding = Embedding(
            input_dim=self.intention_fea_col['feat_num'],
            input_length=1,
            output_dim=self.embed_dim,
            mask_zero=True,
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(self.embed_reg),
            name='from_intention_seq_embedding')

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

        self.itemloss = ItemLoss(input_dim=None, name='item_loss_layer')
        self.intentionloss = IntentionLoss(input_dim=None, name='intention_loss_layer')

    def call(self, inputs):

        (user_inputs, item_seq_inputs, type_seq_inputs, intention_seq_inputs,
         next_item_inputs, next_item_neg_inputs,
         target_item_pos_inputs, target_intention_pos_inputs,
         target_item_neg_inputs, target_intention_neg_inputs) = (inputs[col] for col in self.column_names)

        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))  # (None, dim)
        user_embed = tf.expand_dims(user_embed, axis=1)  # (None, 1, dim)

        mask = tf.where(tf.equal(item_seq_inputs, 0), 0, 1)  # (None, att_len)
        mask = mask[:, -self.att_len:][:, ::-1]  # (None, att_len)
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)  # (None, att_len, 1)

        def cal_intention_score(target_intention_inputs):

            to_intention_embed = self.to_intention_embedding(target_intention_inputs)  # (None, 1(000), dim)

            seq_len = intention_seq_inputs.shape[1]
            from_intention_list = list()

            for i in range(self.att_len):
                from_intention_embed = self.from_intention_embedding(intention_seq_inputs[:, seq_len-i-1:seq_len-i])  # (None, 1, dim)
                from_intention_list.append(from_intention_embed)

            from_intention = tf.stack(from_intention_list, axis=1)  # (None, att_len, 1, dim)

            # mtd
            from_intention = self.timestep_scale(
                inputs=from_intention,
                mask=mask,
                )
            from_intention = tf.reduce_sum(from_intention, axis=1)

            intention_intensity = tf.reduce_sum(tf.multiply(from_intention, to_intention_embed), axis=-1, keepdims=False)
            short_term = tf.expand_dims(intention_intensity, axis=-1)  # (None, 1(000), 1)

            return short_term

        def cal_score(target_item_inputs, short_term):
            item_embed = self.item_embedding(target_item_inputs)  # (None, 1(000), dim)

            long_term = tf.reduce_sum(tf.multiply(user_embed, item_embed), axis=-1, keepdims=True)
            alpha = self.alpha

            score = alpha * long_term + (1 - alpha) * short_term

            return score

        pos_intention_score = cal_intention_score(target_intention_pos_inputs)
        neg_intention_score = cal_intention_score(target_intention_neg_inputs)

        intention_loss_inputs = (
            pos_intention_score,
            neg_intention_score,
            tf.expand_dims(target_intention_pos_inputs, axis=-1),
            tf.expand_dims(target_intention_neg_inputs, axis=-1),
            )
        self.intentionloss(intention_loss_inputs)

        pos_score = cal_score(target_item_pos_inputs, pos_intention_score)
        neg_score = cal_score(target_item_neg_inputs, neg_intention_score)

        item_loss_inputs = (pos_score, neg_score)
        self.itemloss(item_loss_inputs)

        # output the probabilities
        pos_prob = tf.nn.sigmoid(pos_score)
        neg_prob = tf.nn.sigmoid(neg_score)
        outputs = tf.concat([tf.squeeze(pos_prob, axis=-1), tf.squeeze(neg_prob, axis=-1)], axis=-1)

        return outputs

    def summary(self):
        item_seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32, name='item_id_seq')
        type_seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32, name='type_seq')
        intention_seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32, name='intention_seq')
        user_inputs = Input(shape=(1, ), dtype=tf.int32, name='user_id')
        next_item_seq_inputs = Input(shape=(1,), dtype=tf.int32, name='next_item_seq')
        next_item_neg_seq_inputs = Input(shape=(1,), dtype=tf.int32, name='next_item_neg_seq')
        target_item_pos_inputs = Input(shape=(1,), dtype=tf.int32, name='target_item_pos')
        target_intention_pos_inputs = Input(shape=(1,), dtype=tf.int32, name='target_intention_pos')
        target_item_neg_inputs = Input(shape=(1,), dtype=tf.int32, name='target_item_neg')
        target_intention_neg_inputs = Input(shape=(1,), dtype=tf.int32, name='target_intention_neg')

        inputs = {
            'user_id': user_inputs,
            'item_id_seq': item_seq_inputs,
            'type_seq': type_seq_inputs,
            'intention_seq': intention_seq_inputs,
            'next_item_seq': next_item_seq_inputs,
            'next_item_neg_seq': next_item_neg_seq_inputs,
            'target_item_pos': target_item_pos_inputs,
            'target_intention_pos': target_intention_pos_inputs,
            'target_item_neg': target_item_neg_inputs,
            'target_intention_neg': target_intention_neg_inputs,
                  }

        Model(inputs=inputs,
              outputs=self.call(inputs)).summary()


def test_model():
    features = {}
    for col in ['user_id', 'item_id', 'type', 'intention', 'time_interval']:
        features[col] = {'feat': col, 'feat_num': 10, 'embed_dim': 8}
    features['column_names'] = [
        'user_id', 'item_id_seq', 'type_seq', 'intention_seq',
        'next_item_seq', 'next_item_neg_seq',
        'target_item_pos', 'target_intention_pos',
        'target_item_neg', 'target_intention_neg']

    config = dict(
        maxlen=30,
        activation='relu',
        embed_reg=1e-6,
        alpha=0.8,
        att_len=5,
        )

    model = IMRec(features, config)
    model.summary()


if __name__ == '__main__':
    test_model()
