"""
Created on January 4

@author: Shiying Ni
"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import pickle
import os

random.seed(17)  # for reproductivity


def get_dataset_params(dataset, mode):

    func_params = {}
    func = None

    if dataset == 'taobao':
        func_params['data_dir'] = './data/taobao/'
        if mode == 'train_500K':
            func_params['action_file_name'] = 'sample_action.csv'
            func_params['limit'] = 1626000
        # [begin, end)
        for key, value in zip(('month_begin', 'day_begin', 'month_end', 'day_end'), (11, 25, 12, 4)):
            func_params[key] = value
        func = prepare_tb_dataset

    return func_params, func

def prepare_dataset(data, cold_start, item_intention, re_encode_cols):
    item_features = {}
    data = data.copy()

    print('Filter Cold users and items'.center(50, '='))
    data_target = data.copy()
    print(f'Total records: {len(data_target)}')

    while True:
        before = len(data_target)
        # delete items whose # interactions < cold_start
        data_target['users_per_item'] = data_target.groupby(['item'])['user_id'].transform('count')
        data_target = data_target[data_target['users_per_item'] >= cold_start]

        # delete users whose # interactions < cold_start
        data_target['items_per_user'] = data_target.groupby(['user_id'])['item'].transform('count')
        data_target = data_target[data_target['items_per_user'] >= cold_start]
        after = len(data_target)

        if before == after:
            break

    print(f'{cold_start}-core records of actions:', len(data_target))
    core_user = list(data_target['user_id'].unique())
    core_item = list(data_target['item'].unique())
    data = data[(data['user_id'].isin(core_user)) & (data['item'].isin(core_item))].copy()

    print('Re-encoding Features'.center(50, '='))
    for col in re_encode_cols:
        old_codes = data[col].unique().tolist()
        if (min(old_codes) == 1) & (max(old_codes) == len(old_codes)):
            print('No need of encoding for {}: min={}, max={}'.format(col, data[col].min(), data[col].max()))
        else:
            new_codes = list(range(1, len(old_codes) + 1))
            old_col = f'old_{col}'
            data.rename(columns={col: old_col}, inplace=True)
            codes_map = pd.DataFrame({old_col: old_codes, col: new_codes})
            data = data.merge(codes_map)
            data.drop(old_col, axis=1, inplace=True)
            print('Encoding for {}: min={}, max={}'.format(col, data[col].min(), data[col].max()))

    if item_intention:
        data['cate'] = data['item']
        print('Use item as cate...')

    col = 'intention'
    n_type = data['type'].max()
    data.loc[:, 'intention'] = (data['cate'] - 1) * n_type + data['type']
    print(f'Use cate * type as intention: min={data[col].min()}, max={data[col].max()}')

    # item-cate lookup table
    for item_feature_col in ['cate']:
        df = data[['item', item_feature_col]].drop_duplicates()
        df = df.drop_duplicates('item', keep='first')
        df = df.sort_values('item')
        if df['item'].max()==len(df):
            item_feature_dict = {}
            for key, value in zip(df['item'], df[item_feature_col]):
                item_feature_dict[key] = int(value)
            item_feature_dict[0] = 0
            item_features[item_feature_col] = item_feature_dict
        else:
            print('Encoding out of range for {}.'.format(item_feature_col))
    print('Preprocessing Done'.center(50, '='))
    return data, item_features


def prepare_tb_dataset(data_dir, action_file_name, cold_start, month_begin, day_begin, month_end, day_end, limit, item_intention):
    print('Data Preprocessing'.center(50, '='))

    data = pd.read_csv(
        data_dir + action_file_name,
        header=0,
        names=['user_id', 'item', 'cate', 'type_name', 'timestamp'],
        )
    print(f'Read {len(data)} raw data...')

    # drop null
    data = data.dropna(how='any', axis=0)

    print('Selecting data within time range...')
    data['action_time'] = pd.to_datetime(data['timestamp'], unit='s')
    data['action_time'] = pd.to_datetime(data['action_time'])
    date_begin = datetime(2017, month_begin, day_begin)
    date_end = datetime(2017, month_end, day_end)
    data = data[(data['action_time'] > date_begin) & (data['action_time'] < date_end)]
    print(f'{len(data)} records within time range')

    print('Re-encoding type...')
    type_map = pd.DataFrame({
        'type_name': ['pv', 'buy', 'cart', 'fav'],
        'type': [1, 2, 3, 4],
        })
    data = data.merge(type_map, how='inner')
    data.drop(['type_name'], axis=1, inplace=True)

    data = data[data['type'] <= 4]

    # truncate data for a certain scale
    data = data.sort_values(by=['action_time'])
    if limit < len(data):
        print(f'Total records: {len(data)}, limit: {limit}')
    else:
        print('No limit in records')

    data = data[:limit]

    re_encode_cols = ['user_id', 'item', 'cate', 'type']
    df, item_features = prepare_dataset(data, cold_start, item_intention, re_encode_cols)

    return df, item_features



def create_dataset(
    data_df, item_features,
    embed_dim=32, maxlen=20, train_neg_ratio=1, test_neg_ratio=100,
    recurrent=True, rand_neg_intention=False, with_ts=False,
    ):

    print('Create Dataset'.center(50, '='))

    data_df = data_df.sort_values(by=['user_id', 'action_time'])

    summary = {'actions_multi_act': len(data_df),
               'actions_single_act': len(data_df[data_df['type']==1])}

    for col in ['item', 'cate', 'user_id', 'intention']:
        summary[col] = data_df[col].max()

    cate_list = item_features['cate']
    item_id_max = data_df['item'].max()
    n_cate = data_df['cate'].max()
    n_type = data_df['type'].max()
    intention_id_max = n_cate * n_type
    hist_cols = ['item', 'intention']
    next_cols = []
    if not recurrent:
        print('Generating next sequence...')
        for col in hist_cols:
            next_col = f'next_{col}_pos'
            data_df.loc[:, next_col] = data_df.groupby('user_id')[col].shift(-1)
            data_df.loc[:, next_col] = data_df[next_col].fillna(0)
            data_df.loc[:, next_col] = data_df[next_col].astype(int)
            next_cols.append(next_col)
        hist_cols.extend(next_cols)

    if with_ts:
        hist_cols.append('timestamp')

    train_data, val_data, test_data = [], [], []

    # generate negative samples from the items which users have not interacted with currently
    def gen_neg_samples(pos_list, max_id):

        def gen_neg_item(i):
            neg = pos_list[0]
            while neg in pos_list[:i + 1]:
                neg = random.randint(1, max_id)
            return neg

        def gen_train_neg_items(ratio):
            neg_list = []
            seq_len = len(pos_list)
            for i in range(seq_len - 2):
                for n in range(ratio):
                    neg_list.append(gen_neg_item(i))
            return neg_list

        def gen_val_neg_items(ratio):
            neg_list = []
            seq_len = len(pos_list)
            for n in range(ratio):
                neg_list.append(gen_neg_item(seq_len - 2))
            return neg_list

        def gen_test_neg_items(ratio):
            neg_list = []
            seq_len = len(pos_list)
            for n in range(ratio):
                neg_list.append(gen_neg_item(seq_len - 1))
            return neg_list

        train_neg_list = gen_train_neg_items(train_neg_ratio)
        val_neg_list = gen_val_neg_items(test_neg_ratio)
        test_neg_list = gen_test_neg_items(test_neg_ratio)

        return train_neg_list, val_neg_list, test_neg_list

    for user_id, df_raw in tqdm(data_df[['user_id', 'type']+hist_cols].groupby('user_id')):
        df = df_raw[-maxlen:]  # the max sequence length

        pos_item_list = df['item'].tolist()
        pos_type_list = df['type'].tolist()
        pos_intention_list = df['intention'].tolist()

        if ((np.array(pos_type_list)==1).sum() < 4):  # only keep the users with >=3 valid target timesteps
            continue

        hist_list = {}

        for col in hist_cols:
            hist_list[col] = df[col].tolist()

        train_neg_list, val_neg_list, test_neg_list = gen_neg_samples(pos_item_list, item_id_max)
        train_neg_intention_list, _, _ = gen_neg_samples(pos_intention_list, intention_id_max)  # placeholder to make sure the random generators are the same for different dataset settings
        if not recurrent:
            if not rand_neg_intention:
                train_neg_intention_list = [(cate_list[item] - 1) * n_type + 1 for item in train_neg_list]

        flag = 0  # flag=0 -> test; flag=1 -> valid; flag=2: train (not recurrent); flag>=2: train (recurrent);
        for i in range(1, len(pos_item_list))[::-1]:  # from back to the front

            # skip if it is not the target action
            if pos_type_list[i] != 1:
                continue

            hist_i = []

            for key, hist in hist_list.items():
                hist_i.append(hist[:i])

            if not recurrent:
                next_item_neg = train_neg_list[:i]
                next_intention_neg = train_neg_intention_list[:i]
                hist_i.extend([next_item_neg, next_intention_neg])

            pos_item = [pos_item_list[i]]
            pos_intention = [pos_intention_list[i]]
            target_pos_i = [pos_item, pos_intention]

            neg_item, neg_intention = [], []

            # generate test/validation/training datasets
            if flag == 0:  # test: neg:pos = test_neg_ratio
                neg_item = test_neg_list[:]
                neg_intention = [(cate_list[item] - 1) * n_type + 1 for item in neg_item]
                test_data.append([user_id] + hist_i + target_pos_i + [neg_item, neg_intention])

            elif flag == 1:  # validation: neg:pos = test_neg_ratio
                neg_item = val_neg_list[:]
                neg_intention = [(cate_list[item] - 1) * n_type + 1 for item in neg_item]
                val_data.append([user_id] + hist_i + target_pos_i + [neg_item, neg_intention])

            else:  # train: neg:pos = train_neg_ratio
                neg_item = train_neg_list[train_neg_ratio*i: train_neg_ratio*(i+1)]
                neg_intention = [(cate_list[item] - 1) * n_type + 1 for item in neg_item]
                train_data.append([user_id] + hist_i + target_pos_i + [neg_item, neg_intention])

                if not recurrent:
                    break

            flag += 1

    print('Data Info'.center(50, '='))
    data_info = {}
    for col in ['user_id', 'item', 'cate', 'type', 'intention']:
        if col == 'intention':
            feat_num = max(data_df['cate'].max() * data_df['type'].max() + 1, data_df[col].max() + 1)
        else:
            feat_num = data_df[col].max() + 1
        data_info[col] = {'feat_num': feat_num, 'embed_dim': embed_dim}
    data_info['maxlen'] = maxlen

    if not recurrent:
        hist_cols.extend(['next_item_neg', 'next_intention_neg'])

    target_cols = [
        'target_item_pos', 'target_intention_pos',
        'target_item_neg', 'target_intention_neg',
    ]

    data_info['column_names'] = ['user_id'] + [i + '_seq' for i in hist_cols] + target_cols

    # shuffle
    print('Shuffle Data'.center(50, '='))
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    column_names = data_info['column_names']
    train = pd.DataFrame(train_data, columns=column_names)
    val = pd.DataFrame(val_data, columns=column_names)
    test = pd.DataFrame(test_data, columns=column_names)
    del train_data, val_data, test_data

    # pad and format
    print('Pad Data'.center(50, '='))

    res = []
    for data in [train, test, val]:
        f_data = {'user_id': data['user_id'].values.astype(np.int32).reshape(-1,1)}

        for i in hist_cols:
            f_data[i+'_seq'] = pad_sequences(data[i+'_seq'], maxlen=maxlen).astype(np.int32)

        for i in target_cols:
            f_data[i] = np.array(list(data[i])).astype(np.int32)

        res.append(f_data)

    train, test, val = res

    print('Dataset Done'.center(50, '='))
    return data_info, train, val, test, summary


def get_data(dataset='taobao', mode='debug', regenerate=False,
             embed_dim=32, maxlen=20,
             train_neg_ratio=1, test_neg_ratio=100, cold_start=5,
             item_intention=False, rand_neg_intention=False,
             recurrent=True, with_ts=False,
             ):

    print('Get Data'.center(50, '='))
    train_neg_ratio = train_neg_ratio
    test_neg_ratio = test_neg_ratio
    cold_start = cold_start
    item_intention = item_intention
    rand_neg_intention = rand_neg_intention
    recurrent = recurrent
    with_ts = with_ts

    print(f'Dataset: {dataset}')
    print(f'Mode: {mode}')
    print(f'Train negative ratio = {train_neg_ratio}')
    print(f'Val & Test negative ratio = {test_neg_ratio}')
    print(f'Cold start = {cold_start}')
    print(f'Max length = {maxlen}')
    print(f'Embedding Dimension = {embed_dim}')

    params, prepare_dataset_func = get_dataset_params(dataset, mode)
    data_dir = params['data_dir']
    addition = ''
    if 'trans_score' in params:
        trans_score = params['trans_score']
        addition += f'_trans_score_{trans_score}'
        print(f'Trans Score = {trans_score}')

    if item_intention:
        addition += '_item_intention'
    if rand_neg_intention:
        addition += '_rand_neg_intention'
    if recurrent:
        addition += '_recurrent'
    if with_ts:
        addition += '_with_ts'

    # try to load date from pickle
    pickle_filename = f'{data_dir}{dataset}_{mode}_maxlen_{maxlen}' + \
                      f'_train_neg_{train_neg_ratio}_test_neg_{test_neg_ratio}' + \
                      f'_{cold_start}-core{addition}.pkl'

    if (regenerate is False) & (os.path.exists(pickle_filename)):
        print('Loading Data'.center(50, '='))
        with open(pickle_filename, 'rb') as file:
            data_info = pickle.load(file)
            train = pickle.load(file)
            val = pickle.load(file)
            test = pickle.load(file)
            summary = pickle.load(file)
            print('Load data from pickle: {}'.format(pickle_filename))
            print('Data Summary'.center(50, '='))
            for key, value in summary.items():
                print(f'{key}: {value}')
            for key, value in data_info.items():
                if isinstance(value, dict):
                    if 'embed_dim' in value:
                        if value['embed_dim'] != embed_dim:
                            value['embed_dim'] = embed_dim
                            print(f'Reset embed_dim as {embed_dim} for column {key}.')

    else:
        print('Generating Data'.center(50, '='))

        params['cold_start'] = cold_start
        params['item_intention'] = item_intention

        # prepare dataset
        data, item_features = prepare_dataset_func(**params)
        # creat train, val, test
        data_info, train, val, test, summary = create_dataset(
            data, item_features,
            embed_dim=embed_dim, maxlen=maxlen,
            train_neg_ratio=train_neg_ratio, test_neg_ratio=test_neg_ratio,
            recurrent=recurrent,
            rand_neg_intention=rand_neg_intention,
            with_ts=with_ts,
            )

        print('Data Summary'.center(50, '='))
        for key, value in summary.items():
            print(f'{key}: {value}')
        with open(pickle_filename, 'wb') as file:
            pickle.dump(data_info, file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(train, file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(val, file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test, file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(summary, file, pickle.HIGHEST_PROTOCOL)
            print('Dump data to {}'.format(pickle_filename))
    print('Data Done'.center(50, '='))
    return(data_info, train, val, test)


if(__name__ == '__main__'):


    data_info, train, val, test = get_data(
        dataset='taobao', mode='train_500K',
        regenerate=False, embed_dim=100, maxlen=70, cold_start=5,
        train_neg_ratio=1, test_neg_ratio=3000, recurrent=True,
        item_intention=False, rand_neg_intention=False, with_ts=True,
        )

    print('Len of train, val and test:')
    for i in train, val, test:
        print(i['user_id'].shape)

    print('Shape of data in test')
    for c, i in test.items():
        print(f'{c}: {i.shape}, min={i.min()}, max={i.max()}')
