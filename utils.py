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

random.seed(17)  # set seed


def get_dataset_params(dataset, mode):
    """ Get params for prepare_dataset by dataset and mode

    Args:
        dataset: str
        mode: str

    Returns:
        func_params: dict

    """
    func_params = {}
    func = None

    if dataset == 'taobao':
        func_params['data_dir'] = './data/taobao/'
        func_params['limit'] = 1 << 32

        if mode == 'train_500K':
            func_params['action_file_name'] = 'sample_action.csv'
            func_params['limit'] = 1096000  # trunc records

        # set data time range
        for key, value in zip(('month_begin', 'day_begin', 'month_end', 'day_end'), (11, 25, 12, 2)):
            func_params[key] = value

        func = preprocess_tb_data

    return func_params, func


def preprocess_tb_data(data_dir, action_file_name, cold_start, month_begin, day_begin, month_end, day_end, limit):
    print('Data Preprocessing'.center(50, '='))

    data = pd.read_csv(
        data_dir + action_file_name,
        header=0,
        names=['user_id', 'item_id', 'cate', 'type_name', 'timestamp'],
        )
    print(f'Read {len(data)} raw data...')



    print('Selecting data within time range...')
    data['action_time'] = pd.to_datetime(data['timestamp'], unit='s')
    date_begin = datetime(2017, month_begin, day_begin)
    date_end = datetime(2017, month_end, day_end)
    data = data[(data['action_time'] > date_begin) & (data['action_time'] < date_end)]

    # drop nan data
    data = data.dropna(how='any', axis=0)
    # drop duplicates
    data = data.drop_duplicates(subset=['user_id', 'item_id', 'type_name', 'timestamp'])

    print('Re-encoding action type...')
    type_map = pd.DataFrame({
        'type_name': ['pv', 'buy', 'cart', 'fav'],
        'type': [1, 2, 3, 4],
        })
    data = data.merge(type_map, how='inner')
    data.drop(['type_name'], axis=1, inplace=True)

    # sort by action time
    data = data.sort_values(by=['action_time'])

    # trunc records
    if limit < len(data):
        print(f'Total records: {len(data)}, limit: {limit}')
        data = data[:limit]

    re_encode_cols = ['user_id', 'item_id', 'cate', 'type']
    df, item_features = prepare_dataset(data, cold_start, re_encode_cols)

    return df, item_features


def prepare_dataset(data, cold_start, re_encode_cols):
    """


    Args:
        data (pd.DataFrame)
        cold_start (int)
        re_encode_cols (list)

    Returns:
        data (pd.DataFrame)
        item_features (dict)

    """

    item_features = {}
    data = data.copy()

    print('Filter cold users and items'.center(50, '='))
    data_target = data.copy()
    print(f'Total records: {len(data_target)}')

    # generate n-core data
    while True:
        before = len(data_target)

        # remove cold items
        data_target['users_per_item'] = data_target.groupby(['item_id'])['user_id'].transform('count')
        data_target = data_target[data_target['users_per_item'] >= cold_start]

        # remove cold users
        data_target['items_per_user'] = data_target.groupby(['user_id'])['item_id'].transform('count')
        data_target = data_target[data_target['items_per_user'] >= cold_start]

        after = len(data_target)

        # if no items and users are removed, break the loop
        if before == after:
            break

    print(f'{cold_start}-core records of actions:', len(data_target))
    core_user = list(data_target['user_id'].unique())
    core_item = list(data_target['item_id'].unique())
    data = data[(data['user_id'].isin(core_user)) & (data['item_id'].isin(core_item))].copy()

    print('Re-encoding Features'.center(50, '='))
    for col in re_encode_cols:
        old_codes = data[col].unique().tolist()
        if (min(old_codes) == 1) & (max(old_codes) == len(old_codes)):
            print('No need of encoding for {}: min={}, max={}'.format(col, data[col].min(), data[col].max()))
        else:
            new_codes = list(range(1, len(old_codes) + 1))  # 0 is helf out for padding
            old_col = f'old_{col}'
            data.rename(columns={col: old_col}, inplace=True)
            codes_map = pd.DataFrame({old_col: old_codes, col: new_codes})
            data = data.merge(codes_map)
            data.drop(old_col, axis=1, inplace=True)
            print('Encoding for {}: min={}, max={}'.format(col, data[col].min(), data[col].max()))


    intention_col = 'cate'
    col = 'intention'

    n_type = data['type'].max()
    data.loc[:, 'intention'] = (data[intention_col] - 1) * n_type + data['type']
    print(f'Use {intention_col} * type as intention: min={data[col].min()}, max={data[col].max()}')

    # gnerate item-cate lookup table
    for item_feature_col in ['cate']:
        df = data[['item_id', item_feature_col]].drop_duplicates()
        df = df.drop_duplicates('item_id', keep='first')
        df = df.sort_values('item_id')
        if df['item_id'].max()==len(df):
            item_feature_dict = {}
            for key, value in zip(df['item_id'], df[item_feature_col]):
                item_feature_dict[key] = int(value)
            item_feature_dict[0] = 0
            item_features[item_feature_col] = item_feature_dict
        else:
            print('Encoding out of range for {}.'.format(item_feature_col))

    print('Generating next item...')

    # generate next_item
    data.loc[:, 'next_item'] = data.groupby('user_id')['item_id'].shift(-1)
    data.loc[:, 'next_item'] = data['next_item'].fillna(0)
    data.loc[:, 'next_item'] = data['next_item'].astype(int)

    print('Preprocessing Done'.center(50, '='))

    return data, item_features


def create_dataset(
    data, item_features,
    embed_dim=32, maxlen=20,
    train_neg_ratio=1, test_neg_ratio=100,
    recurrent=True,
    ):
    """ Create train, validation and test datasets prepared data
    Args:
        data : DataFrame
        item_features : dict
        embed_dim : int
        maxlen : int, max length of user interaction history
        train_neg_ratio: int, the number of negtive samples for on positve sample in train dataset
        test_neg_ratio: int, the number of negtive samples for on positve sample in validation/test datasets
        recurrent: bool, if to generate train dataset recurrently
    Returns:
        feature_columns: dict
        train, val, test: dict of array
    """
    print('Create Dataset'.center(50, '='))

    data_df = data

    summary = {'actions_multi_actions': len(data_df),
               'actions_clicks': len(data_df[data_df['type']==1])}

    for col in ['item_id', 'cate', 'user_id', 'intention']:
        summary[col] = data_df[col].max()

    cate_list = item_features['cate']
    item_id_max = data_df['item_id'].max()
    n_type = data_df['type'].max()

    hist_cols = ['item_id', 'type', 'intention', 'next_item']

    train_data, val_data, test_data = [], [], []

    for user_id, df in tqdm(data_df[['user_id'] + hist_cols].groupby('user_id')):

        pos_list = df['item_id'].tolist()
        pos_type_list = df['type'].tolist()

        hist_list = {}

        # add interaction history
        for col in hist_cols:
            hist_list[col] = df[col].tolist()

        # negtive samples are choosen from items which has not been interacted with currently
        def gen_neg_item(i):
            neg = pos_list[0]
            while neg in pos_list[:i + 1]:
                neg = random.randint(1, item_id_max)
            return neg

        def gen_train_neg_items(ratio):
            neg_list = []
            seq_len = len(pos_list)
            for i in range(seq_len - 1):
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

        # generate train/validation/test dataset from history data
        flag = 0  # flag==0 -> test; flag==1 -> validation; flag==2 -> train(non-recurrent); flag>=2: train(recurrent);
        for i in range(1, len(pos_list))[::-1]:  # from back to front

            # if it is not the target action (click)
            if pos_type_list[i] != 1:
                continue

            hist_i = []

            for key, hist in hist_list.items():
                hist_i.append(hist[:i])

            pos = [pos_list[i]]
            pos_intention = [(cate_list[pos_list[i]] - 1) * n_type + pos_type_list[i]]

            neg = []
            neg_intention = []


            if flag == 0:
                neg = test_neg_list[:]
                neg_cate = [cate_list[item] for item in neg]
                neg_intention = [(cate - 1) * n_type + 1 for cate in neg_cate]
                next_item_neg_seq = train_neg_list[:i]
                test_data.append([user_id] + hist_i + [next_item_neg_seq] + [pos, pos_intention, neg, neg_intention])

            elif flag == 1:
                neg = val_neg_list[:]
                neg_cate = [cate_list[item] for item in neg]
                neg_intention = [(cate - 1) * n_type + 1 for cate in neg_cate]
                next_item_neg_seq = train_neg_list[:i]
                val_data.append([user_id] + hist_i + [next_item_neg_seq] + [pos, pos_intention, neg, neg_intention])

            else:
                neg = train_neg_list[train_neg_ratio*(i-1): train_neg_ratio*i]
                neg_cate = [cate_list[item] for item in neg]
                neg_intention = [(cate - 1) * n_type + 1 for cate in neg_cate]
                next_item_neg_seq = train_neg_list[:i]
                train_data.append([user_id] + hist_i + [next_item_neg_seq] + [pos, pos_intention, neg, neg_intention])

                if not recurrent:
                    break

            flag += 1

    # item feature columns
    feature_columns = {}
    for col in ['user_id', 'item_id', 'cate', 'type', 'intention']:
        if col == 'intention':
            feat_num = max(data_df['cate'].max() * data_df['type'].max() + 1, data_df[col].max() + 1)
        else:
            feat_num = data_df[col].max() + 1
        feature_columns[col] = {'feat': col, 'feat_num': feat_num, 'embed_dim': embed_dim}

    target_cols = [
    'target_item_pos', 'target_intention_pos',
    'target_item_neg', 'target_intention_neg',
    ]

    hist_cols.append('next_item_neg')

    feature_columns['column_names'] = ['user_id'] + [i + '_seq' for i in hist_cols] + target_cols


    # shuffle data
    print('Shuffle Data'.center(50, '='))
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    column_names = feature_columns['column_names']
    train = pd.DataFrame(train_data, columns=column_names)
    val = pd.DataFrame(val_data, columns=column_names)
    test = pd.DataFrame(test_data, columns=column_names)
    del train_data, val_data, test_data

    # pad and format
    print('Pad Data'.center(50, '='))

    res = []
    for data in [train, test, val]:
        f_data = {'user_id': data['user_id'].values}

        for i in hist_cols:
            f_data[i+'_seq'] = pad_sequences(data[i+'_seq'], maxlen=maxlen)

        for i in target_cols:
            f_data[i] = np.array(list(data[i]))

        res.append(f_data)

    train, test, val = res

    print('Dataset Done'.center(50, '='))
    return feature_columns, train, val, test, summary


def get_data(dataset='jd', mode='debug', redump=False,
             embed_dim=32, maxlen=20,
             train_neg_ratio=1, test_neg_ratio=100, cold_start=5,
             recurrent=True,
             ):
    """ Read data from pickle if it exists or generate data

    Args:
        dataset: str, 'taobao' for Taobao dataset
        mode: str
        redump: boolean, True for regenerating data
        embed_dim: int
        maxlen : int, max length of user interaction history
        train_neg_ratio: int, the number of negtive samples for on positve sample in train dataset
        test_neg_ratio: int, the number of negtive samples for on positve sample in validation/test datasets
        cold_start: int
        recurrent: bool, if to generate train dataset recurrently

    Returns:
        features_columns: dict, a description of features
        train, val, test: dict of data array

    """
    print('Get Data'.center(50, '='))
    train_neg_ratio = train_neg_ratio
    test_neg_ratio = test_neg_ratio
    cold_start = cold_start
    recurrent = recurrent

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

    if recurrent:
        addition += '_recurrent'

    # try to load date from pickle
    pickle_filename = f'{data_dir}{dataset}_{mode}_maxlen_{maxlen}' + \
                      f'_train_neg_{train_neg_ratio}_test_neg_{test_neg_ratio}' + \
                      f'_{cold_start}-core{addition}.pkl'

    if (redump is False) & (os.path.exists(pickle_filename)):
        print('Loading Data'.center(50, '='))
        with open(pickle_filename, 'rb') as file:
            feature_columns = pickle.load(file)
            train = pickle.load(file)
            val = pickle.load(file)
            test = pickle.load(file)
            summary = pickle.load(file)
            print('Load data from pickle: {}'.format(pickle_filename))
            print('Data Summary'.center(50, '='))
            for key, value in summary.items():
                print(f'{key}: {value}')
            for key, value in feature_columns.items():
                if 'embed_dim' in value:
                    if value['embed_dim'] != embed_dim:
                        value['embed_dim'] = embed_dim
                        print(f'Reset embed_dim as {embed_dim} for column {key}.')

    else:
        print('Generating Data'.center(50, '='))

        params['cold_start'] = cold_start

        # prepare dataset
        data, item_features = prepare_dataset_func(**params)
        # creat train, val, test
        feature_columns, train, val, test, summary = create_dataset(
            data, item_features,
            embed_dim=embed_dim, maxlen=maxlen,
            train_neg_ratio=train_neg_ratio, test_neg_ratio=test_neg_ratio,
            recurrent=recurrent)

        print('Data Summary'.center(50, '='))
        for key, value in summary.items():
            print(f'{key}: {value}')

        # save data
        with open(pickle_filename, 'wb') as file:
            pickle.dump(feature_columns, file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(train, file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(val, file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test, file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(summary, file, pickle.HIGHEST_PROTOCOL)
            print('Dump data to {}'.format(pickle_filename))

    print('Data Done'.center(50, '='))
    return(feature_columns, train, val, test)

if __name__ == '__main__':

    # test
    feature_columns, train, val, test = get_data(
        dataset='taobao', mode='train_500K',
        redump=False, embed_dim=100, maxlen=60, cold_start=5,
        train_neg_ratio=1, test_neg_ratio=1000,
        )

    print('Len of train, val and test:')
    for i in train, val, test:
        print(i['user_id'].shape)

    print('Shape of data in test')
    for c, i in test.items():
        print(c, ':', i.shape)
