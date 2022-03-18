from module import MyConfig
from model import IMRec

models = ['IMRec']
datasets = ['taobao']
modes = ['train_500K']

_model_config = MyConfig(models, datasets, modes)
_exp_config = MyConfig(models, datasets, modes)
_ds_config = MyConfig(models, datasets, modes)

_models = {}
for model in models:
    _models[model] = eval(model)

_model_config.update(
    model='IMRec',
    dataset='taobao',
    mode='train_500K',
    update_dict=dict(
        alpha=0.5,
        att_len=2,
        embed_reg=1e-6,
        activation='relu',
        )
    )

_ds_config.update(
    model='IMRec',
    dataset='taobao',
    mode='train_500K',
    update_dict=dict(
        train_neg_ratio=1,
        test_neg_ratio=1000,
        cold_start=5,
        embed_dim=100,
        maxlen=60,
        )
    )

_exp_config.update(
    model='IMRec',
    dataset='taobao',
    mode='train_500K',
    update_dict=dict(
        batch_size=512,
        learning_rate=0.003,
        )
    )
