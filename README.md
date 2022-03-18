# IMRec-tf2
This is the TensorFlow 2 implementation for the paper:
An Intention-aware Markov Chain based Method for Top-K Recommendation (link)

## Set Up
Run
```
pip install -r requirements.txt
```

## Dataset
### Taobao-500K
It is a sample with 500,000 interactions from the Taobao dataset ([link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1)). The dataset will be automatically generated for the first time you train the model. For preprocessing details, see `utils.py`.

## Model Training
To train our model in the Taobao-500K dataset, please run
```
python train.py --dataset taobao --mode train_500K --model IMRec
```

Experimental settings are defined in `config.py`. For easy use, some of the settings can be overwritten by passing command-line arguments. For example, to set the embedding dimension as 50, run

```
python train.py --dataset taobao --mode train_500K --model IMRec --embed_dim 50
```

## Key Arguments
Some key arguments:

`embed_dim`: the embedding dimension

`maxlen`: the max length of interaction sequence

`att_len`: the order of the Markov chain

`alpha`: the long-term and short-term preference weighting factor