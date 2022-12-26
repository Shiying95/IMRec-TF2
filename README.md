# IMRec-TF2
This is the TensorFlow 2 implementation for the paper:
An Intention-aware Markov Chain based Method for Top-K Recommendation ([link](https://ieeexplore.ieee.org/document/9997243))

Corrections to the paper:
1. Page 5, equation (13): <img src="http://chart.googleapis.com/chart?cht=tx&chl=argmin_{\Theta}\quad\quad\mathcal{L}_{Item}%2B\mathcal{L}_{Intention}%2B\mathcal{L}_{Reg}" style="border:none;">
2. Page 10, left column, the 4th line from the bottom: <img src="http://chart.googleapis.com/chart?cht=tx&chl=\mathcal{L}_{Item}=-\frac{1}{n}\sum_{j=1}^n(\log{(\sigma(\hat{p}_{u,t,i^u_t}))}%2B\log{(\sigma(1-\hat{p}_{u,t,{i'}^u_t}))})" style="border:none;">
3. Page 11, left column, line 10: There is also a larger performance loss in 5M datasets than in 500K datasets.
4. Page 11, left column, line 13: A likely reason is that Taobao and 5M datasets have much more items and sparser item-item transitions.


## Set Up
Run
```
pip install -r requirements.txt
```

## Dataset
### Taobao-500K
It is a sample with around 500,000 interactions from the Taobao dataset ([link](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1)). The dataset will be automatically generated for the first time you train the model. For preprocessing details, see `utils.py`.

## Model Training
To train our model in the Taobao-500K dataset, please run
```
python train.py --dataset taobao --mode train_500K --model IMRec
```

Experimental settings are defined in `config.py`. For easy use, some of the settings can be overwritten by passing command-line arguments. For example, to set the embedding dimension as 50, run

```
python train.py --dataset taobao --mode train_500K --model IMRec --embed_dim 50
```

The model is evaluated on the test datasets automatically at the end of each epoch. Check the experimental results in the `log` folder.


## Key Arguments
Some key arguments:

`embed_dim`: the embedding dimension

`maxlen`: the max length of interaction sequence

`att_len`: the order of the Markov chain

`alpha`: the long-term and short-term preference weighting factor

`without_il`: whether to disable the intention loss module

`bpr`: whether to use BPR loss

`time_threshold`: the time threshold of strict short-term definition. 0 for not setting the threshold

`item_intention`: whether to use (item, action) as intention


## Model Checkpoint
We provide the best model of the Taobao-500K dataset. To load the model, run:

```
python train.py --dataset taobao --mode train_500K --model IMRec --weights_dir ./best_model/best_epoch.ckpt
```

If you want to save checkpoints when training, run:
```
python train.py --dataset taobao --mode train_500K --model IMRec --save_weights 1
```
