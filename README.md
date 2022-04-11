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

The model is evaluated on the test datasets automatically at the end of each epoch. Check the experimental results as well as the saved models in the `log` folder.


## Key Arguments
Some key arguments:

`embed_dim`: the embedding dimension

`maxlen`: the max length of interaction sequence

`att_len`: the order of the Markov chain

`alpha`: the long-term and short-term preference weighting factor


## Model Checkpoint
We provide the best model of taobao-500K dataset for you to test, please download it from [link](https://pan.baidu.com/s/1nGQ4KZuOiYALuRZVNM_8DQ) (password: gusw). To load the model, run the following python scripts:

```python

import tensorflow as tf
from module import RecordLoss, RecordMetrics, HR, MRR

model_path = 'best_model/'  # the folder of the saved model
model = tf.keras.models.load_model(
	model_path,
	custom_objects={
	   'RecordLoss': RecordLoss,
	   'RecordMetrics': RecordMetrics,
	   'HR': HR,
	   'MRR': MRR,
	   })
```


## Acknowledgement
Many thanks to the work of Ziyao Geng ([link](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0)), for providing TensorFlow2 implementation of several recommendation algorithms.